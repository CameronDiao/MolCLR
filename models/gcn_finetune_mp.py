from typing import List, Optional, Tuple, Type
import math

import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch.nn.utils import weight_norm

import torch_sparse
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_scatter import scatter
from torch_scatter import scatter_add

from torch_geometric.data.batch import Batch
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils.num_nodes import maybe_num_nodes


num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

def _weight_reset(block):
    try:
        block.reset_parameters()
    except:
        if isinstance(block, nn.Sequential):
            for layer in block:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()

class MAB(torch.nn.Module):
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, dropout: float = 0.0,
                 Conv: Optional[Type]  = None, layer_norm: bool = False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.dropout = dropout

        self.fc_q = Linear(dim_Q, dim_V)

        if Conv is None:
            self.layer_k = Linear(dim_K, dim_V)
            self.layer_v = Linear(dim_K, dim_V)
        else:
            self.layer_k = Conv(dim_K, dim_V)
            self.layer_v = Conv(dim_K, dim_V)

        if layer_norm:
            self.ln0 = LayerNorm(dim_V)
            self.ln1 = LayerNorm(dim_V)

        self.fc_o = Linear(dim_V, dim_V)
        self.ffn = nn.Sequential(
                nn.Linear(dim_V, dim_V),
                nn.ReLU(inplace=True),
                nn.Linear(dim_V, dim_V)
        )

    def reset_parameters(self):
        _weight_reset(self.fc_q)
        _weight_reset(self.layer_k)
        _weight_reset(self.layer_v)
        if self.layer_norm:
            _weight_reset(self.ln0)
            _weight_reset(self.ln1)
        _weight_reset(self.fc_o)
        _weight_reset(self.ffn)
        pass

    def forward(
            self,
            Q: Tensor,
            K: Tensor,
            graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        num_queries = Q.shape[1]
        num_values = K.shape[1]
        Qn = self.fc_q(Q)

        if graph is not None:
            x, edge_index, batch = graph
            Kn, Vn = self.layer_k(x, edge_index), self.layer_v(x, edge_index)
            Kn, _ = to_dense_batch(Kn, batch)
            Vn, _ = to_dense_batch(Vn, batch)
        else:
            Kn, Vn = self.layer_k(K), self.layer_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Qn.split(dim_split, 2), dim=0)
        K_ = torch.cat(Kn.split(dim_split, 2), dim=0)
        V_ = torch.cat(Vn.split(dim_split, 2), dim=0)

        if mask is not None:
            mask = mask.repeat(self.num_heads, num_queries, 1)
            max_neg_value = -torch.finfo(Q_.dtype).max
            attention_score = Q_.bmm(K_.transpose(1, 2))
            attention_score = attention_score / math.sqrt(self.dim_V)
            attention_score.masked_fill_(~mask, max_neg_value)
            A = torch.softmax(attention_score, 1)
        else:
            A = torch.softmax(
                    Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 1)

        out = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        out = Q + F.dropout(self.fc_o(out), self.dropout, training=self.training)

        if self.layer_norm:
            out = self.ln0(out)

        out = out + F.dropout(self.ffn(out), self.dropout, training=self.training)

        if self.layer_norm:
            out = self.ln1(out)

        return out


def gcn_norm(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()
        self.emb_dim = emb_dim
        self.aggr = aggr

        self.weight = Parameter(torch.Tensor(emb_dim, emb_dim))
        self.bias = Parameter(torch.Tensor(emb_dim))
        self.reset_parameters()

        self.edge_embedding1 = nn.Embedding(num_bond_type, 1)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, 1)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def reset_parameters(self):
        # glorot(self.weight)
        # zeros(self.bias)
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        edge_index, __ = gcn_norm(edge_index)

        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_attr):
        # return x_j if edge_attr is None else edge_attr.view(-1, 1) * x_j
        return x_j if edge_attr is None else edge_attr + x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)


class GCN(nn.Module):
    def __init__(self, num_motifs, task='classification', num_layer=5, emb_dim=300, feat_dim=512, 
            drop_ratio=0, enc_dropout = 0, tfm_dropout = 0, dec_dropout=0, 
            enc_ln = True, tfm_ln = False, conc_ln = False,
            pool='mean', pred_n_layer=2, n_heads = 4, pred_act='softplus'):
        super(GCN, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.enc_dropout = enc_dropout
        self.tfm_dropout = tfm_dropout
        self.dec_dropout = dec_dropout
        self.enc_ln = enc_ln
        self.tfm_ln = tfm_ln
        self.conc_ln = conc_ln
        self.task = task

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GCNConv(emb_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')
        
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        if self.task == 'classification':
            out_dim = 2
        elif self.task == 'regression':
            out_dim = 1

        self.clique_embedding = nn.Embedding(num_motifs, self.feat_dim)

        self.motif_pool = MAB(self.feat_dim, self.feat_dim, self.feat_dim, num_heads=n_heads, dropout=self.tfm_dropout,
                layer_norm=self.tfm_ln)
        self.motif_pool.reset_parameters()

        if self.enc_ln:
            self.motif_norm1 = LayerNorm(self.feat_dim)
            _weight_reset(self.motif_norm1)

        self.motif_enc = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim)
        )
        _weight_reset(self.motif_enc)
        self.motif_dec = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim)
        )
        _weight_reset(self.motif_dec)

        if self.conc_ln:
            self.conc_norm1 = LayerNorm(2 * self.feat_dim)
            _weight_reset(self.conc_norm1)

        self.pred_n_layer = max(1, pred_n_layer)

        if pred_act == 'relu':
            pred_head = [
                    nn.Linear(2 * self.feat_dim, self.feat_dim//2),
                    nn.ReLU(inplace=True)
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.feat_dim//2, self.feat_dim//2),
                    nn.ReLU(inplace=True)
                ])
        elif pred_act == 'softplus':
            pred_head = [
                    nn.Linear(2 * self.feat_dim, self.feat_dim//2),
                    nn.Softplus()
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.feat_dim//2, self.feat_dim//2),
                    nn.Softplus()
                ])
        else:
            raise ValueError('Undefined activation function')
        
        pred_head.append(nn.Linear(self.feat_dim//2, out_dim))
        self.pred_head = nn.Sequential(*pred_head)

    def init_clique_emb(self, init):
        with torch.no_grad():
            self.clique_embedding.weight.data.copy_(init)

    def get_clique_emb(self):
        for name, param in self.named_parameters():
            if "clique_embedding" in name:
                return param

    def forward(self, data, mol_idx, clique_idx):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)

        hp = self.clique_embedding(clique_idx)
        batch, mask = to_dense_batch(hp, mol_idx)
        mask = mask.unsqueeze(1)
        batch = self.motif_enc(batch)
        if self.enc_ln:
            batch = self.motif_norm1(batch)
        batch = F.dropout(batch, self.enc_dropout, training=self.training)
        batch = self.motif_pool(h.detach().unsqueeze(1), batch, None, mask)
        batch = self.motif_dec(batch)
        batch = F.dropout(batch, self.dec_dropout, training=self.training)
        hp = batch.squeeze(1)

        hp = torch.cat((h, hp), dim=1)
        if self.conc_ln:
            hp = self.conc_norm1(hp)
        else:
            hp = F.normalize(hp, dim=1)

        hp = self.pred_head(hp)

        return h, hp

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

if __name__ == "__main__":
    model = GCN()
    print(model)
