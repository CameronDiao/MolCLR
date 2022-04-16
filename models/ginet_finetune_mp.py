from typing import List, Optional, Tuple, Type
import math

import torch
from torch import nn
from torch import Tensor
from torch.nn import LayerNorm, Linear

import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention

num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

class MAB(torch.nn.Module):
    r"""Multihead-Attention Block."""
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int,
                 Conv: Optional[Type]  = None, layer_norm: bool = False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.layer_norm = layer_norm

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

    def reset_parameters(self):
        self.fc_q.reset_parameters()
        self.layer_k.reset_parameters()
        self.layer_v.reset_parameters()
        if self.layer_norm:
            self.ln0.reset_parameters()
            self.ln1.reset_parameters()
        self.fc_o.reset_parameters()
        pass

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:

        Q = self.fc_q(Q)

        if graph is not None:
            x, edge_index, batch = graph
            K, V = self.layer_k(x, edge_index), self.layer_v(x, edge_index)
            K, _ = to_dense_batch(K, batch)
            V, _ = to_dense_batch(V, batch)
        else:
            K, V = self.layer_k(K), self.layer_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1, 2))
            attention_score = attention_score / math.sqrt(self.dim_V)
            A = torch.softmax(mask + attention_score, 1)
        else:
            A = torch.softmax(
                Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 1)

        out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        if self.layer_norm:
            out = self.ln0(out)

        out = out + self.fc_o(out).relu()

        if self.layer_norm:
            out = self.ln1(out)

        return out


class SAB(torch.nn.Module):
    r"""Self-Attention Block."""
    def __init__(self, in_channels: int, out_channels: int, num_heads: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.mab = MAB(in_channels, in_channels, out_channels, num_heads,
                       Conv=Conv, layer_norm=layer_norm)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.mab(x, x, graph, mask)


class PMA(torch.nn.Module):
    r"""Graph pooling with Multihead-Attention."""
    def __init__(self, channels: int, num_heads: int, num_seeds: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, channels))
        self.mab = MAB(channels, channels, channels, num_heads, Conv=Conv,
                       layer_norm=layer_norm)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.S)
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.mab(self.S.repeat(x.size(0), 1, 1), x, graph, mask)


class GraphMultisetTransformer(torch.nn.Module):
    r"""The global Graph Multiset Transformer pooling operator from the
    `"Accurate Learning of Graph Representations
    with Graph Multiset Pooling" <https://arxiv.org/abs/2102.11533>`_ paper.

    The Graph Multiset Transformer clusters nodes of the entire graph via
    attention-based pooling operations (:obj:`"GMPool_G"` or
    :obj:`"GMPool_I"`).
    In addition, self-attention (:obj:`"SelfAtt"`) can be used to calculate
    the inter-relationships among nodes.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        conv (Type, optional): A graph neural network layer
            for calculating hidden representations of nodes for
            :obj:`"GMPool_G"` (one of
            :class:`~torch_geometric.nn.conv.GCNConv`,
            :class:`~torch_geometric.nn.conv.GraphConv` or
            :class:`~torch_geometric.nn.conv.GATConv`).
            (default: :class:`~torch_geometric.nn.conv.GCNConv`)
        num_nodes (int, optional): The number of average
            or maximum nodes. (default: :obj:`300`)
        pooling_ratio (float, optional): Graph pooling ratio
            for each pooling. (default: :obj:`0.25`)
        pool_sequences ([str], optional): A sequence of pooling layers
            consisting of Graph Multiset Transformer submodules (one of
            :obj:`["GMPool_I"]`,
            :obj:`["GMPool_G"]`,
            :obj:`["GMPool_G", "GMPool_I"]`,
            :obj:`["GMPool_G", "SelfAtt", "GMPool_I"]` or
            :obj:`["GMPool_G", "SelfAtt", "SelfAtt", "GMPool_I"]`).
            (default: :obj:`["GMPool_G", "SelfAtt", "GMPool_I"]`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`4`)
        layer_norm (bool, optional): If set to :obj:`True`, will make use of
            layer normalization. (default: :obj:`False`)

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          batch vector :math:`(|\mathcal{V}|)`,
          edge indices :math:`(2, |\mathcal{E}|)` *(optional)*
        - **output:** graph features :math:`(|\mathcal{G}|, F_{out})` where
          :math:`|\mathcal{G}|` denotes the number of graphs in the batch
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        Conv: Optional[Type] = None,
        num_nodes: int = 300,
        pooling_ratio: float = 0.25,
        pool_sequences: List[str] = ['GMPool_G', 'SelfAtt', 'GMPool_I'],
        num_heads: int = 4,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.Conv = Conv or GCNConv
        self.num_nodes = num_nodes
        self.pooling_ratio = pooling_ratio
        self.pool_sequences = pool_sequences
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

        self.pools = torch.nn.ModuleList()
        num_out_nodes = math.ceil(num_nodes * pooling_ratio)
        for i, pool_type in enumerate(pool_sequences):
            if pool_type not in ['GMPool_G', 'GMPool_I', 'SelfAtt']:
                raise ValueError("Elements in 'pool_sequences' should be one "
                                 "of 'GMPool_G', 'GMPool_I', or 'SelfAtt'")

            if i == len(pool_sequences) - 1:
                num_out_nodes = 1

            if pool_type == 'GMPool_G':
                self.pools.append(
                    PMA(hidden_channels, num_heads, num_out_nodes,
                        Conv=self.Conv, layer_norm=layer_norm))
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)

            elif pool_type == 'GMPool_I':
                self.pools.append(
                    PMA(hidden_channels, num_heads, num_out_nodes, Conv=None,
                        layer_norm=layer_norm))
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)

            elif pool_type == 'SelfAtt':
                self.pools.append(
                    SAB(hidden_channels, hidden_channels, num_heads, Conv=None,
                        layer_norm=layer_norm))

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()


    def forward(self, x: Tensor, batch: Tensor,
                edge_index: Optional[Tensor] = None) -> Tensor:
        """"""
        x = self.lin1(x)
        batch_x, mask = to_dense_batch(x, batch)
        mask = (~mask).unsqueeze(1).to(dtype=x.dtype) * -1e9

        for i, (name, pool) in enumerate(zip(self.pool_sequences, self.pools)):
            graph = (x, edge_index, batch) if name == 'GMPool_G' else None
            batch_x = pool(batch_x, graph, mask)
            mask = None

        return self.lin2(batch_x.squeeze(1))


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, pool_sequences={self.pool_sequences})')

class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + \
            self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, 
        num_motifs, task='classification', num_layer=5, emb_dim=300, feat_dim=512, 
        drop_ratio=0, pool='mean', pred_n_layer=2, pred_act='softplus'
    ):
        super(GINet, self).__init__()
        self.num_motifs = num_motifs
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.task = task

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        #self.motif_embedding = nn.Embedding(num_motifs, feat_dim)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)
        if self.task == 'classification':
            out_dim = 2
        elif self.task == 'regression':
            out_dim = 1
        
        #self.motif_lin = nn.Linear(self.feat_dim, self.feat_dim//2)
        #nn.init.xavier_uniform_(self.motif_lin.weight.data)

        #self.motif_pool = GlobalAttention(gate_nn=nn.Sequential(nn.Linear(self.feat_dim, 1)),
        #                                  nn=nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim//2)))

        #self.motif_pool = GraphMultisetTransformer(in_channels=self.feat_dim, hidden_channels=self.feat_dim,
        #                                           out_channels=self.feat_dim, pool_sequences=["GMPool_I"])
        
        #self.motif_trans = SAB(in_channels=self.feat_dim, out_channels=self.feat_dim, num_heads=4)
        self.motif_pool = PMA(channels=self.feat_dim, num_heads=4, num_seeds=1, layer_norm=True)

        self.pred_n_layer = max(1, pred_n_layer)

        if pred_act == 'relu':
            pred_head = [
                nn.Linear(2 * self.feat_dim, self.feat_dim//2), 
                nn.ReLU(inplace=True)
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend([
                    nn.Linear(self.feat_dim//2, self.feat_dim//2), 
                    nn.ReLU(inplace=True),
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

    #def init_motif_emb(self, init):
    #    with torch.no_grad():
    #        self.motif_embedding.weight.data = nn.Parameter(init)
    
    def forward(self, data, mol_idx, motif_samples):
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

        hp = torch.cat((motif_samples, h), dim=0)
        #hp = self.motif_lin(hp)
        batch, mask = to_dense_batch(hp, mol_idx)
        mask = (~mask).unsqueeze(1).to(dtype=hp.dtype) * -1e9
        batch = self.motif_pool(batch, None, mask)
        #mask = None
        #batch = self.motif_trans(batch, None, mask)
        hp = batch.squeeze(1)
        #hp = self.motif_lin(hp)
        
        #h1 = batch[:, 0, :]
        #h2 = batch[:, 1, :]
        #h1 = torch.cat((h, h1), dim=1)
        #h2 = torch.cat((h, h2), dim=1)
        
        #p = torch.cat((self.pred_head(h1), self.pred_head(h2)), dim=1)

        #return h, p
        
        hp = torch.cat((h, hp), dim=1)
        return h, self.pred_head(hp)

        #h = torch.cat((h, hp), dim=1)

        #return h, p 

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
