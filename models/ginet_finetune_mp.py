import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from dgl.nn.pytorch.glob import SetAttentionBlock, PMALayer

num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

class SetTransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_head, d_ff, n_layers, k, dropouth=0., dropouta=0.):
        super(SetTransformerDecoder, self).__init__()
        self.n_layers=n_layers
        self.k = k
        self.d_model = d_model
        self.pma = PMALayer(k, d_model, num_heads, d_head, d_ff,
                                dropouth=dropouth, dropouta=dropouta)
        layers = []
        for _ in range(n_layers):
            layers.append(
                SetAttentionBlock(d_model, num_heads, d_head, d_ff,
                                  dropouth=dropouth, dropouta=dropouta))
        self.layers = nn.ModuleList(layers)

    def forward(self, data, feat, mol_idx):
        len_pma = list(mol_idx)
        len_sab = [self.k] * data.num_graphs
        feat = self.pma(feat, len_pma)
        for layer in self.layers:
            feat = layer(feat, len_sab)
        return feat[::2, :], feat[1::2, :] 

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

        self.motif_embedding = nn.Embedding(num_motifs, feat_dim)

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

        #self.motif_pool = PMALayer(k=2, d_model=self.feat_dim, num_heads=2, d_head=self.feat_dim,
        #                           d_ff=self.feat_dim)

        self.motif_trans = SetTransformerDecoder(d_model=self.feat_dim, num_heads=2, d_head=self.feat_dim,
                                                 d_ff=self.feat_dim, n_layers=1, k=2)

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

    def init_motif_emb(self, init):
        with torch.no_grad():
            self.motif_embedding.weight.data = nn.Parameter(init)
    
    def forward(self, data, mol_idx, shuffle_idx, clique_idx):
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

        hp = self.motif_embedding(clique_idx)
        hp = torch.cat((hp, h), dim=0).index_select(0, shuffle_idx)
        h1, h2 = self.motif_trans(data, hp, mol_idx)
        h1 = torch.cat((h, h1), dim=1)
        h2 = torch.cat((h, h2), dim=1)
        
        p = torch.cat((self.pred_head(h1), self.pred_head(h2)), dim=1)

        return h, p
        #hp = self.motif_pool(hp, list(mol_idx))
        #hp = self.motif_lin(hp)

        #h = torch.cat((h, hp), dim=1)

        #return h, self.pred_head(h)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
