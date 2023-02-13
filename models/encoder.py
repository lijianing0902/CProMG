import os
from math import pi as PI

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Conv1d, LayerNorm, ReLU, BatchNorm1d, Softmax
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import get_laplacian,to_dense_batch,to_undirected
from torch_scatter import scatter_sum, scatter_softmax

from .common import GaussianSmearing, ShiftedSoftplus


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class ScaledDotProductAttention(Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9) 
        attn = Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention2(Module):
    def __init__(self,hidden_channels,key_channels, num_heads):
        super(MultiHeadAttention2, self).__init__()
        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.W_Q = Linear(hidden_channels, key_channels )
        self.W_K = Linear(hidden_channels, key_channels )
        self.W_V = Linear(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, hidden_channels)
        self.layer_norm = LayerNorm(hidden_channels)

    def forward(self, Q, K, V , attn_mask):
        
        residual, batch_size = Q, Q.size(0) 
        
        q_s = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.key_channels//self.num_heads).transpose(1,2) 
        k_s = self.W_K(K).view(batch_size, -1, self.num_heads, self.key_channels//self.num_heads).transpose(1,2)  
        v_s = self.W_V(V).view(batch_size, -1, self.num_heads, self.hidden_channels//self.num_heads).transpose(1,2)  

       
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)


        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_channels) 
        output = self.linear(context)
        return self.layer_norm(output + residual) 



class MultiHeadAttention(Module):
    def __init__(self,hidden_channels, edge_channels, key_channels, num_heads=1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_channels % num_heads == 0 
        assert key_channels % num_heads == 0

        self.num_heads = num_heads
        self.k_lin = Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False)
        self.q_lin = Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False)
        self.v_lin = Conv1d(hidden_channels, hidden_channels, 1, groups=num_heads, bias=False)

        self.weight_k_net = Sequential(
            Linear(edge_channels, key_channels//num_heads),
            ShiftedSoftplus(),
            Linear(key_channels//num_heads, key_channels//num_heads),
        )
        self.weight_k_lin = Linear(key_channels//num_heads, key_channels//num_heads)

        self.weight_v_net = Sequential(
            Linear(edge_channels, hidden_channels//num_heads),
            ShiftedSoftplus(),
            Linear(hidden_channels//num_heads, hidden_channels//num_heads),
        )
        self.weight_v_lin = Linear(hidden_channels//num_heads, hidden_channels//num_heads)
        self.centroid_lin = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.out_transform = Linear(hidden_channels, hidden_channels)
        self.layer_norm = LayerNorm(hidden_channels)
        # self.batch_norm = BatchNorm1d(hidden_channels)

    def forward(self, node_attr, edge_index, edge_attr):

        N = node_attr.size(0)
        row, col = edge_index  

        # Project to multiple key, query and value spaces
        h_keys = self.k_lin(node_attr.unsqueeze(-1)).view(N, self.num_heads, -1)    
        h_queries = self.q_lin(node_attr.unsqueeze(-1)).view(N, self.num_heads, -1) 
        h_values = self.v_lin(node_attr.unsqueeze(-1)).view(N, self.num_heads, -1)  

        # Compute keys and queries
        W_k = self.weight_k_net(edge_attr) 
        keys_j = self.weight_k_lin(W_k.unsqueeze(1) * h_keys[col])  
        queries_i = h_queries[row]   

        # Compute attention weights (alphas)
        qk_ij = ((queries_i * keys_j).sum(-1))/ np.sqrt(keys_j.size(-1)) 
        alpha = scatter_softmax(qk_ij, row, dim=0)

        # Compose messages
        W_v = self.weight_v_net(edge_attr)  # (E, H_per_head)
        msg_j = self.weight_v_lin(W_v.unsqueeze(1) * h_values[col]) 
        msg_j = alpha.unsqueeze(-1) * msg_j   

         # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N).view(N, -1) 
        out = self.centroid_lin(node_attr) + aggr_msg

        out = self.out_transform(self.act(out))
        return self.layer_norm(out)

class PoswiseFeedForwardNet(Module):
    def __init__(self,hidden_channels):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = Conv1d(in_channels=hidden_channels, out_channels=1024, kernel_size=1)
        self.conv2 = Conv1d(in_channels=1024, out_channels=hidden_channels, kernel_size=1)
        self.layer_norm = LayerNorm(hidden_channels)
        self.batch_norm = BatchNorm1d(hidden_channels)

    def forward(self, inputs):
        residual = inputs 
        inputs = inputs.unsqueeze(0)
        output = ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = output.squeeze(0)
        return self.layer_norm(output + residual)

class EncoderLayer(Module):
    def __init__(self,config):
        super(EncoderLayer, self).__init__()
        self.config = config
        self.enc_self_attn = MultiHeadAttention(
            hidden_channels=config.hidden_channels, 
            edge_channels = config.edge_channels, 
            key_channels = config.key_channels, 
            num_heads = config.num_heads
        )
        self.pos_ffn = PoswiseFeedForwardNet(hidden_channels=config.hidden_channels)

    def forward(self, node_attr, edge_index, edge_attr):
        msa_outputs = self.enc_self_attn(node_attr, edge_index, edge_attr) 
        fnn_outputs = self.pos_ffn(msa_outputs)  
        return msa_outputs, fnn_outputs

class EncoderLayer2(Module):
    def __init__(self,config):
        super(EncoderLayer2, self).__init__()
        self.config = config
        self.enc_self_attn = MultiHeadAttention(
            hidden_channels=config.hidden_channels, 
            edge_channels = config.edge_channels, 
            key_channels = config.key_channels, 
            num_heads = config.num_heads,
        )
        self.proj = Linear(config.hidden_channels,config.hidden_channels)
        self.cross_attn = MultiHeadAttention2(
            hidden_channels = config.hidden_channels,
            key_channels = config.key_channels, 
            num_heads = config.num_heads,
        )
        self.layer_norm = LayerNorm(config.hidden_channels)
        self.pos_ffn = PoswiseFeedForwardNet(hidden_channels=config.hidden_channels)

    def forward(self, node_attr, edge_index, edge_attr, idx, atom_msa_outputs, atom_mask, batch):
        msa_outputs = self.enc_self_attn(node_attr, edge_index, edge_attr) 
        if idx==2 or idx==5:
            atom_msa_output = self.proj(atom_msa_outputs[idx])
            msa_outputs1, msa_outputs1_mask = to_dense_batch(msa_outputs, batch)  
            cross_outputs1 = self.cross_attn(msa_outputs1, atom_msa_output, atom_msa_output, atom_mask)
            cross_outputs = torch.masked_select(cross_outputs1.view(-1,cross_outputs1.size(-1)),msa_outputs1_mask.view(-1).unsqueeze(-1)).view(-1,cross_outputs1.size(-1))
            msa_outputs = self.layer_norm(msa_outputs + cross_outputs)
        fnn_outputs = self.pos_ffn(msa_outputs)   
        return fnn_outputs

# AtomEncoder
class Encoder(Module):
    def __init__(self,config,protein_atom_feature_dim):
        super(Encoder, self).__init__()
        self.config = config
        self.protein_atom_feature_dim = protein_atom_feature_dim       
        self.protein_atom_emb = Linear(protein_atom_feature_dim, config.hidden_channels) 
        self.laplacian_emb = Linear(config.lap_dim, config.hidden_channels)
        self.layers = ModuleList([EncoderLayer(config) for _ in range(config.num_interactions)]) 
        self.distance_expansion = GaussianSmearing(stop=15, num_gaussians=config.edge_channels)
        self.out = Linear(config.hidden_channels,config.hidden_channels)
        self.layer_norm = LayerNorm(config.hidden_channels)
    def forward(self, protein_atom_feature, pos, batch, atom_laplacian):
        node_attr = self.protein_atom_emb(protein_atom_feature)
        atom_laplacian = self.laplacian_emb(atom_laplacian)
        node_attr = node_attr + atom_laplacian
        edge_index_di = knn_graph(pos, self.config.knn, batch=batch, flow='target_to_source')
        # edge_index = radius_graph(pos,4.5 ,batch=batch, flow='target_to_source')
        edge_length = torch.norm(pos[edge_index_di[0]] - pos[edge_index_di[1]], dim=1)
        edge_index,edge_attr = to_undirected(edge_index_di,edge_length,reduce='mean')
        edge_attr = self.distance_expansion(edge_attr)  
        edge_index,edge_attr = get_laplacian(edge_index,edge_attr)
                   
        msa_outputs1 = []
        for layer in self.layers:
            msa_outputs, fnn_outputs = layer(node_attr, edge_index, edge_attr)
            node_attr = fnn_outputs
            msa_outputs, msa_pad_mask = to_dense_batch(msa_outputs, batch)
            msa_outputs1.append(msa_outputs)
        enc_outputs1, enc_pad_mask1 = to_dense_batch(node_attr, batch)
        enc_pad_mask1 = ~enc_pad_mask1.unsqueeze(1)

        return enc_outputs1, enc_pad_mask1, msa_outputs1

#AAencoder
class Encoder2(Module):
    def __init__(self,config,aa_feature_dim):
        super(Encoder2, self).__init__()
        self.config = config
        self.aa_feature_dim = aa_feature_dim       
        self.aa_emb = Linear(aa_feature_dim, config.hidden_channels) 
        self.laplacian_emb = Linear(config.lap_dim, config.hidden_channels)
        self.layers = ModuleList([EncoderLayer2(config) for _ in range(config.num_interactions)]) 
        self.distance_expansion = GaussianSmearing(stop=25, num_gaussians=config.edge_channels)
        self.out = Linear(config.hidden_channels,config.hidden_channels)
        self.layer_norm = LayerNorm(config.hidden_channels)
    def forward(self, aa_feature, aa_pos, aa_batch,aa_laplacian, atom_mask, atom_msa_outputs):
        node_attr = self.aa_emb(aa_feature)
        aa_laplacian = self.laplacian_emb(aa_laplacian)
        node_attr = node_attr + aa_laplacian
        edge_index = knn_graph(aa_pos, 30, batch=aa_batch, flow='target_to_source')
        # edge_index = radius_graph(aa_pos,10 ,batch=aa_batch, flow='target_to_source')
        edge_length = torch.norm(aa_pos[edge_index[0]] - aa_pos[edge_index[1]], dim=1)
        edge_index,edge_attr = to_undirected(edge_index,edge_length,reduce='mean')
        edge_attr = self.distance_expansion(edge_attr)
        edge_index,edge_attr = get_laplacian(edge_index,edge_attr)
        
        for idx, layer in enumerate(self.layers):
            enc_outputs = layer(node_attr, edge_index, edge_attr, idx, atom_msa_outputs, atom_mask, aa_batch)
            node_attr = enc_outputs
        enc_outputs1, enc_pad_mask1 = to_dense_batch(node_attr, aa_batch)     
        enc_pad_mask1 = ~enc_pad_mask1.unsqueeze(1)

        return enc_outputs1, enc_pad_mask1