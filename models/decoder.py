import os
import math

import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Conv1d, LayerNorm, ReLU,Embedding,Softmax,Dropout

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) 
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  

class ScaledDotProductAttention(Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):

        hidden_channels = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(hidden_channels)
        scores.masked_fill_(attn_mask, -1e9)
        attn = Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(Module):
    def __init__(self,hidden_channels,key_channels, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.W_Q = Linear(hidden_channels, key_channels)
        self.W_K = Linear(hidden_channels, key_channels)
        self.W_V = Linear(hidden_channels, hidden_channels )
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


class PoswiseFeedForwardNet(Module):
    def __init__(self,hidden_channels):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = Conv1d(in_channels=hidden_channels, out_channels=1024, kernel_size=1)
        self.conv2 = Conv1d(in_channels=1024, out_channels=hidden_channels, kernel_size=1)
        self.layer_norm = LayerNorm(hidden_channels)

    def forward(self, inputs):
        residual = inputs 
        output = ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


def get_attn_pad_mask(seq_q, seq_k,pad_id):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(pad_id).unsqueeze(1)  
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class PositionalEncoding(Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class DecoderLayer(Module):
    def __init__(self,config):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            hidden_channels = config.hidden_channels,
            key_channels = config.key_channels, 
            num_heads = config.num_heads
        )
        self.dec_enc_attn = MultiHeadAttention(
            hidden_channels = config.hidden_channels,
            key_channels = config.key_channels, 
            num_heads = config.num_heads
        )
        self.pos_ffn = PoswiseFeedForwardNet(hidden_channels = config.hidden_channels)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask , dec_enc_attn_mask):

        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs


class Decoder(Module):
    def __init__(self, config, num_props=None):
        super(Decoder, self).__init__()
        self.config = config
        self.num_props = num_props
        self.mol_emb = Embedding(len(config.smiVoc), config.hidden_channels, 0)
        self.pos_emb = PositionalEncoding(config.hidden_channels)
        self.type_emb = Embedding(2, config.hidden_channels)
        if self.num_props:
            self.prop_nn = Linear(self.num_props, config.hidden_channels)

        self.layers = ModuleList([DecoderLayer(config) for _ in range(config.num_interactions)])

    def forward(self ,smiles_index, enc_outputs, enc_pad_mask, tgt_len,prop = None): 

        b, t = smiles_index.size()
        dec_inputs = self.mol_emb(smiles_index)
        dec_inputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1) 
        
        if self.num_props:
            assert prop.shape[-1] == self.num_props  
            type_embeddings = self.type_emb(torch.ones((b,t), dtype = torch.long).to(device))
            dec_inputs = dec_inputs + type_embeddings
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long).to(device))
            p = self.prop_nn(prop.unsqueeze(1))  
            p += type_embd
            dec_inputs = torch.cat([p, dec_inputs], 1)

            con = torch.ones(smiles_index.shape[0],1).to(device)
            smiles_index = torch.cat([con,smiles_index],1)        
        if self.num_props:
            num = int(bool(self.num_props))
        else:
            num = 0

        dec_self_attn_pad_mask = get_attn_pad_mask(smiles_index, smiles_index,self.config.smiVoc.index('^'))

        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(smiles_index).to(device)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = enc_pad_mask.expand(enc_pad_mask.size(0), tgt_len+num, enc_pad_mask.size(2)) # batch_size x len_q x len_k

        for layer in self.layers:
            dec_outputs = layer(dec_inputs, enc_outputs, dec_self_attn_mask , dec_enc_attn_mask)
            dec_inputs = dec_outputs    

        return dec_outputs


