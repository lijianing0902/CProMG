import math

import numpy as np
import mindspore as ms
from mindspore import ops, nn, Parameter
import mindspore.numpy as mnp


def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.shape[0], seq.shape[1], seq.shape[1]]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) 
    subsequence_mask = ms.Tensor(subsequence_mask).byte()
    return subsequence_mask  

class ScaledDotProductAttention(nn.Cell):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def construct(self, Q, K, V, attn_mask):

        hidden_channels = Q.size(-1)
        scores = ops.matmul(Q, K.transpose(-1, -2)) / np.sqrt(hidden_channels)
        scores.masked_fill(attn_mask, -1e9 if scores.dtype == ms.float32 else -1e4)
        attn = nn.Softmax(dim=-1)(scores)
        context = ops.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Cell):
    def __init__(self,hidden_channels,key_channels, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.W_Q = nn.Dense(hidden_channels, key_channels)
        self.W_K = nn.Dense(hidden_channels, key_channels)
        self.W_V = nn.Dense(hidden_channels, hidden_channels )
        self.linear = nn.Dense(hidden_channels, hidden_channels)
        self.layer_norm = nn.LayerNorm([hidden_channels])

    def construct(self, Q, K, V , attn_mask):
        
       
        residual, batch_size = Q, Q.shape[0]
        q_s = self.W_Q(Q).view(batch_size, self.num_heads, -1, self.key_channels//self.num_heads)
        k_s = self.W_K(K).view(batch_size, self.num_heads, -1, self.key_channels//self.num_heads)
        v_s = self.W_V(V).view(batch_size, self.num_heads, -1, self.hidden_channels//self.num_heads) 

        attn_mask = attn_mask.unsqueeze(1).tile((1, self.num_heads, 1, 1))


        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = (
            context.swapaxes(1, 2)
            .view(batch_size, -1, self.hidden_channels) 
        )

        output = self.linear(context)
        return self.layer_norm(output + residual) 


class PoswiseFeedForwardNet(nn.Cell):
    def __init__(self,hidden_channels):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=hidden_channels, out_channels=1024, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=1024, out_channels=hidden_channels, kernel_size=1)
        self.layer_norm = nn.LayerNorm([hidden_channels])

    def construct(self, inputs):
        residual = inputs 
        output = nn.ReLU()(self.conv1(inputs.swapaxes(1, 2)))
        output = self.conv2(output).swapaxes(1, 2)
        return self.layer_norm(output + residual)


def get_attn_pad_mask(seq_q, seq_k, pad_idx):
    batch_size, len_q = seq_q.shape
    batch_size, len_k = seq_k.shape
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.equal(pad_idx).unsqueeze(1)
    return pad_attn_mask.broadcast_to((batch_size, len_q, len_k))


class PositionalEncoding(nn.Cell):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = ops.zeros((max_len, d_model))
        position = mnp.arange(0, max_len, dtype=ms.float32).unsqueeze(1)
        div_term = ops.exp(
            mnp.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = ops.sin(position * div_term)
        pe[:, 1::2] = ops.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.pe = Parameter(pe, name="pe", requires_grad=False)

    def construct(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class DecoderLayer(nn.Cell):
    def __init__(self, config):
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

    def construct(self, dec_inputs, enc_outputs, dec_self_attn_mask , dec_enc_attn_mask):

        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs


class Decoder(nn.Cell):
    def __init__(self, config, num_props=None):
        super().__init__()
        self.config = config
        self.num_props = num_props
        self.mol_emb = nn.Embedding(len(config.smiVoc), config.hidden_channels, padding_idx=0)
        self.pos_emb = PositionalEncoding(config.hidden_channels)
        self.type_emb = nn.Embedding(2, config.hidden_channels)
        if self.num_props:
            self.prop_nn = nn.Dense(self.num_props, config.hidden_channels)
        self.layers = nn.CellList([DecoderLayer(config) for _ in range(config.num_interactions)])
    
    def construct(self, smiles_index, enc_outputs, enc_pad_mask, tgt_len, prop=None):
        b, t = smiles_index.shape
        dec_inputs = self.mol_emb(smiles_index)
        dec_inputs = self.pos_emb(dec_inputs.swapaxes(0, 1)).swapaxes(0, 1)
        if self.num_props:
            assert prop.shape[-1] == self.num_props
            type_embeddings = self.type_emb(ops.ones((b, t), dtype=ms.int32))
            dec_inputs = dec_inputs + type_embeddings
            type_embd = self.type_emb(ops.zeros((b, 1), dtype=ms.int32))
            p = self.prop_nn(prop.unsqueeze(1))
            p += type_embd
            dec_inputs = ops.concat([p, dec_inputs], 1)
            con = ops.ones(smiles_index.shape[0], 1)
            smiles_index = ops.concat([con, smiles_index], 1)
        if self.num_props:
            num = int(bool(self.num_props))
        else:
            num = 0
        dec_self_attn_pad_mask = get_attn_pad_mask(smiles_index, smiles_index, self.config.smiVoc.index('^'))
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(smiles_index)
        dec_self_attn_mask = ops.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = enc_pad_mask.expand(enc_pad_mask.shape[0], tgt_len + num, enc_pad_mask.shape[2])
        for layer in self.layers:
            dec_outputs = layer(dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_inputs = dec_outputs
        return dec_outputs
