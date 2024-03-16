import numpy as np
import mindspore as ms
from mindspore import ops, nn



class ScaledDotProductAttention(nn.Cell):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):

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

    def construct(self, Q, K, V, attn_mask):
        
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


class EncoderLayer(nn.Cell):
    def __init__(self,config):
        super(EncoderLayer, self).__init__()
        self.config = config
        self.enc_self_attn = MultiHeadAttention(
            hidden_channels=config.hidden_channels, 
            key_channels = config.key_channels, 
            num_heads = config.num_heads
        )
        self.pos_ffn = PoswiseFeedForwardNet(hidden_channels=config.hidden_channels)

    def construct(self, node_attr, attn_mask):
        msa_outputs = self.enc_self_attn(node_attr, node_attr, node_attr, attn_mask) 
        fnn_outputs = self.pos_ffn(msa_outputs)  

        return fnn_outputs


# AtomEncoder
class Encoder(nn.Cell):
    def __init__(self,config, feature_dim):
        super(Encoder, self).__init__()
        self.config = config
        self.feature_dim = feature_dim       
        self.emb = nn.Dense(feature_dim, config.hidden_channels) 
        self.layers = nn.CellList([EncoderLayer(config) for _ in range(config.num_interactions)]) 
        self.out = nn.Dense(config.hidden_channels,config.hidden_channels)
        self.layer_norm = nn.LayerNorm(config.hidden_channels)

    def construct(self, feature, feature_mask):
        node_attr = self.emb(feature)

        for layer in self.layers:
            fnn_outputs = layer(node_attr, feature_mask)
            node_attr = fnn_outputs
        
        # enc_pad_mask1 = ~enc_pad_mask1.unsqueeze(1)

        return node_attr
