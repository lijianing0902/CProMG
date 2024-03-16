import numpy as np
import mindspore as ms
from mindspore import ops, nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Cell):
    def __init__(self,config,atom_dim,aa_dim,num_props=None):
        super(Transformer, self).__init__()
        self.config = config
        self.num_props = num_props
        self.encoder = Encoder(config.encoder, atom_dim)
        self.encoder2 = Encoder(config.encoder, aa_dim)
        self.decoder = Decoder(config.decoder,self.num_props)
        self.projection = nn.Dense(config.hidden_channels, len(config.decoder.smiVoc) , bias=False)

    def forward(self, atom_attr, aa_attr, atom_mask, aa_mask, smiles_index, tgt_len, prop=None):
        atom_outputs = self.encoder(atom_attr, atom_mask)
        aa_outputs = self.encoder2(aa_attr, aa_mask)
        enc_outputs = ops.cat([atom_outputs, aa_outputs], 1)
        enc_pad_mask = ops.cat([atom_mask,aa_mask], 1)
        dec_outputs = self.decoder(smiles_index, enc_outputs, enc_pad_mask,tgt_len,prop)
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]

        if self.num_props:
            num = int(bool(self.num_props))
        else:
            num = 0
        dec_logits = dec_logits[:, num:, :]

        return dec_logits.view(-1, dec_logits.shape[-1])
