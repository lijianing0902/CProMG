import torch
import torch.nn as nn

from .encoder import Encoder, Encoder2
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(self,config,protein_atom_feature_dim,num_props=None):
        super(Transformer, self).__init__()
        self.config = config
        self.num_props = num_props
        self.encoder = Encoder(config.encoder,protein_atom_feature_dim)
        self.encoder2 = Encoder2(config.encoder,20)
        self.decoder = Decoder(config.decoder,self.num_props)
        self.projection = nn.Linear(config.hidden_channels, len(config.decoder.smiVoc) , bias=False)

    def forward(self, node_attr, pos, batch,atom_laplacian, smiles_index, tgt_len, aa_node_attr, aa_pos, aa_batch,aa_laplacian, prop=None):
        enc_outputs1, enc_pad_mask1, msa_outputs = self.encoder(node_attr, pos, batch,atom_laplacian)
        enc_outputs2, enc_pad_mask2 = self.encoder2(aa_node_attr, aa_pos, aa_batch,aa_laplacian, enc_pad_mask1, msa_outputs)
        enc_outputs = torch.cat([enc_outputs1,enc_outputs2],dim=1)
        enc_pad_mask = torch.cat([enc_pad_mask1,enc_pad_mask2],dim=2)
        dec_outputs = self.decoder(smiles_index, enc_outputs, enc_pad_mask,tgt_len,prop)
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]

        if self.num_props:
            num = int(bool(self.num_props))
        else:
            num = 0
        dec_logits = dec_logits[:, num:, :]

        return dec_logits.reshape(-1, dec_logits.size(-1))
