import torch
import torch.nn as nn

from .encoder import Encoder, Encoder2
from .decoder import Decoder

#python models/TRM.py /home/shilab/ljn/3D/3D-Generative-SBDD-main/configs/main.yml
# chars = [" ", "7", "6", "o", "]", "3", "s", "(", "-", "S", "/", "B", "4", "[", ")", "#", "I", "l", "O", 
#             "H", "c", "1", "@", "=", "n", "P", "8", "C", "2", "F", "5", "r", "N", "+", "\\","A","Z","p"]
class Transformer(nn.Module):
    def __init__(self,config,protein_atom_feature_dim,num_props=None):
        super(Transformer, self).__init__()
        self.config = config
        self.num_props = num_props
        self.encoder = Encoder(config.encoder,protein_atom_feature_dim) ## 原子编码
        self.encoder2 = Encoder2(config.encoder,20) ##残基编码*************************
        self.decoder = Decoder(config.decoder,self.num_props)  ## 解码层
        self.projection = nn.Linear(config.hidden_channels, len(config.decoder.smiVoc) , bias=False) ## 输出层 d_model 是我们解码层每个token输出的维度大小，之后会做一个 tgt_vocab_size 大小的softmax
    def forward(self, node_attr, pos, batch,atom_laplacian, smiles_index, tgt_len, aa_node_attr, aa_pos, aa_batch,aa_laplacian, prop=None):
        enc_outputs1, enc_pad_mask1, msa_outputs = self.encoder(node_attr, pos, batch,atom_laplacian)     ##细粒度
        enc_outputs2, enc_pad_mask2 = self.encoder2(aa_node_attr, aa_pos, aa_batch,aa_laplacian, enc_pad_mask1, msa_outputs)     ##粗粒度
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

def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


if __name__ == '__main__':
    import os
    import yaml
    import argparse
    from easydict import EasyDict
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    
    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    
    model = Transformer(config.model,4)

    protein_atom_feature = torch.tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5]]).float()
    pos = torch.tensor([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]).float()
    batch = torch.LongTensor([0,0,1,1,1])
    smiles = ['CSCC(=O)NNC(=O)c1c(C)oc(C)c1C','CSCC(=O)NNC(=O)c1c(C)oc(C)c1C']

    output, enc_self_attns, dec_self_attns, dec_enc_attns = model(protein_atom_feature,pos,batch,smiles)

    print(output.shape)
