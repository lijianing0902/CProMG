import os
import shutil
import argparse
import torch
import pandas as pd
import re
import torch.utils.tensorboard
from models.search import *
from models.TRM import Transformer
from utils.datasets import *
from utils.transforms import *
from utils.misc import *
from utils.train import *
from utils.early_stop import *

from torch_geometric.transforms.add_positional_encoding import AddLaplacianEigenvectorPE
from torch_geometric.nn import radius_graph, knn_graph
from models.common import GaussianSmearing,Gaussian
from torch_geometric.utils import to_undirected

from utils.protein_ligand import PDBProtein
from utils.data import ProteinLigandData, torchify_dict

# python3 gen.py /home/lijianing/ljn/new/configs/main3.yml

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/CProMG-VQS.yml')
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--model', type=str, default='./pretrained/CProMG-VQS.pt')
    parser.add_argument('--out', type=str, default='./result/out.csv')
    parser.add_argument('--input', type=str, default='./data/crossdocked_pocket10/FABPI_RAT_2_132_0/1ifc_A_rec_2ifb_plm_lig_tt_min_0_pocket10.pdb')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # Logging
    log_dir = get_new_log_dir(args.logdir, )
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Transforms
    protein_featurizer = FeaturizeProteinAtom(config)
    residue_featurizer = FeaturizeProteinResidue(config)
    transform = Compose([
        protein_featurizer,
        residue_featurizer,
    ])


    # Model
    logger.info('Building model...')
    model = Transformer(
        config.model,
        protein_featurizer.feature_dim,
        config.train.num_props,
    ).to(args.device)

    # Optimizer and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    # 加载模型
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    it = checkpoint['iteration']
    model.eval()

    #data
    aa_laplacian = AddLaplacianEigenvectorPE(k=config.model.encoder.lap_dim,attr_name='protein_aa_laplacian')
    atom_laplacian = AddLaplacianEigenvectorPE(k=config.model.encoder.lap_dim,attr_name='protein_atom_laplacian')
    distance_expansion = GaussianSmearing(stop=10, num_gaussians=2)
    aa_distance_expansion = GaussianSmearing(stop=5, num_gaussians=2)
    gaussian = Gaussian(sigma=15)
    aa_gaussian = Gaussian(sigma=30)


    pocket_dict = PDBProtein(args.input).to_dict_atom()
    residue_dict = PDBProtein(args.input).to_dict_residue()

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        residue_dict=torchify_dict(residue_dict),
    )

    data.protein_filename = 'pocket1'
    edge_index = knn_graph(data.protein_pos, 8, flow='target_to_source')     
    edge_length = torch.norm(data.protein_pos[edge_index[0]] - data.protein_pos[edge_index[1]], dim=1)   
    edge_attr =  gaussian(edge_length)                 
    edge_index,edge_attr = to_undirected(edge_index,edge_attr,reduce='mean')  
    data.protein_atom_laplacian = atom_laplacian(data.protein_element.size(0), edge_index, edge_attr)


    aa_edge_index = knn_graph(data.residue_center_of_mass, 30,flow='target_to_source')                     
    aa_edge_length = torch.norm(data.residue_center_of_mass[aa_edge_index[0]] - data.residue_center_of_mass[aa_edge_index[1]], dim=1)
    aa_edge_attr = aa_gaussian(aa_edge_length) 
    aa_edge_index,aa_edge_attr = to_undirected(aa_edge_index,aa_edge_attr,reduce='mean')
    data.protein_aa_laplacian = aa_laplacian(data.residue_amino_acid.size(0), aa_edge_index, aa_edge_attr)


    transform(data)
    data.protein_element_batch = torch.zeros([len(data.protein_element)]).long()
    data.residue_amino_acid_batch = torch.zeros([len(data.residue_amino_acid)]).long()

    data.to(args.device)


    batch_size = 1
    num_beams = 20  
    topk = 1
    filename = data.protein_filename


    if config.train.num_props:
        prop = torch.tensor([config.generate.prop for i in range(batch_size*num_beams)],dtype = torch.float).to(args.device)
        assert prop.shape[-1] == config.train.num_props
        num = int(bool(config.train.num_props))
    else:
        num = 0
        prop = None
    beam_output= beam_search(model, config.model.decoder.smiVoc, num_beams, 
                                    batch_size, config.model.decoder.tgt_len + num, topk, data, prop)
    beam_output = beam_output.view(batch_size,topk,-1)

    for i,item in enumerate(beam_output):
        generate = []
        for j in item:
            smile = [config.model.decoder.smiVoc[n.item()] for n in j.squeeze()]
            smile = re.sub('[&$^]'
            , '',''.join(smile))
            generate.append(smile)
        
        logger.info('\n[protein] : %s \n [generate]: %s \n' % (
        filename[i], generate
        ))

        df1 = pd.DataFrame([filename[i]]*topk,columns=['PROTEINS'])
        df2 = pd.DataFrame(generate,columns=['SMILES'])
        df3 = pd.concat([df1,df2],join='outer',axis=1)

    df3.to_csv(args.out,index=False) 
