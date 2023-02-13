import os
import pickle

import lmdb
import torch
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch_geometric.transforms.add_positional_encoding import AddLaplacianEigenvectorPE
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import to_undirected

from models.common import GaussianSmearing, ShiftedSoftplus, Gaussian
from ..protein_ligand import PDBProtein, parse_sdf_file
from ..data import ProteinLigandData, torchify_dict


class PocketLigandPairDataset(Dataset):

    def __init__(self, config,raw_path, transform=None):
        super().__init__()
        self.config = config
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + '_2.lmdb')
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + '_name2id_2.pt')

        self.aa_laplacian = AddLaplacianEigenvectorPE(k=config.model.encoder.lap_dim,attr_name='protein_aa_laplacian')
        self.atom_laplacian = AddLaplacianEigenvectorPE(k=config.model.encoder.lap_dim,attr_name='protein_atom_laplacian')
        self.distance_expansion = GaussianSmearing(stop=10, num_gaussians=2)
        self.aa_distance_expansion = GaussianSmearing(stop=5, num_gaussians=2)
        self.gaussian = Gaussian(sigma=15)
        self.aa_gaussian = Gaussian(sigma=30)
        self.transform = transform
        self.db = None

        self.keys = None
        
        if not os.path.exists(self.processed_path):
            self._process()
            self._precompute_name2id()
        
        self.name2id = torch.load(self.name2id_path)
        

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=15*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False, # Writable
        )
        #./data/crossdocked_pocket10/index.pkl
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)
            index.reverse()

        num_skipped = 0
        df = pd.read_csv('/data/dock_scores.csv')
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
               
                try:
                    pocket_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_atom()
                    residue_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_residue()
                    ligand_dict = parse_sdf_file(self.config,os.path.join(self.raw_path, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        residue_dict=torchify_dict(residue_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn


                    edge_index = knn_graph(data.protein_pos, self.config.model.encoder.knn, flow='target_to_source')     
                    edge_length = torch.norm(data.protein_pos[edge_index[0]] - data.protein_pos[edge_index[1]], dim=1)   
                    edge_attr =  self.gaussian(edge_length)                  
                    edge_index,edge_attr = to_undirected(edge_index,edge_attr,reduce='mean')  
                    data.protein_atom_laplacian = self.atom_laplacian(data.protein_element.size(0), edge_index, edge_attr)


                    aa_edge_index = knn_graph(data.residue_center_of_mass, 30,flow='target_to_source')                     
                    aa_edge_length = torch.norm(data.residue_center_of_mass[aa_edge_index[0]] - data.residue_center_of_mass[aa_edge_index[1]], dim=1)
                    aa_edge_attr = self.aa_gaussian(aa_edge_length)
                    aa_edge_index,aa_edge_attr = to_undirected(aa_edge_index,aa_edge_attr,reduce='mean')
                    data.protein_aa_laplacian = self.aa_laplacian(data.residue_amino_acid.size(0), aa_edge_index, aa_edge_attr)
                    
                    try:
                        data.vina_score = float(df[df.loc[:,'4']==ligand_fn].loc[:,'5'].item())
                    except:
                        data.vina_score = float(0)

                    txn.put(
                        key = str(i).encode(),
                        value = pickle.dumps(data)
                    )
                except :
                    continue

        db.close()

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data.protein_filename, data.ligand_filename)
            name2id[name] = i
        torch.save(name2id, self.name2id_path)
    
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        print(len(self.keys))
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data.id = idx
        assert data.protein_pos.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data
        


