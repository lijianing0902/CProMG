import copy
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.transforms.add_positional_encoding import AddLaplacianEigenvectorPE
from torch_geometric.transforms import Compose
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

from .data import ProteinLigandData
from .protein_ligand import ATOM_FAMILIES
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import get_laplacian
from models.common import GaussianSmearing, ShiftedSoftplus

class FeaturizeProteinAtom(object):

    def __init__(self,config):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        self.max_num_aa = 20
        self.config = config
    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0)  + 1

    def __call__(self, data:ProteinLigandData):

        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements) onehot    
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        x = torch.cat([element, is_backbone], dim=-1)
        data.protein_atom_feature = x
        del data.protein_molecule_name, data.protein_is_backbone, data.protein_atom_name, data.protein_atom_to_aa_type
        return data
    
class FeaturizeProteinResidue(object):

    def __init__(self,config):
        super().__init__()
        self.max_num_aa = 20
        self.config = config

    @property
    def feature_dim(self):
        return self.max_num_aa 

    def __call__(self, data:ProteinLigandData):
        amino_acid = F.one_hot(data.residue_amino_acid, num_classes=self.max_num_aa)
        data.residue_feature = amino_acid

        return data
