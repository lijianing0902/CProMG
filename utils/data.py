import numpy as np
import mindspore as ms
# from torch_geometric.data import Data, Batch

# class ProteinLigandData():

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     @staticmethod
#     def from_protein_ligand_dicts(protein_dict=None, residue_dict=None, ligand_dict=None, **kwargs):
#         instance = ProteinLigandData(**kwargs)

#         if protein_dict is not None:
#             for key, item in protein_dict.items():
#                 instance['protein_' + key] = item

#         if residue_dict is not None:
#             for key, item in residue_dict.items():
#                 instance['residue_' + key] = item

#         if ligand_dict is not None:
#             for key, item in ligand_dict.items():
#                 instance['ligand_' + key] = item

#         # instance['ligand_nbh_list'] = {i.item():[j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if instance.ligand_bond_index[0, k].item() == i] for i in instance.ligand_bond_index[0]}
#         return instance
    
def misify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = ms.Tensor.from_numpy(v)
        else:
            output[k] = v
    return output
    
def add_prefix(protein_dict=None, residue_dict=None, ligand_dict=None):
    pro_dic = {}
    for key, item in protein_dict.items():
        pro_dic['protein_' + key] = item
    resi_dic = {}
    for key, item in residue_dict.items():
        resi_dic['residue_' + key] = item
    lig_dic = {}
    for key, item in ligand_dict.items():
        lig_dic['ligand_' + key] = item

    return misify_dict({**pro_dic, **resi_dic, **lig_dic})





