import torch
from torch.utils.data import Subset
from .pl import PocketLigandPairDataset


def get_dataset(config, *args, **kwargs):
    name = config.dataset.name
    root = config.dataset.path
    if name == 'pl':
        dataset = PocketLigandPairDataset(config,root, *args, **kwargs)
        print(dataset.__len__())
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    
    if 'split' in config.dataset:
        split_by_name = torch.load(config.dataset.split)
        split = {
            k: [dataset.name2id[n] for n in names if n in dataset.name2id]
            for k, names in split_by_name.items()
        }
        # for k,v in split.items():
        #     if k == 'valid':
        #         print(v)
        #         for i in v:
        #             if i > 165706:
        #                 print(i)
        subsets = {k:Subset(dataset, indices=v) for k, v in split.items()}

    return subsets
