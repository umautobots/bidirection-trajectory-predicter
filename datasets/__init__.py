__all__ = ['build_dataset']
import dill
import collections.abc 
from torch.utils.data._utils.collate import default_collate

import torch
from .JAAD import JAADDataset as JAAD # dataset name + method name
from .PIE import PIEDataset as PIE
from .ETH_UCY import ETHUCYDataset
from torch.utils.data import DataLoader
import pdb
_DATA_LAYERS = {
    'JAAD_BiTraPNP': JAAD, # dataset name + method name
    'JAAD_BiTraPGMM': JAAD,
    'PIE_BiTraPNP': PIE,
    'PIE_BiTraPGMM': PIE,
    'eth_BiTraPNP': ETHUCYDataset,
    'hotel_BiTraPNP': ETHUCYDataset,
    'univ_BiTraPNP': ETHUCYDataset,
    'zara1_BiTraPNP': ETHUCYDataset,
    'zara2_BiTraPNP': ETHUCYDataset,
    'eth_BiTraPGMM': ETHUCYDataset,
    'hotel_BiTraPGMM': ETHUCYDataset,
    'univ_BiTraPGMM': ETHUCYDataset,
    'zara1_BiTraPGMM': ETHUCYDataset,
    'zara2_BiTraPGMM': ETHUCYDataset,
}


def make_dataset(cfg, split):
    try:
        data_layer = _DATA_LAYERS[cfg.DATASET.NAME + '_' + cfg.METHOD]
    except:
        raise NameError("Unknown method and dataset combination:{} + {}".format(cfg.METHOD, cfg.DATASET.NAME))
    
    return data_layer(cfg, split)

def make_dataloader(cfg, split='train', logger=None):
    if split == 'test':
        batch_size = cfg.TEST.BATCH_SIZE
    else:
        batch_size = cfg.SOLVER.BATCH_SIZE
    dataloader_params ={
            "batch_size": batch_size,
            "shuffle":split == 'train',
            "num_workers": cfg.DATALOADER.NUM_WORKERS,
            "collate_fn": collate_dict,
            }
    
    dataset = make_dataset(cfg, split)
    dataloader = DataLoader(dataset, **dataloader_params)
    if hasattr(logger, 'info'):
        logger.info("{} dataloader: {}".format(split, len(dataloader)))
    else:
        print("{} dataloader: {}".format(split, len(dataloader)))
    return dataloader

def collate_dict(batch):
    '''
    batch: a list of dict
    '''
    if len(batch) == 0:
        return batch
    elem = batch[0]
    collate_batch = {}
    all_keys = list(elem.keys())
    for key in all_keys:
        # e.g., key == 'bbox' or 'neighbors_st' or so
        if elem[key] is None:
            collate_batch[key] = None
        elif isinstance(elem[key], collections.abc.Mapping):
            # We have to dill the neighbors structures. Otherwise each tensor is put into
            # shared memory separately -> slow, file pointer overhead
            # we only do this in multiprocessing
            neighbor_dict = {sub_key: [b[key][sub_key] for b in batch] for sub_key in elem[key]}
            collate_batch[key] = dill.dumps(neighbor_dict) if torch.utils.data.get_worker_info() else neighbor_dict
        else:
            collate_batch[key] = default_collate([b[key] for b in batch])
    return collate_batch
                