'''
NOTE: May 6 
Adopt the Trajectron dataset to make experiment easier

TODO: convert to our own dataset format later
'''
import os
import sys
sys.path.append('/home/brianyao/Documents/Trajectron-plus-plus')
sys.path.append('/home/brianyao/Documents/Trajectron-plus-plus/trajectron')
from trajectron.model.dataset.dataset import NodeTypeDataset
import numpy as np
import torch
from torch.utils import data
import dill
import json
import pdb

class ETHUCYDataset(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split

        conf_json = open(cfg.DATASET.ETH_CONFIG, 'r')
        hyperparams = json.load(conf_json)
        
        # May 20, updated based on discussiing with Trajectron++
        # June 4th, change hist len to 7 so that the total len is 8
        hyperparams['minimum_history_length'] = self.cfg.MODEL.INPUT_LEN-1 if self.split == 'test' else 1
        hyperparams['maximum_history_length'] = self.cfg.MODEL.INPUT_LEN-1
        
        # hyperparams['minimum_history_length'] = cfg.MODEL.MIN_HIST_LEN #1 # different from trajectron++, we don't use short histories.
        hyperparams['state'] = {'PEDESTRIAN':{'position':['x','y'], 'velocity':['x','y'], 'acceleration':['x','y']}}
        hyperparams['pred_state'] = {'PEDESTRIAN':{'position':['x','y']}}
        
        if split == 'train':
            f = open(os.path.join(cfg.DATASET.TRAJECTORY_PATH, cfg.DATASET.NAME+'_train.pkl'), 'rb')
        elif split == 'val':
            f = open(os.path.join(cfg.DATASET.TRAJECTORY_PATH, cfg.DATASET.NAME+'_val.pkl'), 'rb')
        elif split == 'test':
            f = open(os.path.join(cfg.DATASET.TRAJECTORY_PATH, cfg.DATASET.NAME+'_test.pkl'), 'rb')
        else:
            raise ValueError()
        train_env = dill.load(f, encoding='latin1')
        
        node_type=train_env.NodeType[0]
        train_env.attention_radius[(node_type, node_type)] = 3.0 #10.0
        augment = False
        if split=='train':
            min_history_timesteps = 1
            augment = True if self.cfg.DATASET.AUGMENT else False
        else:
            min_history_timesteps = 7
        self.dataset = NodeTypeDataset(train_env, 
                                        node_type, 
                                        hyperparams['state'],
                                        hyperparams['pred_state'],
                                        scene_freq_mult=hyperparams['scene_freq_mult_train'],
                                        node_freq_mult=hyperparams['node_freq_mult_train'],
                                        hyperparams=hyperparams, 
                                        augment=augment, 
                                        min_history_timesteps=min_history_timesteps,
                                        min_future_timesteps=hyperparams['prediction_horizon'],
                                        return_robot=False)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data, neighbors_data_st, neighbors_lower_upper, neighbors_future, \
            neighbors_edge_value, robot_traj_st_t, map_tuple, scene_name, timestep = self.dataset.__getitem__(index) # 
        ret = {}
        ret['first_history_index'] = first_history_index
        ret['input_x'] = x_t
        ret['input_x_st'] = x_st_t
        ret['target_y'] = y_t
        ret['target_y_st'] = y_st_t
        ret['cur_image_file'] = ''
        ret['pred_resolution'] = torch.ones_like(y_t)
        ret['neighbors_x'] = neighbors_data
        ret['neighbors_x_st'] = neighbors_data_st
        ret['neighbors_lower_upper'] = neighbors_lower_upper
        ret['neighbors_target_y'] = neighbors_future
        ret['neighbors_adjacency'] = neighbors_edge_value
        ret['scene_name'] = scene_name
        ret['timestep'] = timestep
        return ret

if __name__=='__main__':
    dataset = ETHUCYDataset(hyperparams)

    pdb.set_trace()