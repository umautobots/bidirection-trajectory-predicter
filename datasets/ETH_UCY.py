'''
NOTE: May 6 
Adopt the Trajectron dataset to make experiment easier

TODO: convert to our own dataset format later
'''
import os
import sys
from .preprocessing import get_node_timestep_data
sys.path.append(os.path.realpath('./datasets'))
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
        # get some statistics on the dataset.
        all_obs_distance, all_obs_vel = [], []
        all_pred_distance, all_pred_vel = [], []
        for d in self.dataset:
            distance = torch.norm(d[1][-1, :2] - d[1][0, :2])
            all_obs_distance.append(distance)
            all_obs_vel.append(d[1][:, 2:4])

            distance = torch.norm(d[2][-1] - d[2][0])
            all_pred_distance.append(distance)
            all_pred_vel.append((d[2][1:] - d[2][:-1])/0.4)

        all_obs_vel = torch.cat(all_obs_vel, dim=0).norm(dim=1)
        all_obs_distance = torch.tensor(all_obs_distance)
        all_pred_vel = torch.cat(all_pred_vel, dim=0).norm(dim=1)
        all_pred_distance = torch.tensor(all_pred_distance)
        
        print("obs dist mu/sigma: {:.2f}/{:.2f}, obs vel mu/sigma: {:.2f}/{:.2f}, pred dist mu/sigma: {:.2f}/{:.2f}, pred vel mu/sigma: {:.2f}/{:.2f}".format(\
                all_obs_distance.mean(), all_obs_distance.std(), all_obs_vel.mean(), all_obs_vel.std(),
                all_pred_distance.mean(), all_pred_distance.std(), all_pred_vel.mean(), all_pred_vel.std()))
        
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

class NodeTypeDataset(data.Dataset):
    '''
    from Trajectron++: https://github.com/StanfordASL/Trajectron-plus-plus
    '''
    def __init__(self, env, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, augment=False, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']

        self.augment = augment

        self.node_type = node_type
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]
    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = list()
        
        for scene in self.env.scenes:
            present_node_dict = scene.present_nodes(np.arange(0, scene.timesteps), type=self.node_type, **kwargs)
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    index += [(scene, t, node)] *\
                             (scene.frequency_multiplier if scene_freq_mult else 1) *\
                             (node.frequency_multiplier if node_freq_mult else 1)

        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t, node) = self.index[i]

        if self.augment:
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)
        return get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                      self.edge_types, self.max_ht, self.max_ft, self.hyperparams)

if __name__=='__main__':
    dataset = ETHUCYDataset(hyperparams)
