'''
Defined classes:
    class BiTraPNP()
Some utilities are cited from Trajectron++
'''
import sys
import numpy as np
import copy
from collections import defaultdict
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn

from .latent_net import CategoricalLatent, kl_q_p
from .gmm2d import GMM2D
from .gmm4d import GMM4D
from .dynamics.integrator import SingleIntegrator
from bitrap.layers.loss import cvae_loss, mutual_inf_mc

class BiTraPNP(nn.Module):
    def __init__(self, cfg, dataset_name=None):
        super(BiTraPNP, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.K = self.cfg.K 
        self.param_scheduler = None
        # encoder
        self.box_embed = nn.Sequential(nn.Linear(self.cfg.GLOBAL_INPUT_DIM, self.cfg.INPUT_EMBED_SIZE), 
                                        nn.ReLU()) 
        self.box_encoder = nn.GRU(input_size=self.cfg.INPUT_EMBED_SIZE,
                                hidden_size=self.cfg.ENC_HIDDEN_SIZE,
                                batch_first=True)

        #encoder for future trajectory
        # self.gt_goal_encoder = nn.Sequential(nn.Linear(self.cfg.DEC_OUTPUT_DIM, 16),
        #                                         nn.ReLU(),
        #                                         nn.Linear(16, 32),
        #                                         nn.ReLU(),
        #                                         nn.Linear(32, self.cfg.GOAL_HIDDEN_SIZE),
        #                                         nn.ReLU())
        self.node_future_encoder_h = nn.Linear(self.cfg.GLOBAL_INPUT_DIM, 32)
        self.gt_goal_encoder = nn.GRU(input_size=self.cfg.DEC_OUTPUT_DIM,
                                        hidden_size=32,
                                        bidirectional=True,
                                        batch_first=True)
        
            
        self.hidden_size = self.cfg.ENC_HIDDEN_SIZE        
        self.p_z_x = nn.Sequential(nn.Linear(self.hidden_size,  
                                            128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.cfg.LATENT_DIM*2))
        # posterior
        self.q_z_xy = nn.Sequential(nn.Linear(self.hidden_size + self.cfg.GOAL_HIDDEN_SIZE,
                                            128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.cfg.LATENT_DIM*2))

        # goal predictor
        self.goal_decoder = nn.Sequential(nn.Linear(self.hidden_size + self.cfg.LATENT_DIM,
                                                    128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, self.cfg.DEC_OUTPUT_DIM))
        #  add bidirectional predictor
        self.dec_init_hidden_size = self.hidden_size + self.cfg.LATENT_DIM if self.cfg.DEC_WITH_Z else self.hidden_size

        self.enc_h_to_forward_h = nn.Sequential(nn.Linear( self.dec_init_hidden_size, 
                                                      self.cfg.DEC_HIDDEN_SIZE),
                                                nn.ReLU(),
                                                )
        self.traj_dec_input_forward = nn.Sequential(nn.Linear(self.cfg.DEC_HIDDEN_SIZE, 
                                                              self.cfg.DEC_INPUT_SIZE),
                                                    nn.ReLU(),
                                                    )
        self.traj_dec_forward = nn.GRUCell(input_size=self.cfg.DEC_INPUT_SIZE,
                                            hidden_size=self.cfg.DEC_HIDDEN_SIZE) 
        
        self.enc_h_to_back_h = nn.Sequential(nn.Linear( self.dec_init_hidden_size,
                                                      self.cfg.DEC_HIDDEN_SIZE),
                                            nn.ReLU(),
                                            )
        
        self.traj_dec_input_backward = nn.Sequential(nn.Linear(self.cfg.DEC_OUTPUT_DIM, # 2 or 4 
                                                                self.cfg.DEC_INPUT_SIZE),
                                                        nn.ReLU(),
                                                        )
        self.traj_dec_backward = nn.GRUCell(input_size=self.cfg.DEC_INPUT_SIZE,
                                            hidden_size=self.cfg.DEC_HIDDEN_SIZE)

        self.traj_output = nn.Linear(self.cfg.DEC_HIDDEN_SIZE * 2, # merged forward and backward 
                                     self.cfg.DEC_OUTPUT_DIM)

    def gaussian_latent_net(self, enc_h, cur_state, target=None, z_mode=None):
        # get mu, sigma
        # 1. sample z from piror
        z_mu_logvar_p = self.p_z_x(enc_h)
        z_mu_p = z_mu_logvar_p[:, :self.cfg.LATENT_DIM]
        z_logvar_p = z_mu_logvar_p[:, self.cfg.LATENT_DIM:]
        if target is not None:
            # 2. sample z from posterior, for training only
            initial_h = self.node_future_encoder_h(cur_state)
            initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=initial_h.device)], dim=0)
            _, target_h = self.gt_goal_encoder(target, initial_h)
            target_h = target_h.permute(1,0,2)
            target_h = target_h.reshape(-1, target_h.shape[1] * target_h.shape[2])
            
            target_h = F.dropout(target_h,
                                p=0.25,
                                training=self.training)

            z_mu_logvar_q = self.q_z_xy(torch.cat([enc_h, target_h], dim=-1))
            z_mu_q = z_mu_logvar_q[:, :self.cfg.LATENT_DIM]
            z_logvar_q = z_mu_logvar_q[:, self.cfg.LATENT_DIM:]
            Z_mu = z_mu_q
            Z_logvar = z_logvar_q

            # 3. compute KL(q_z_xy||p_z_x)
            KLD = 0.5 * ((z_logvar_q.exp()/z_logvar_p.exp()) + \
                        (z_mu_p - z_mu_q).pow(2)/z_logvar_p.exp() - \
                        1 + \
                        (z_logvar_p - z_logvar_q))
            KLD = KLD.sum(dim=-1).mean()
            KLD = torch.clamp(KLD, min=0.001)
        else:
            Z_mu = z_mu_p
            Z_logvar = z_logvar_p
            KLD = 0.0
        
        # 4. Draw sample
        K_samples = torch.randn(enc_h.shape[0], self.K, self.cfg.LATENT_DIM).cuda()
        Z_std = torch.exp(0.5 * Z_logvar)
        Z = Z_mu.unsqueeze(1).repeat(1, self.K, 1) + K_samples * Z_std.unsqueeze(1).repeat(1, self.K, 1)

        if z_mode:
            Z = torch.cat((Z_mu.unsqueeze(1), Z), dim=1)
        return Z, KLD 

    def encode_variable_length_seqs(self, original_seqs, lower_indices=None, upper_indices=None, total_length=None):
        '''
        take the input_x, pack it to remove NaN, embed, and run GRU
        '''
        bs, tf = original_seqs.shape[:2]
        if lower_indices is None:
            lower_indices = torch.zeros(bs, dtype=torch.int)
        if upper_indices is None:
            upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
        if total_length is None:
            total_length = max(upper_indices) + 1
        # This is done so that we can just pass in self.prediction_timesteps
        # (which we want to INCLUDE, so this will exclude the next timestep).
        inclusive_break_indices = upper_indices + 1
        pad_list = []
        length_per_batch = []
        for i, seq_len in enumerate(inclusive_break_indices):
            pad_list.append(original_seqs[i, lower_indices[i]:seq_len])
            length_per_batch.append(seq_len-lower_indices[i])
        
        # 1. embed and convert back to pad_list
        x = self.box_embed(torch.cat(pad_list, dim=0))
        pad_list = torch.split(x, length_per_batch)
        
        # 2. run temporal
        packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False) 
        packed_output, h_x = self.box_encoder(packed_seqs)
        # pad zeros to the end so that the last non zero value 
        output, _ = rnn.pad_packed_sequence(packed_output,
                                            batch_first=True,
                                            total_length=total_length)
        return output, h_x

    def encoder(self, x, first_history_indices=None):
        '''
        x: encoder inputs
        '''
        outputs, _ = self.encode_variable_length_seqs(x,
                                                      lower_indices=first_history_indices)
        outputs = F.dropout(outputs,
                            p=self.cfg.DROPOUT,
                            training=self.training)
        if first_history_indices is not None:
            last_index_per_sequence = -(first_history_indices + 1)
            return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]
        else:
            # if no first_history_indices, all sequences are full length
            return outputs[:, -1, :]

    def forward(self, input_x, 
                target_y=None, 
                neighbors_st=None, 
                adjacency=None, 
                z_mode=False, 
                cur_pos=None, 
                first_history_indices=None):
        '''
        Params:
            input_x: (batch_size, segment_len, dim =2 or 4)
            target_y: (batch_size, pred_len, dim = 2 or 4)
        Returns:
            pred_traj: (batch_size, K, pred_len, 2 or 4)
        '''
        gt_goal = target_y[:, -1] if target_y is not None else None
        cur_pos = input_x[:, -1, :] if cur_pos is None else cur_pos
        batch_size, seg_len, _ = input_x.shape
        # 1. encoder
        h_x = self.encoder(input_x, first_history_indices)
                   
        # 2-3. latent net and goal decoder
        Z, KLD = self.gaussian_latent_net(h_x, input_x[:, -1, :], target_y, z_mode=False)
        enc_h_and_z = torch.cat([h_x.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1)
        pred_goal = self.goal_decoder(enc_h_and_z)
        
        dec_h = enc_h_and_z if self.cfg.DEC_WITH_Z else h_x
        pred_traj = self.pred_future_traj(dec_h, pred_goal)
        cur_pos = input_x[:, None, -1, :] if cur_pos is None else cur_pos.unsqueeze(1)
        pred_goal = pred_goal + cur_pos
        pred_traj = pred_traj + cur_pos.unsqueeze(1)
        # 5. compute loss
        if target_y is not None:
            # train and val
            loss_goal, loss_traj = cvae_loss(pred_goal, 
                                            pred_traj, 
                                            target_y, 
                                            best_of_many=self.cfg.BEST_OF_MANY
                                            ) 
            loss_dict =  {'loss_goal':loss_goal, 'loss_traj':loss_traj, 'loss_kld':KLD}
        else:
            # test
            loss_dict = {}
  
        return pred_goal, pred_traj, loss_dict, None, None

    def pred_future_traj(self, dec_h, G):
        '''
        use a bidirectional GRU decoder to plan the path.
        Params:
            dec_h: (Batch, hidden_dim) if not using Z in decoding, otherwise (Batch, K, dim) 
            G: (Batch, K, pred_dim)
        Returns:
            backward_outputs: (Batch, T, K, pred_dim)
        '''
        pred_len = self.cfg.PRED_LEN

        K = G.shape[1]
        # 1. run forward
        forward_outputs = []
        forward_h = self.enc_h_to_forward_h(dec_h)
        if len(forward_h.shape) == 2:
            forward_h = forward_h.unsqueeze(1).repeat(1, K, 1)
        forward_h = forward_h.view(-1, forward_h.shape[-1])
        forward_input = self.traj_dec_input_forward(forward_h)
        for t in range(pred_len): # the last step is the goal, no need to predict
            forward_h = self.traj_dec_forward(forward_input, forward_h)
            forward_input = self.traj_dec_input_forward(forward_h)
            forward_outputs.append(forward_h)
        
        forward_outputs = torch.stack(forward_outputs, dim=1)
        
        # 2. run backward on all samples
        backward_outputs = []
        backward_h = self.enc_h_to_back_h(dec_h)
        if len(dec_h.shape) == 2:
            backward_h = backward_h.unsqueeze(1).repeat(1, K, 1)
        backward_h = backward_h.view(-1, backward_h.shape[-1])
        backward_input = self.traj_dec_input_backward(G)#torch.cat([G])
        backward_input = backward_input.view(-1, backward_input.shape[-1])
        
        for t in range(pred_len-1, -1, -1):
            backward_h = self.traj_dec_backward(backward_input, backward_h)
            output = self.traj_output(torch.cat([backward_h, forward_outputs[:, t]], dim=-1))
            backward_input = self.traj_dec_input_backward(output)
            backward_outputs.append(output.view(-1, K, output.shape[-1]))
        
        # inverse because this is backward 
        backward_outputs = backward_outputs[::-1]
        backward_outputs = torch.stack(backward_outputs, dim=1)
        
        return backward_outputs