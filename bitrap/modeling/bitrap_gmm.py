'''
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

class BiTraPGMM(nn.Module):
    def __init__(self, cfg, dataset_name=None):
        super(BiTraPGMM, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.K = self.cfg.K 
        self.param_scheduler = None
        # encoder
        self.box_embed = nn.Sequential(nn.Linear(self.cfg.GLOBAL_INPUT_DIM, self.cfg.INPUT_EMBED_SIZE), 
                                        nn.ReLU()) 
        self.box_encoder = nn.GRU(input_size=self.cfg.INPUT_EMBED_SIZE,
                                hidden_size=self.cfg.ENC_HIDDEN_SIZE,
                                batch_first=True)

        #encoder of futrue trajectory
        self.node_future_encoder_h = nn.Linear(6, 32)
        self.gt_goal_encoder = nn.GRU(input_size=self.cfg.DEC_OUTPUT_DIM,
                                        hidden_size=32,
                                        bidirectional=True,
                                        batch_first=True)
        

        # latent net
        self.hidden_size = self.cfg.ENC_HIDDEN_SIZE 
        self.GMM = GMM2D if self.cfg.DEC_OUTPUT_DIM == 2 else GMM4D
        self.p_z_x = CategoricalLatent(self.cfg, input_size=self.hidden_size, dropout=self.cfg.PRIOR_DROPOUT)
        self.q_z_xy = CategoricalLatent(self.cfg, input_size=self.hidden_size + self.cfg.GOAL_HIDDEN_SIZE, dropout=0.0)
        
        # goal predictor
        self.h_to_gmm_mu = nn.Linear(self.hidden_size + self.cfg.LATENT_DIM,
                                        self.cfg.DEC_OUTPUT_DIM)
        self.h_to_gmm_log_var = nn.Linear(self.hidden_size + self.cfg.LATENT_DIM,
                                            self.cfg.DEC_OUTPUT_DIM)
        self.h_to_gmm_corr = nn.Linear(self.hidden_size + self.cfg.LATENT_DIM, 
                                        int(self.cfg.DEC_OUTPUT_DIM / 2))
        # vel predictor
        self.h_to_gmm_log_pis_per_t = nn.Linear(self.cfg.DEC_HIDDEN_SIZE * 2,
                                                1)
        self.h_to_gmm_mu_per_t = nn.Linear(self.cfg.DEC_HIDDEN_SIZE * 2,
                                            self.cfg.DEC_OUTPUT_DIM)
        self.h_to_gmm_log_var_per_t = nn.Linear(self.cfg.DEC_HIDDEN_SIZE * 2,
                                                self.cfg.DEC_OUTPUT_DIM)
        self.h_to_gmm_corr_per_t = nn.Linear(self.cfg.DEC_HIDDEN_SIZE * 2, 
                                                int(self.cfg.DEC_OUTPUT_DIM / 2))
        self.integrator = SingleIntegrator(dt=self.cfg.dt, device='cuda')
        self.integrator_reverse = SingleIntegrator(dt=self.cfg.dt, device='cuda')

        # add bidirectional predictor
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

        self.traj_dec_input_backward_vel = nn.Sequential(nn.Linear(self.cfg.DEC_OUTPUT_DIM, # 2 or 4 
                                                                    self.cfg.DEC_INPUT_SIZE),
                                                            nn.ReLU(),
                                                            )
        
        self.traj_dec_input_backward = nn.Linear(self.cfg.DEC_OUTPUT_DIM, 
                                                    self.cfg.DEC_OUTPUT_DIM)
        self.traj_dec_backward = nn.GRUCell(input_size=self.dec_init_hidden_size + self.cfg.DEC_OUTPUT_DIM,
                                            hidden_size=self.cfg.DEC_HIDDEN_SIZE)
        
        self.traj_output = nn.Linear(self.cfg.DEC_HIDDEN_SIZE * 2, # merged forward and backward 
                                     self.cfg.DEC_OUTPUT_DIM)
    
    def categorical_latent_net(self, enc_h, cur_state, target=None, z_mode=False):
        if target is not None:
            # train and val
            # use GRU  and initialize encoder state
            initial_h = self.node_future_encoder_h(cur_state)
            initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=initial_h.device)], dim=0)
            _, target_h = self.gt_goal_encoder(target, initial_h)
            target_h = target_h.permute(1,0,2)
            target_h = target_h.reshape(-1, target_h.shape[1] * target_h.shape[2])
            
            target_h = F.dropout(target_h,
                                p=0.25,
                                training=self.training)

            # run prior and posterior
            if self.cfg.Z_CLIP:
                self.p_z_x(enc_h, 
                           self.param_scheduler.z_logit_clip)
                self.q_z_xy(torch.cat([enc_h, target_h], dim=-1), 
                            self.param_scheduler.z_logit_clip)
            else:
                self.p_z_x(enc_h)
                self.q_z_xy(torch.cat([enc_h, target_h], dim=-1))
            
            sampled_Z = self.q_z_xy.sample(1, full_dist=True, z_mode=z_mode)
            full_Z = sampled_Z
            KLD = kl_q_p(self.q_z_xy.dist, self.p_z_x.dist, kl_min=self.cfg.KL_MIN)
        else:
            self.p_z_x(enc_h)
            full_Z = self.p_z_x.sample(1, full_dist=True, z_mode=z_mode)
            sampled_Z = self.p_z_x.sample(self.K, full_dist=False, z_mode=z_mode)
            KLD = 0.0
        return sampled_Z, full_Z, KLD

    def project_to_GMM_params(self, h):
        """
        Projects tensor to parameters of a GMM with N components and D dimensions.

        :param h: Input tensor.
        :return: tuple(mus, log_sigmas, corrs)
            WHERE
            - mus: Mean of each GMM component. [N, D]
            - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
            - corrs: Correlation between the GMM components. [N]
        """
        mus = self.h_to_gmm_mu(h)
        log_var = self.h_to_gmm_log_var(h)
        corrs = torch.tanh(self.h_to_gmm_corr(h))
        return mus, log_var, corrs

    def project_to_GMM_params_per_t(self, h):
        """
        Projects tensor to parameters of a GMM with N components and D dimensions.

        :param h: Input tensor.
        :return: tuple(mus, log_sigmas, corrs)
            WHERE
            - mus: Mean of each GMM component. [N, D]
            - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
            - corrs: Correlation between the GMM components. [N]
        """
        log_pis = self.h_to_gmm_log_pis_per_t(h)
        mus = self.h_to_gmm_mu_per_t(h)
        log_var = self.h_to_gmm_log_var_per_t(h)
        corrs = torch.tanh(self.h_to_gmm_corr_per_t(h))
        return log_pis, mus, log_var, corrs

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
        z_mode = False
        Z, full_Z, KLD = self.categorical_latent_net(h_x, input_x[:, -1, :], target_y, z_mode=False) # , z_mode=False)
        mutual_info_p = mutual_inf_mc(self.p_z_x.dist)
        mutual_info_q = mutual_inf_mc(self.q_z_xy.dist) if self.q_z_xy.dist else mutual_info_p

        enc_h_and_z = torch.cat([h_x.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1)
        mu, log_var, corr = self.project_to_GMM_params(enc_h_and_z)
        if gt_goal is not None:
            # train val
            # NOTE: try predicting global goal 
            ret = self.get_train_val_GMM(mu, log_var, corr, gt_goal)#- cur_pos)
            pred_goal, loc_gmm_goal, goal_NLL = ret
            loc_gmm_goal_full = loc_gmm_goal
        else:
            # test
            ret = self.get_eval_GMM(mu, log_var, corr, h_x, per_step=False, full_dist=False)
            pred_goal, loc_gmm_goal, loc_gmm_goal_full = ret
                        
        loc_gmm_traj = None
        self.integrator.set_initial_condition({'pos': cur_pos})
        self.integrator_reverse.set_initial_condition({'pos': gt_goal})
            
        # NOTE: June 8 predict GMM for each time step
        pred_traj = self.pred_future_traj_GMM(enc_h_and_z, loc_gmm_goal, K=Z.shape[1])
        if target_y is not None:
            # train and val
            ret = self.get_train_val_GMM(pred_traj['mus'], pred_traj['log_vars'], pred_traj['corrs'], 
                                            target_y, per_step=True)
            pred_traj, loc_gmm_traj, traj_NLL = ret 
            loss_dict = {'loss_goal': goal_NLL, 'loss_traj': traj_NLL, 'loss_kld': KLD, 'mutual_info_p': mutual_info_p, 'mutual_info_q': mutual_info_q}

        else:
            # test, also get the full_dist results.
            dec_h_full = torch.cat([h_x.unsqueeze(1).repeat(1, full_Z.shape[1], 1), full_Z], dim=-1)
            pred_traj_full = self.pred_future_traj_GMM(dec_h_full, loc_gmm_goal_full, K=loc_gmm_goal_full.mus.shape[2])
            pred_traj, _, _ = self.get_eval_GMM(pred_traj['mus'], pred_traj['log_vars'], pred_traj['corrs'], 
                                                                        per_step=True, full_dist=False)
            _, loc_gmm_traj, _ = self.get_eval_GMM(pred_traj_full['mus'], pred_traj_full['log_vars'], pred_traj_full['corrs'], 
                                                                        per_step=True, full_dist=True)
            loss_dict = {}
        
        # NOTE: comment cur_pos if not using residual prediction for goal
        loc_gmm_goal_viz = self.GMM(loc_gmm_goal_full.input_log_pis.detach(), 
                                    loc_gmm_goal_full.mus.detach(),# + cur_pos[:, None, None, :], 
                                    loc_gmm_goal_full.log_sigmas.detach(),
                                    loc_gmm_goal_full.corrs.detach())
        loc_gmm_traj_viz = loc_gmm_traj
        return pred_goal, pred_traj, loss_dict, loc_gmm_goal_viz, loc_gmm_traj_viz
        
    def get_train_val_GMM(self, mu, log_var, corr, target, per_step=False):
        '''
        generate the GMM object with given mu, log_var and corr
        Params:
            mu: (Batch, K, dim) for goal  or (Batch, T, K, dim) for trajectory
        Returns:
            predictions: (Batch, K, dim) for goal or (Batch, T, K, dim) for trajectory
        '''
        # NOTE: pi_i uses q_z_xy for train and p_z_x for validation
        log_pi = self.q_z_xy.dist.logits if self.training else self.p_z_x.dist.logits
        if per_step:
            log_pi = log_pi.unsqueeze(2).repeat(1, 1, mu.shape[1], 1)

        loc_gmm = self.GMM(log_pi, mu.unsqueeze(1), log_var.unsqueeze(1), corr.unsqueeze(1))
        if per_step:
            predictions = loc_gmm.mode() 
            loc_gmm_reverse = self.GMM(log_pi.flip(2), -loc_gmm.mus.flip(2), loc_gmm.log_sigmas.flip(2), loc_gmm.corrs.flip(2))
            
            # Integrate to get location GMM if we are predicting velocity gmm
            loc_gmm = self.integrator.integrate_distribution(loc_gmm)
            predictions = self.integrator.integrate_samples(predictions)
            NLL_loss = -torch.clamp(loc_gmm.log_prob(target.unsqueeze(1)), max=6)
            # Reverse integration to get reverse NLL_loss.
            loc_gmm_reverse = self.integrator_reverse.integrate_distribution(loc_gmm_reverse)
            target_reverse = torch.cat([self.integrator.initial_conditions['pos'][:,None,:], target[:,:-1]], dim=1).unsqueeze(1).flip(2)
            NLL_loss += (-torch.clamp(loc_gmm_reverse.log_prob(target_reverse), max=6))            
        else:
            predictions = loc_gmm.mode()
            # compute location NLL loss
            NLL_loss = -torch.clamp(loc_gmm.log_prob(target.unsqueeze(1)), max=6)
        predictions = predictions.squeeze(1)

        if per_step:
            NLL_loss = NLL_loss.sum(dim=-1)
        NLL_loss = NLL_loss.mean()
        return predictions, loc_gmm, NLL_loss

    def get_eval_GMM(self, mu, log_var, corr, h_x=None, per_step=False, full_dist=False):
        '''
        get the GMM model and goal for the evaluation process
        full_dist: whether we get the GMMs for the full dist or get the GMMs as sampled Gaussians
        Returns:
            predictions: (batch, T, sample, components, 2) or (batch, sample, components, 2)
        '''
        loc_gmm, loc_gmm_full = None, None
        # first, draw sample for test
        if not full_dist:
            if per_step:
                # enforce the 2nd dim to be sample and the 3rd dim to be time
                mu, log_var, corr = mu.permute(0,2,1,3), log_var.permute(0,2,1,3), corr.permute(0,2,1,3)
            log_pi = torch.ones_like(corr[..., 0:1]).to(mu.device)
            loc_gmm = self.GMM(log_pi, mu.unsqueeze(-2), log_var.unsqueeze(-2), corr.unsqueeze(-2))
            predictions = loc_gmm.rsample()
            
        else:
            log_pi = self.p_z_x.dist.logits
            if per_step:
                log_pi = log_pi.unsqueeze(2).repeat(1, 1, mu.shape[1], 1)
            loc_gmm = self.GMM(log_pi, mu.unsqueeze(1), log_var.unsqueeze(1), corr.unsqueeze(1))
            predictions = loc_gmm.mode()
        
        if per_step:
            # Integrate to get location GMM if we are predicting velocity gmm
            loc_gmm = self.integrator.integrate_distribution(loc_gmm)
            predictions = self.integrator.integrate_samples(predictions)

        # Second, Get the full GMM as well
        if not per_step:
            Z = self.p_z_x.sample(1, full_dist=True, z_mode=False)
            enc_h_and_z = torch.cat([h_x.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1)
            mu_full, log_var_full, corr_full = self.project_to_GMM_params(enc_h_and_z)
            log_pi_full = self.p_z_x.dist.logits
            loc_gmm_full = self.GMM(log_pi_full, mu_full.unsqueeze(1), log_var_full.unsqueeze(1), corr_full.unsqueeze(1))
        else:
            predictions = predictions.permute(0,2,1,3) if not full_dist else predictions.squeeze(1)

        return predictions, loc_gmm, loc_gmm_full

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
        backward_input = self.traj_dec_input_backward(G)
        backward_input = backward_input.view(-1, backward_input.shape[-1])
        
        for t in range(pred_len-1, -1, -1):
            backward_h = self.traj_dec_backward(backward_input, backward_h)
            output = self.traj_output(torch.cat([backward_h, forward_outputs[:, t]], dim=-1))
            backward_input = self.traj_dec_input_backward(output)
            backward_outputs.append(output.view(-1, K, output.shape[-1]))
        
        # inverse because this is backward 
        backward_outputs = backward_outputs[::-1]
        backward_outputs = torch.stack(backward_outputs, dim=1)
        # append goal to the end of traj
        return backward_outputs
    

    def pred_future_traj_GMM(self, dec_h, goal_loc_gmm, K=25):
        '''
        Let the traj pred to predict GMM at each timestep instead of 25 trajectories.
        forward is the same to the original bi-directional predictor
        backwards predict [log_pis, ]
        dec_h: (Batch, K, dim) or (Batch, dim)
        K: number of components, for train/val K is the defined num_components, e.g., 25
                                 for testing, K is the number of samples, e.g., 20

        
        '''
        pred_len = self.cfg.PRED_LEN

        # 1. run forward
        forward_outputs = []
        forward_h = self.enc_h_to_forward_h(dec_h)
        if len(forward_h.shape) == 2:
            forward_h = forward_h.unsqueeze(1).repeat(1, K, 1)
        forward_h = forward_h.view(-1, forward_h.shape[-1])
        forward_input = self.traj_dec_input_forward(forward_h)
        for t in range(pred_len): # the last step is the goal, no need to predict
            # --------------------
            # forward propagation input is updated from last hidden state
            forward_h = self.traj_dec_forward(forward_input, forward_h)
            forward_input = self.traj_dec_input_forward(forward_h)
            forward_outputs.append(forward_h)
            # --------------------
        
        forward_outputs = torch.stack(forward_outputs, dim=1)
        
        # 2. run backward on all samples
        backward_outputs = []
        backward_h = self.enc_h_to_back_h(dec_h)
        if len(dec_h.shape) == 2:
            backward_h = backward_h.unsqueeze(1).repeat(1, K, 1)
        backward_h = backward_h.view(-1, backward_h.shape[-1])
        
        
        flatten_goal_dist = self.GMM(torch.reshape(goal_loc_gmm.input_log_pis, [-1, 1]),
                                        torch.reshape(goal_loc_gmm.mus, [-1, 1, 2]),
                                        torch.reshape(goal_loc_gmm.log_sigmas, [-1, 1, 2]),
                                        torch.reshape(goal_loc_gmm.corrs, [-1, 1, 1]))
        inv_loc_mus = flatten_goal_dist.rsample()
        
        # generate backward input
        backward_input = torch.cat([dec_h.view(-1, dec_h.shape[-1]), self.traj_dec_input_backward(inv_loc_mus)], dim=-1)        
        
        backward_outputs = defaultdict(list)
        for t in range(pred_len-1, -1, -1):
            backward_h = self.traj_dec_backward(backward_input, backward_h)
            # predict the parameter of GMM at this timestep
            log_pis_t, mu_t, log_var_t, corrs_t = self.project_to_GMM_params_per_t(torch.cat([backward_h, forward_outputs[:, t]], dim=-1))
            backward_input = self.generate_backward_input(dec_h, log_pis_t, mu_t, log_var_t, corrs_t)
            backward_outputs['mus'].append(mu_t.view(-1, K, mu_t.shape[-1]))
            backward_outputs['log_vars'].append(log_var_t.view(-1, K, log_var_t.shape[-1]))
            backward_outputs['corrs'].append(corrs_t.view(-1, K, corrs_t.shape[-1]))
        
        # inverse because this is backward 
        for k, v in backward_outputs.items():
            backward_outputs[k] = torch.stack(v[::-1], dim=1)
        return backward_outputs
    
    def generate_backward_input(self, dec_h, log_pi_t, mu_t, log_var_t=None, corrs_t=None):
        '''
        generate the backward RNN inputs
        '''
        if log_pi_t.shape[0] != corrs_t.shape[0]:
            log_pi_t = torch.ones_like(corrs_t[..., :1])
        gmm = self.GMM(log_pi_t, mu_t.unsqueeze(1), log_var_t.unsqueeze(1), corrs_t.unsqueeze(1))
        backward_input = torch.cat([dec_h.view(-1, dec_h.shape[-1]), gmm.rsample()], dim=-1)

        return backward_input
