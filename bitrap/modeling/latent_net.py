'''
May 20th 2020

Differen latent networks used in endpoint prediction
1. Gaussian Z
2. Categorical Z (can be used as  mixture component weights \pi for GMM)
'''
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.distributions as td
import pdb

class CategoricalLatent(nn.Module):
    def __init__(self, cfg, input_size, dropout=0.10):
        '''
        input_size: size of input from the encoder
        '''
        super(CategoricalLatent, self).__init__()
        self.cfg = cfg
        self.N = 1 # ?
        self.K = self.cfg.LATENT_DIM # number of components
        self.h_to_logit = nn.Sequential(nn.Linear(input_size,
                                                128),
                                        nn.ReLU(),
                                        nn.Linear(128, 64),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(64, self.cfg.LATENT_DIM))

        self.dist = None # the categorical distribution object
        
    def forward(self, h, z_logit_clip=None):
        '''
        h: hidden state used to compute distribution parameter, (batch, self.K)
        '''
        self.device = h.device
        h = self.h_to_logit(h)
        logits_separated = torch.reshape(h, (-1, self.N, self.K))
        logits_separated_mean_zero = logits_separated - torch.mean(logits_separated, dim=-1, keepdim=True)
        if z_logit_clip is not None and self.training:
            logits = torch.clamp(logits_separated_mean_zero, min=-z_logit_clip, max=z_logit_clip)
        else:
            logits = logits_separated_mean_zero

        self.dist = td.OneHotCategorical(logits=logits)
    
    def sample(self, num_samples, full_dist=False, z_mode=False):
        '''
        there are three sample mode
        1. full dist: get one z vector for each category, resulting in an Identity matrix
        2. z_mode: get the most likely z vector by maximizing the logits.
        '''
        if full_dist:
            bs = self.dist.probs.size()[0]
            z_NK = torch.from_numpy(self.all_one_hot_combinations(self.N, self.K)).float().to(self.device).repeat(bs, num_samples)
            num_components = self.K ** self.N
            k = num_samples * num_components
        elif z_mode:
            eye_mat = torch.eye(self.dist.event_shape[-1], device=self.device)
            argmax_idxs = torch.argmax(self.dist.probs, dim=2)
            z_NK = torch.unsqueeze(eye_mat[argmax_idxs], dim=0).expand(num_samples, -1, -1, -1)
            k = num_samples
        else:
            z_NK = self.dist.sample((num_samples,))
            k = num_samples

        return torch.reshape(z_NK, (-1, k, self.N * self.K))
    
    @staticmethod
    def all_one_hot_combinations(N, K):
        return np.eye(K).take(np.reshape(np.indices([K] * N), [N, -1]).T, axis=0).reshape(-1, N * K)  # [K**N, N*K]

    
def kl_q_p(q_dist, p_dist, kl_min=0.07):
    '''
    '''
    kl_separated = td.kl_divergence(q_dist, p_dist)
    if len(kl_separated.size()) < 2:
        kl_separated = torch.unsqueeze(kl_separated, dim=0)

    kl_minibatch = torch.mean(kl_separated, dim=0, keepdim=True)

    if kl_min > 0:
        kl_lower_bounded = torch.clamp(kl_minibatch, min=kl_min)
        kl = torch.sum(kl_lower_bounded)
    else:
        kl = torch.sum(kl_minibatch)

    return kl