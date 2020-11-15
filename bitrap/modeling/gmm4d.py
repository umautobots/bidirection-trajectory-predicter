'''
Modified from Trajectron++
'''
import torch
import torch.distributions as td
import numpy as np
import pdb

def to_one_hot(labels, n_labels):
    return torch.eye(n_labels, device=labels.device)[labels]

class GMM4D(td.Distribution):
    r"""
    NOTE: June 18
    4D Gaussian Mixture Model using two 2D Multivariate Gaussians each of as N components:
    NOTE: we assume the first 2 dimensions and second 2 dimensions are not correlated.

    Cholesky decompesition and affine transformation for sampling:

    .. math:: Z \sim N(0, I)

    .. math:: S = \mu + LZ

    .. math:: S \sim N(\mu, \Sigma) \rightarrow N(\mu, LL^T)

    where :math:`L = chol(\Sigma)` and

    .. math:: \Sigma = \left[ {\begin{array}{cc} \sigma^2_x & \rho \sigma_x \sigma_y \\ \rho \sigma_x \sigma_y & \sigma^2_y \\ \end{array} } \right]

    such that

    .. math:: L = chol(\Sigma) = \left[ {\begin{array}{cc} \sigma_x & 0 \\ \rho \sigma_y & \sigma_y \sqrt{1-\rho^2} \\ \end{array} } \right]

    :param log_pis: Log Mixing Proportions :math:`log(\pi)`. [..., N]
    :param mus: Mixture Components mean :math:`\mu`. [..., N * 2]
    :param log_sigmas: Log Standard Deviations :math:`log(\sigma_d)`. [..., N * 2]
    :param corrs: Cholesky factor of correlation :math:`\rho`. [..., N]
    :param clip_lo: Clips the lower end of the standard deviation.
    :param clip_hi: Clips the upper end of the standard deviation.
    """
    def __init__(self, log_pis, mus, log_sigmas, corrs):
        super(GMM4D, self).__init__(batch_shape=log_pis.shape[0], event_shape=log_pis.shape[1:])
        self.components = log_pis.shape[-1]
        self.dimensions = 2
        self.device = log_pis.device
        self.input_log_pis = log_pis
        log_pis = torch.clamp(log_pis, min=-1e5)
        self.log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)  # [..., N]
        self.mus = mus#self.reshape_to_components(mus)         # [..., N, 2]
        self.log_sigmas = log_sigmas#self.reshape_to_components(log_sigmas)  # [..., N, 2]
        self.sigmas = torch.exp(self.log_sigmas)                       # [..., N, 2]
        self.one_minus_rho2 = 1 - corrs**2                        # [..., N]
        self.one_minus_rho2 = torch.clamp(self.one_minus_rho2, min=1e-5, max=1)  # otherwise log can be nan
        self.corrs = corrs  # [..., N]
        zero_vector = torch.zeros_like(self.log_pis)
        
        sigma_0, sigma_1, sigma_2, sigma_3 = self.sigmas[..., 0], self.sigmas[..., 1], self.sigmas[..., 2], self.sigmas[..., 3]
        rho_01, rho_23 = self.corrs[..., 0], self.corrs[..., 1]
        self.L = torch.stack([torch.stack([         sigma_0,                                       zero_vector,       zero_vector,                                       zero_vector], dim=-1),
                              torch.stack([sigma_1 * rho_01, sigma_1 * torch.sqrt(self.one_minus_rho2[..., 0]),       zero_vector,                                       zero_vector], dim=-1),
                              torch.stack([     zero_vector,                                       zero_vector,           sigma_2,                                       zero_vector], dim=-1),
                              torch.stack([     zero_vector,                                       zero_vector,  sigma_3 * rho_23, sigma_3 * torch.sqrt(self.one_minus_rho2[..., 1])], dim=-1)
                              ],
                             dim=-2)
        self.cov = self.get_covariance_matrix()
        self.pis_cat_dist = td.Categorical(logits=log_pis)

    def to(self, device):
        self.device = device
        for key in self.__dict__.keys():
            if isinstance(getattr(self, key), torch.Tensor):
                setattr(self, key, getattr(self, key).detach().to(device))
        self.pis_cat_dist = td.Categorical(logits=self.log_pis)
        
    def squeeze(self, dim=None):
        for key in self.__dict__.keys():
            if isinstance(getattr(self, key), torch.Tensor):
                setattr(self, key, getattr(self, key).squeeze(dim))

    @classmethod
    def from_log_pis_mus_cov_mats(cls, log_pis, mus, cov_mats):
        '''
        NOTE: Jun 18
        Generate the GMM4D object given covariance matrix.
        Assume first 2 dims are independent to second 2 dims (xy vs wh)
        '''
        corrs_sigma12 = cov_mats[..., 0, 1]
        corrs_sigma34 = cov_mats[..., 2, 3]
        sigma_1 = torch.clamp(cov_mats[..., 0, 0], min=1e-8)
        sigma_2 = torch.clamp(cov_mats[..., 1, 1], min=1e-8)
        sigma_3 = torch.clamp(cov_mats[..., 2, 2], min=1e-8)
        sigma_4 = torch.clamp(cov_mats[..., 3, 3], min=1e-8)
        sigmas = torch.stack([torch.sqrt(sigma_1), torch.sqrt(sigma_2), torch.sqrt(sigma_3), torch.sqrt(sigma_4)], dim=-1)
        log_sigmas = torch.log(sigmas)
        corrs_12 = corrs_sigma12 / (torch.prod(sigmas[..., :2], dim=-1))
        corrs_34 = corrs_sigma34 / (torch.prod(sigmas[..., 2:], dim=-1))
        corrs = torch.stack([corrs_12, corrs_34], dim=-1)
        return cls(log_pis, mus, log_sigmas, corrs)
    
    def rsample(self, sample_shape=torch.Size(), full=False):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.

        :param sample_shape: Shape of the samples
               full: whether to get one sample from each components
        :return: Samples from the GMM.
        """
        mvn_samples = (self.mus +
                       torch.squeeze(
                           torch.matmul(self.L,
                                        torch.unsqueeze(
                                            torch.randn(size=sample_shape + self.mus.shape, device=self.device),
                                            dim=-1)
                                        ),
                           dim=-1))
        if full:
            return mvn_samples
        else:
            component_cat_samples = self.pis_cat_dist.sample(sample_shape)
            selector = torch.unsqueeze(to_one_hot(component_cat_samples, self.components), dim=-1)
            return torch.sum(mvn_samples * selector, dim=-2)
    

    def log_prob(self, value):
        r"""
        Calculates the log probability of a value using the PDF for bivariate normal distributions:

        .. math::
            f(x | \mu, \sigma, \rho)={\frac {1}{2\pi \sigma _{x}\sigma _{y}{\sqrt {1-\rho ^{2}}}}}\exp
            \left(-{\frac {1}{2(1-\rho ^{2})}}\left[{\frac {(x-\mu _{x})^{2}}{\sigma _{x}^{2}}}+
            {\frac {(y-\mu _{y})^{2}}{\sigma _{y}^{2}}}-{\frac {2\rho (x-\mu _{x})(y-\mu _{y})}
            {\sigma _{x}\sigma _{y}}}\right]\right)

        :param value: The log probability density function is evaluated at those values.
        :return: Log probability
        """
        
        value = torch.unsqueeze(value, dim=-2)       # [..., 1, 4]
        dx = value - self.mus                       # [..., N, 4]

        xy_exp_nominator = ((torch.sum((dx[..., :2]/self.sigmas[..., :2])**2, dim=-1)  # first and second term of exp nominator
                          - 2*self.corrs[..., 0]*torch.prod(dx[..., :2], dim=-1)/torch.prod(self.sigmas[...,:2], dim=-1)))    # [..., N]

        xy_log_p = -(2*np.log(2*np.pi)
                            + torch.log(self.one_minus_rho2[..., 0])
                            + 2*torch.sum(self.log_sigmas[...,:2], dim=-1)
                            + xy_exp_nominator/self.one_minus_rho2[..., 0]) / 2
        
        wh_exp_nominator = ((torch.sum((dx[..., 2:]/self.sigmas[..., 2:])**2, dim=-1)  # first and second term of exp nominator
                          - 2*self.corrs[..., 1]*torch.prod(dx[..., 2:], dim=-1)/torch.prod(self.sigmas[..., 2:], dim=-1)))    # [..., N]

        wh_log_p = -(2*np.log(2*np.pi)
                            + torch.log(self.one_minus_rho2[..., 1])
                            + 2*torch.sum(self.log_sigmas[..., 2:], dim=-1)
                            + wh_exp_nominator/self.one_minus_rho2[..., 1]) / 2

        return torch.logsumexp(self.log_pis + xy_log_p + wh_log_p, dim=-1)
    
    def log_prob_component(self, value, idx):
        '''
        NOTE: May 27 compute log prob given value and the component index that value should be sampled from
        value: (batch_size, dim)
        idx: (batch_size)
        ''' 
        dx = value - self.mus[range(len(idx)), idx]
        sigmas = self.sigmas[range(len(idx)), idx]
        log_sigmas = self.log_sigmas[range(len(idx)), idx]
        one_minus_rho2 = self.one_minus_rho2[range(len(idx)), idx]
        corrs = self.corrs[range(len(idx)), idx]
        
        xy_exp_nominator = ((torch.sum((dx[..., :2] / sigmas[..., :2]) ** 2, dim=-1)  # first and second term of exp nominator
                          - 2 * corrs[..., 0] * torch.prod(dx[..., :2], dim=-1) / torch.prod(sigmas[..., :2], dim=-1)))    # [..., N]

        xy_log_p = -(2 * np.log(2 * np.pi)
                            + torch.log(one_minus_rho2[..., 0])
                            + 2 * torch.sum(log_sigmas[..., :2], dim=-1)
                            + xy_exp_nominator / one_minus_rho2[..., 0]) / 2
        
        wh_exp_nominator = ((torch.sum((dx[..., 2:] / sigmas[..., 2:]) ** 2, dim=-1)  # first and second term of exp nominator
                          - 2 * corrs[..., 1] * torch.prod(dx[..., 2:], dim=-1) / torch.prod(sigmas[..., 2:], dim=-1)))    # [..., N]

        wh_log_p = -(2 * np.log(2 * np.pi)
                            + torch.log(one_minus_rho2[..., 1])
                            + 2 * torch.sum(log_sigmas[..., 2:], dim=-1)
                            + wh_exp_nominator / one_minus_rho2[..., 1]) / 2
        
        output = torch.log(torch.exp(self.log_pis[range(len(idx)), idx] + xy_log_p + wh_log_p))
        return output

    def reshape_to_components(self, tensor):
        if len(tensor.shape) == 5:
            return tensor
        return torch.reshape(tensor, list(tensor.shape[:-1]) + [self.components, self.dimensions])

    def get_covariance_matrix(self):
        cov_12 = self.corrs[..., 0] * torch.prod(self.sigmas[...,:2], dim=-1)
        cov_34 = self.corrs[..., 1] * torch.prod(self.sigmas[...,2:], dim=-1)
        zeros = torch.zeros_like(cov_12)
        E = torch.stack([torch.stack([self.sigmas[..., 0]**2, cov_12,                 zeros,                  zeros], dim=-1),
                         torch.stack([cov_12,                 self.sigmas[..., 1]**2, zeros,                  zeros], dim=-1),
                         torch.stack([zeros,                  zeros,                  self.sigmas[..., 2]**2, cov_34], dim=-1),
                         torch.stack([zeros,                  zeros,                  cov_34,                 self.sigmas[..., 3]**2], dim=-1)
                         ], dim=-2)
        return E

    def mode(self):
        """
        Calculates the mode of the GMM by calculating probabilities of a 2D mesh grid

        :param required_accuracy: Accuracy of the meshgrid
        :return: Mode of the GMM
        """
        return self.mus