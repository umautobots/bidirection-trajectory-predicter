'''
Adopted from Trajectron++
'''
import torch
from .utils import block_diag
from bitrap.modeling.gmm2d import GMM2D
from bitrap.modeling.gmm4d import GMM4D
import pdb
class Dynamic(object):
    # def __init__(self, dt, dyn_limits, device, model_registrar, xz_size, node_type):
    def __init__(self, dt, device):
        self.dt = dt
        self.device = device
        # self.dyn_limits = dyn_limits
        self.initial_conditions = None
        # self.model_registrar = model_registrar
        # self.node_type = node_type
        self.init_constants()
        # self.create_graph(xz_size)

    def set_initial_condition(self, init_con):
        self.initial_conditions = init_con

    def init_constants(self):
        pass

    # def create_graph(self, xz_size):
    #     pass

    def integrate_samples(self, s, x):
        raise NotImplementedError

    def integrate_distribution(self, dist, x):
        raise NotImplementedError

class SingleIntegrator(Dynamic):
    def init_constants(self):
        self.F = torch.eye(4, device=self.device, dtype=torch.float32)
        self.F[0:2, 2:] = torch.eye(2, device=self.device, dtype=torch.float32) * self.dt
        self.F_t = self.F.transpose(-2, -1)

    def integrate_samples(self, v, x=None):
        """
        Integrates deterministic samples of velocity.

        :param v: Velocity samples
        :param x: Not used for SI.
        :return: Position samples
        (Batch, sample, T, component, 2)
        """
        p_0 = self.initial_conditions['pos']
        if len(v.shape) - len(p_0.shape) == 3:
            p_0 = p_0[:, None, None, None, :] # (128, 1, 1, 1, 2)
        elif len(v.shape) - len(p_0.shape) == 2:
            p_0 = p_0[:, None, None, :]
        return torch.cumsum(v, dim=2) * self.dt + p_0


    def integrate_distribution(self, v_dist):
        r"""
        Integrates the GMM velocity distribution to a distribution over position.
        The Kalman Equations are used.

        .. math:: \mu_{t+1} =\textbf{F} \mu_{t}

        .. math:: \mathbf{\Sigma}_{t+1}={\textbf {F}} \mathbf{\Sigma}_{t} {\textbf {F}}^{T}

        .. math::
            \textbf{F} = \left[
                            \begin{array}{cccc}
                                \sigma_x^2 & \rho_p \sigma_x \sigma_y & 0 & 0 \\
                                \rho_p \sigma_x \sigma_y & \sigma_y^2 & 0 & 0 \\
                                0 & 0 & \sigma_{v_x}^2 & \rho_v \sigma_{v_x} \sigma_{v_y} \\
                                0 & 0 & \rho_v \sigma_{v_x} \sigma_{v_y} & \sigma_{v_y}^2 \\
                            \end{array}
                        \right]_{t}

        :param v_dist: Joint GMM Distribution over velocity in x and y direction.
        :param x: Not used for SI.
        :return: Joint GMM Distribution over position in x and y direction.

        Ours: (batch, n_samples, T, components, 2)
        """
        if len(self.initial_conditions['pos'].shape) == 2:
            p_0 = self.initial_conditions['pos'].unsqueeze(1).unsqueeze(1) # (Batch, dim) -> (Batch, 1, 1, dim)
        elif len(self.initial_conditions['pos'].shape) == 3: 
            p_0 = self.initial_conditions['pos'].unsqueeze(1) # (Batch, component, dim) -> (Batch, 1, component, dim)
        elif len(self.initial_conditions['pos'].shape) == 4:
            p_0 = self.initial_conditions['pos'] # (Batch, 1, component, dim)
        
        pos_mus = p_0[:, None] + torch.cumsum(v_dist.mus, dim=2) * self.dt
        vel_dist_sigma_matrix = v_dist.get_covariance_matrix()

        
        pos_dist_sigma_matrix = self.integrate_sigma(v_dist, vel_dist_sigma_matrix[..., :2, :2])
        if isinstance(v_dist, GMM2D):
            return GMM2D.from_log_pis_mus_cov_mats(v_dist.log_pis, pos_mus, pos_dist_sigma_matrix)
        elif isinstance(v_dist, GMM4D):
            size_dist_sigma_matrix = self.integrate_sigma(v_dist, vel_dist_sigma_matrix[..., 2:, 2:])
            pos_size_sigma_matrix =  torch.cat([torch.cat([pos_dist_sigma_matrix, torch.zeros_like(pos_dist_sigma_matrix)], dim=-1),
                                                torch.cat([torch.zeros_like(size_dist_sigma_matrix), size_dist_sigma_matrix], dim=-1)
                                                ], dim=-2)
            return GMM4D.from_log_pis_mus_cov_mats(v_dist.log_pis, pos_mus, pos_size_sigma_matrix)
        else:
            raise ValueError()

    def integrate_sigma(self, v_dist, vel_dist_sigma_matrix):
        ph = v_dist.mus.shape[-3]
        sample_batch_dim = list(v_dist.mus.shape[0:2])
        
        pos_dist_sigma_matrix_list = []
        pos_dist_sigma_matrix_t = torch.zeros(sample_batch_dim + [v_dist.components, 2, 2], device=self.device)
        for t in range(ph):
            vel_sigma_matrix_t = vel_dist_sigma_matrix[:, :, t]
            full_sigma_matrix_t = block_diag([pos_dist_sigma_matrix_t, vel_sigma_matrix_t])
            pos_dist_sigma_matrix_t = self.F[..., :2, :].matmul(full_sigma_matrix_t.matmul(self.F_t)[..., :2])
            pos_dist_sigma_matrix_list.append(pos_dist_sigma_matrix_t)

        pos_dist_sigma_matrix = torch.stack(pos_dist_sigma_matrix_list, dim=2)
        return pos_dist_sigma_matrix