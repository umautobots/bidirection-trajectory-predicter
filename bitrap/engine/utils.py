import os
import numpy as np
import torch
from bitrap.modeling.gmm4d import GMM4D

def print_info(epoch, model, optimizer, loss_dict, logger):
    loss_dict['kld_weight'] = model.param_scheduler.kld_weight.item()
    loss_dict['z_logit_clip'] = model.param_scheduler.z_logit_clip.item()

    info = "Epoch:{},\t lr:{:6},\t loss_goal:{:.4f},\t loss_traj:{:.4f},\t loss_kld:{:.4f},\t \
            kld_w:{:.4f},\t z_clip:{:.4f} ".format( 
            epoch, optimizer.param_groups[0]['lr'], loss_dict['loss_goal'], loss_dict['loss_traj'], 
            loss_dict['loss_kld'], loss_dict['kld_weight'], loss_dict['z_logit_clip']) 
    if 'grad_norm' in loss_dict:
        info += ", \t grad_norm:{:.4f}".format(loss_dict['grad_norm'])
    
    if hasattr(logger, 'log_values'):
        logger.info(info)
        logger.log_values(loss_dict)#, step=max_iters * epoch + iters)
    else:
        print(info)

def viz_results(viz, 
                X_global, 
                y_global, 
                pred_traj, 
                img_path, 
                dist_goal, 
                dist_traj,
                bbox_type='cxcywh',
                normalized=True,
                logger=None, 
                name=''):
    '''
    given prediction output, visualize them on images or in matplotlib figures.
    '''
    id_to_show = np.random.randint(pred_traj.shape[0])

    # 1. initialize visualizer
    viz.initialize(img_path[id_to_show])

    # 2. visualize point trajectory or box trajectory
    if y_global.shape[-1] == 2:
        viz.visualize(pred_traj[id_to_show], color=(0, 1, 0), label='pred future', viz_type='point')
        viz.visualize(X_global[id_to_show], color=(0, 0, 1), label='past', viz_type='point')
        viz.visualize(y_global[id_to_show], color=(1, 0, 0), label='gt future', viz_type='point')
    elif y_global.shape[-1] == 4:
        T = X_global.shape[1]
        viz.visualize(pred_traj[id_to_show], color=(0, 255., 0), label='pred future', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=[-1])
        viz.visualize(X_global[id_to_show], color=(0, 0, 255.), label='past', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=list(range(0, T, 3))+[-1])
        viz.visualize(y_global[id_to_show], color=(255., 0, 0), label='gt future', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=[-1])        

    # 3. optinaly visualize GMM distribution
    if hasattr(dist_goal, 'mus') and viz.mode == 'plot':
        dist = {'mus':dist_goal.mus.numpy(), 'log_pis':dist_goal.log_pis.numpy(), 'cov': dist_goal.cov.numpy()}
        viz.visualize(dist, id_to_show=id_to_show, viz_type='distribution')
    
    # 4. get image. 
    if y_global.shape[-1] == 2:
        viz_img = viz.plot_to_image(clear=True)
    else:
        viz_img = viz.img

    if hasattr(logger, 'log_image'):
        logger.log_image(viz_img, label=name)

def post_process(cfg, X_global, y_global, pred_traj, pred_goal=None, dist_traj=None, dist_goal=None):
    '''post process the prediction output'''
    if len(pred_traj.shape) == 4:
        batch_size, T, K, dim = pred_traj.shape
    else:
        batch_size, T, dim = pred_traj.shape
    X_global = X_global.detach().to('cpu').numpy()
    y_global = y_global.detach().to('cpu').numpy()
    if pred_goal is not None:
        pred_goal = pred_goal.detach().to('cpu').numpy()
    pred_traj = pred_traj.detach().to('cpu').numpy()
    
    if hasattr(dist_traj, 'mus'):
        dist_traj.to('cpu')
        dist_traj.squeeze(1)
    if hasattr(dist_goal, 'mus'):
        dist_goal.to('cpu')
        dist_goal.squeeze(1)
    if dim == 4:
        # BBOX: denormalize and change the mode
        _min = np.array(cfg.DATASET.MIN_BBOX)[None, None, :] # B, T, dim
        _max = np.array(cfg.DATASET.MAX_BBOX)[None, None, :]
        if cfg.DATASET.NORMALIZE == 'zero-one':
            if pred_goal is not None:
                pred_goal = pred_goal * (_max - _min) + _min
            pred_traj = pred_traj * (_max - _min) + _min
            y_global = y_global * (_max - _min) + _min
            X_global = X_global * (_max - _min) + _min
        elif cfg.DATASET.NORMALIZE == 'plus-minus-one':
            if pred_goal is not None:
                pred_goal = (pred_goal + 1) * (_max - _min)/2 + _min
            pred_traj = (pred_traj + 1) * (_max[None,...] - _min[None,...])/2 + _min[None,...]
            y_global = (y_global + 1) * (_max - _min)/2 + _min
            X_global = (X_global + 1) * (_max - _min)/2 + _min
        elif cfg.DATASET.NORMALIZE == 'none':
            pass
        else:
            raise ValueError()

        # NOTE: June 19, convert distribution from cxcywh to image resolution x1y1x2y2
        if hasattr(dist_traj, 'mus') and cfg.DATASET.NORMALIZE != 'none':
        
            _min = torch.FloatTensor(cfg.DATASET.MIN_BBOX)[None, None, :].repeat(batch_size, T, 1) # B, T, dim
            _max = torch.FloatTensor(cfg.DATASET.MAX_BBOX)[None, None, :].repeat(batch_size, T, 1)
            zeros = torch.zeros_like(_min[..., 0])
            
            if cfg.DATASET.NORMALIZE == 'zero-one':
                A = torch.stack([torch.stack([(_max-_min)[..., 0], zeros, zeros, zeros], dim=-1),
                                torch.stack([zeros, (_max-_min)[..., 1], zeros, zeros], dim=-1),
                                torch.stack([zeros, zeros, (_max-_min)[..., 2], zeros], dim=-1),
                                torch.stack([zeros, zeros, zeros, (_max-_min)[..., 3]], dim=-1),
                                ], dim=-2)
                b = torch.tensor(_min)
            elif cfg.DATASET.NORMALIZE == 'plus-minus-one':
                A = torch.stack([torch.stack([(_max-_min)[..., 0]/2, zeros, zeros, zeros], dim=-1),
                                torch.stack([zeros, (_max-_min)[..., 1]/2, zeros, zeros], dim=-1),
                                torch.stack([zeros, zeros, (_max-_min)[..., 2]/2, zeros], dim=-1),
                                torch.stack([zeros, zeros, zeros, (_max-_min)[..., 3]/2], dim=-1),
                                ], dim=-2)
                b = torch.stack([(_max+_min)[..., 0]/2, (_max+_min)[..., 1]/2, (_max+_min)[..., 2]/2, (_max+_min)[..., 3]/2],dim=-1)
            try:
                traj_mus = torch.matmul(A.unsqueeze(2), dist_traj.mus.unsqueeze(-1)).squeeze(-1) + b.unsqueeze(2)
                traj_cov = torch.matmul(A.unsqueeze(2), dist_traj.cov).matmul(A.unsqueeze(2).transpose(-1,-2))
                goal_mus = torch.matmul(A[:, 0:1, :], dist_goal.mus.unsqueeze(-1)).squeeze(-1) + b[:, 0:1, :]
                goal_cov = torch.matmul(A[:, 0:1, :], dist_goal.cov).matmul(A[:,0:1,:].transpose(-1,-2))
            except:
                raise ValueError()

            dist_traj = GMM4D.from_log_pis_mus_cov_mats(dist_traj.input_log_pis, traj_mus, traj_cov)
            dist_goal = GMM4D.from_log_pis_mus_cov_mats(dist_goal.input_log_pis, goal_mus, goal_cov)
    return X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal