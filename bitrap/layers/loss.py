import torch
import torch.nn.functional as F
import pdb

def mutual_inf_mc(x_dist):
    dist = x_dist.__class__
    H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
    return (H_y - x_dist.entropy().mean(dim=0)).sum()

def cvae_loss(pred_goal, pred_traj, target, best_of_many=True):
        '''
        CVAE loss use best-of-many
        Params:
            pred_goal: (Batch, K, pred_dim)
            pred_traj: (Batch, T, K, pred_dim)
            target: (Batch, T, pred_dim)
            best_of_many: whether use best of many loss or not
        Returns:

        '''
        K = pred_goal.shape[1]
        target = target.unsqueeze(2).repeat(1, 1, K, 1)
    
        # select bom based on  goal_rmse
        goal_rmse = torch.sqrt(torch.sum((pred_goal - target[:, -1, :, :])**2, dim=-1))
        traj_rmse = torch.sqrt(torch.sum((pred_traj - target)**2, dim=-1)).sum(dim=1)
        if best_of_many:
            best_idx = torch.argmin(goal_rmse, dim=1)
            loss_goal = goal_rmse[range(len(best_idx)), best_idx].mean()
            loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
        else:
            loss_goal = goal_rmse.mean()
            loss_traj = traj_rmse.mean()
        
        return loss_goal, loss_traj
        
def bom_traj_loss(pred, target):
    '''
    pred: (B, T, K, dim)
    target: (B, T, dim)
    '''
    K = pred.shape[2]
    target = target.unsqueeze(2).repeat(1, 1, K, 1)
    traj_rmse = torch.sqrt(torch.sum((pred - target)**2, dim=-1)).sum(dim=1)
    best_idx = torch.argmin(traj_rmse, dim=1)
    loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
    return loss_traj

def fol_rmse(x_true, x_pred):
    '''
    Params:
        x_pred: (batch, T, pred_dim) or (batch, T, K, pred_dim)
        x_true: (batch, T, pred_dim) or (batch, T, K, pred_dim)
    Returns:
        rmse: scalar, rmse = \sum_{i=1:batch_size}()
    '''

    L2_diff = torch.sqrt(torch.sum((x_pred - x_true)**2, dim=-1))#
    L2_diff = torch.sum(L2_diff, dim=-1).mean()

    return L2_diff