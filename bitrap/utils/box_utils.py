import torch
import pdb
import copy

def cxcywh_to_x1y1x2y2(bboxes):
    bboxes = copy.deepcopy(bboxes)
    bboxes[..., [0,1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    bboxes[..., [2,3]] = bboxes[..., [0,1]] + bboxes[..., [2, 3]]
    return bboxes
def x1y1x2y2_to_cxcywh(bboxes):
    bboxes = copy.deepcopy(bboxes)
    bboxes[..., [0,1]] = (bboxes[..., [0, 1]] + bboxes[..., [2, 3]]) / 2
    bboxes[..., [2,3]] = (bboxes[..., [2, 3]] - bboxes[..., [0, 1]]) * 2
    return bboxes

def signedIOU(bboxes_1, bboxes_2, mode='x1y1x2y2'):
    '''
    Compute the signed IOU between bboxes
    bboxes_1: (T, 4)
    bboxes_2: (T, 4) or (N, T, 4)
    '''
    
    if len(bboxes_1.shape) < len(bboxes_2.shape):
        N = bboxes_2.shape[0]
        bboxes_1 = bboxes_1.unsqueeze(0).repeat(N, 1, 1)
    x1_max = torch.stack([bboxes_1[...,0], bboxes_2[...,0]], dim=-1).max(dim=-1)[0]
    y1_max = torch.stack([bboxes_1[...,1], bboxes_2[...,1]], dim=-1).max(dim=-1)[0]
    x2_min = torch.stack([bboxes_1[...,2], bboxes_2[...,2]], dim=-1).max(dim=-1)[0]
    y2_min = torch.stack([bboxes_1[...,3], bboxes_2[...,3]], dim=-1).max(dim=-1)[0]

    # intersection
    intersection = torch.where((x2_min - x1_max > 0) * (y2_min - y1_max > 0), 
                                torch.abs(x2_min - x1_max) * torch.abs(y2_min - y1_max), 
                                -torch.abs(x2_min - x1_max) * torch.abs(y2_min - y1_max))
    
    area_1 = (bboxes_1[...,2] - bboxes_1[...,0]) * (bboxes_1[...,3] - bboxes_1[...,1])
    area_2 = (bboxes_2[...,2] - bboxes_2[...,0]) * (bboxes_2[...,3] - bboxes_2[...,1])
    # signed IOU
    signed_iou = intersection/(area_1 + area_2 - intersection + 1e-6)
    
    # ignore [0,0,0,0] boxes, which are place holders
    refined_signed_iou = torch.where(bboxes_2.max(dim=-1)[0] == 0, -1*torch.ones_like(signed_iou), signed_iou)
    return refined_signed_iou