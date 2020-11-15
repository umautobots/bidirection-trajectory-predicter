import os
from PIL import Image
import numpy as np
import cv2
import pdb
import torch
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from .box_utils import cxcywh_to_x1y1x2y2
# plot edges
edge_pairs = [(15,17), (15,0), (0,16), (16, 18), (0,1),
                (1,2), (1,5), (2,3), (3,4), (5,6), (6,7),
                (1,8), (8,9), (8,12), (9,10), (10,11),
                (12, 13), (13, 14), (11, 24,), (11, 22), (22, 23),
                (14, 21), (14, 19), (19, 20)]

def draw_single_pose(img, pose, color=None):
    '''
    Assume the poses are saved in BODY_25 format
    see here for details: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#pose-output-format-body_25
    '''
    if color is None:
        color = np.random.rand(3) * 255
    
    # if isinstance(pose, torch.Tensor):
    #     pose = pose.type(torch.int)
    if isinstance(pose, np.ndarray):
        pose = pose.astype(np.int)
    else:
        raise TypeError('Unknown pose type {}'.format(type(pose)))    
    # plot points
    for point in pose:
        if point.max() > 0:
            cv2.circle(img, tuple(point.tolist()), radius=3, color=color, thickness=-1)
    
    for edge in edge_pairs:
        if pose[edge[0]].max() <= 0 or pose[edge[1]].max() <= 0:
            continue
        else:
            cv2.line(img, tuple(pose[edge[0]].tolist()), tuple(pose[edge[1]].tolist()), color=color, thickness=2)
    return img

def vis_pose_on_img(img, poses, color=None):
    '''skeleton_traj: (T, 50)'''
    # visualize
    for pose in poses: #inversed_X_merged: #: #: 
        pose = pose.reshape(-1, 2)
        img = draw_single_pose(img, pose, color)#, dotted=False)

    return img

def viz_pose_trajectories(poses, img_root, vid_traj_id, frame_id, img=None, color=None):
    '''
    draw the temporal senquence of poses
    poses: (T, 25, 2)
    img: np.array
    '''
    frame_id = int(frame_id[-1])
    # NOTE this only works for JAAD
    vid = vid_traj_id[:10]
    traj_id = vid_traj_id[11:]
    frames_path = os.path.join(img_root, vid)
    if img is None:
        img = Image.open(os.path.join(frames_path, str(frame_id).zfill(5)+'.png'))
        img = np.array(img)
    
    img = vis_pose_on_img(img, poses, color=color)
    
    return img


class Visualizer():
    def __init__(self, mode='image'):
        self.mode = mode
        if self.mode == 'image':
            self.img = None
        elif self.mode == 'plot':
            self.fig, self.ax = None, None
        else:
            raise NameError(mode)
            
    def initialize(self, img_path=None):
        if self.mode == 'image':
            self.img = np.array(Image.open(img_path))
            self.H, self.W, self.CH = self.img.shape
        elif self.mode == 'plot':
            self.fig, self.ax = plt.subplots()
    
    def visualize(self, 
                  inputs, 
                  id_to_show=0,
                  normalized=False, 
                  bbox_type='x1y1x2y2',
                  color=(255,0,0), 
                  thickness=4, 
                  radius=5,
                  label=None,  
                  viz_type='point', 
                  viz_time_step=None):
        if viz_type == 'bbox':
            self.viz_bbox_trajectories(inputs, normalized=normalized, bbox_type=bbox_type, color=color, viz_time_step=viz_time_step)
        elif viz_type == 'point':
            self.viz_point_trajectories(inputs, color=color, label=label, thickness=thickness, radius=radius)
        elif viz_type == 'distribution':
            self.viz_distribution(inputs, id_to_show, thickness=thickness, radius=radius)
    
    def clear(self):
        plt.close()
        # plt.cla()
        # plt.clf()
        self.fig.clear()
        self.ax.clear()
        del self.fig, self.ax
    
    def save_plot(self, fig_path, clear=True):
        self.ax.set_xlabel('x [m]', fontsize=12)
        self.ax.set_ylabel('y [m]', fontsize=12)
        self.ax.legend(fontsize=12)
        plt.savefig(fig_path)
        if clear:
            self.clear()

    def plot_to_image(self, clear=False):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        self.ax.legend()
        # draw the renderer
        self.fig.canvas.draw()
        # Get the RGBA buffer from the figure
        w,h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (h, w, 3)
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        # buf = np.roll ( buf, 3, axis = 2 )
        if clear:
            self.clear()
        return buf

    def viz_distribution(self, dist, id_to_show, radius=3, thickness=3):
        '''
        NOTE: Only plot the endpoint distribution
        Params:
            dist: GMM2D object with shape (Batch, T, Components, dim) or (Batch, Components, dim)
        '''
        if len(dist['mus'].shape) == 3:
            covariances = dist['cov'][id_to_show]
            mus = dist['mus'][id_to_show]
            pis = np.exp(dist['log_pis'])[id_to_show]
        elif len(dist['mus'].shape) == 4:
            covariances = dist['cov'][id_to_show][-1]
            mus = dist['mus'][id_to_show][-1]
            pis = np.exp(dist['log_pis'])[id_to_show][-1]
        
        colors = np.random.rand(mus.shape[0], 3)
        if self.mode == 'image':
            colors *= 255
        for i, (pi, mu, cov) in enumerate(zip(pis, mus, covariances)):
            v, w = np.linalg.eigh(cov)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            if self.mode == 'plot':
                ell = mpl.patches.Ellipse(mu, v[0], v[1], 180. + angle, color=colors[i])
                self.ax.plot([mu[0]], [mu[1]], '*', color=colors[i])
                # ell.set_clip_box(splot.bbox)
                ell.set_alpha(pi)
                self.ax.add_artist(ell)
            elif self.mode == 'image':
                overlay = self.img.copy()
                cv2.ellipse(overlay, tuple(mu.astype(np.int32)), (int(10*v[1]), int(10*v[0])), 180. + angle, #
                            startAngle=0, endAngle=360, color=colors[i], thickness=-1)
                
                cv2.addWeighted(overlay, 2*pi, self.img, 1-2*pi, 0, self.img)
                cv2.drawMarker(self.img,  tuple(mu.astype(np.int32)), color=colors[i], markerType=0, markerSize=radius, thickness=thickness)
        if self.mode == 'plot':
            pi_stats = 'Max pi:{:.3f},  Min pi:{:.3f} \nMean pi:{:.3f}, STD pi:{:.3f}'.format(pis.max(), pis.min(), pis.mean(), pis.std())
            x, y = self.ax.get_xlim()[0], self.ax.get_ylim()[1]
            # self.ax.text(x, y, pi_stats)

        # pdb.set_trace()
    def viz_point_trajectories(self, points, color=(255,0,0), label=None, thickness=4, radius=5):
        '''
        points: (T, 2) or (T, K, 2)
        '''
        if self.mode == 'image':
            # plot traj on image
            if len(points.shape) == 2:
                points = points[:, None, :]
            T, K, _ = points.shape
            points = points.astype(np.int32)
            for k in range(K):
                # pdb.set_trace()
                cv2.polylines(self.img, [points[:, k, :]], isClosed=False, color=color, thickness=thickness)
                    
                for t in range(T):
                    cv2.circle(self.img, tuple(points[t, k, :]), color=color, radius=radius, thickness=-1)
        elif self.mode == 'plot':
            # plot traj in matplotlib 
            # pdb.set_trace()
            if len(points.shape) == 2:
                self.ax.plot(points[:, 0], points[:, 1], '-o', color=color, label=label)
            elif len(points.shape) == 3:
                # multiple traj as (T, K, 2)
                for k in range(points.shape[1]):
                    label = label if k == 0 else None
                    self.ax.plot(points[:, k, 0], points[:, k, 1], '-', color=color, label=label)
            else:
                raise ValueError('points shape wrong:', points.shape)
            self.ax.axis('equal')

    def draw_single_bbox(self, bbox, color=None):
        '''
        img: a numpy array
        bbox: a list or 1d array or tensor with size 4, in x1y1x2y2 format
        
        '''
        
        if color is None:
            color = np.random.rand(3) * 255
        cv2.rectangle(self.img, (int(bbox[0]), int(bbox[1])), 
                    (int(bbox[2]), int(bbox[3])), color, 2)
    

    # def viz_bboxes(self, bboxes, normalized=False, mode='x1y1x2y2', color=None):
    #     '''
    #     NOTE: May 1, draw multiple boxes on an image
    #     '''
    #     H, W, CH = img.shape
    #     if normalized:
    #         bboxes[:,[0, 2]] *= W
    #         bboxes[:,[1, 3]] *= H
    #     if mode == 'cxcywh':
    #         bboxes = cxcywh_to_x1y1x2y2(bboxes)
    #     elif mode == 'x1y1x2y2':
    #         pass
    #     else:
    #         raise ValueError(mode)
    #     bboxes = bboxes.astype(np.int32)
    #     for bbox in bboxes:
    #         self.img = self.draw_single_bbox(bbox, color=color)
    
    def viz_bbox_trajectories(self, bboxes, normalized=False, bbox_type='x1y1x2y2', color=None, thickness=4, radius=5, viz_time_step=None):
        '''
        bboxes: (T,4) or (T, K, 4)
        '''
        if len(bboxes.shape) == 2:
            bboxes = bboxes[:, None, :]

        if normalized:
            bboxes[:,[0, 2]] *= self.W
            bboxes[:,[1, 3]] *= self.H
        if bbox_type == 'cxcywh':
            bboxes = cxcywh_to_x1y1x2y2(bboxes)
        elif bbox_type == 'x1y1x2y2':
            pass
        else:
            raise ValueError(bbox_type)
        bboxes = bboxes.astype(np.int32)
        T, K, _ = bboxes.shape

        # also draw the center points
        center_points = (bboxes[..., [0, 1]] + bboxes[..., [2, 3]])/2 # (T, K, 2)
        self.viz_point_trajectories(center_points, color=color, thickness=thickness, radius=radius)

        # draw way point every several frames, just to make it more visible
        if viz_time_step:
            bboxes = bboxes[viz_time_step, :]
            T = bboxes.shape[0]
        for t in range(T):
            for k in range(K):
                self.draw_single_bbox(bboxes[t, k, :], color=color)
        
    def viz_goal_map(img, goal_map):
        '''
        img:
        goal_map: goal map after sigmoid
        '''
        alpha = 0.5
        # NOTE: de-normalize the image 
        img = copy.deepcopy(img)
        img = img.transpose((1,2,0))
        img = (((img + 1)/2) * 255).astype(np.uint8)
        
        cm = plt.get_cmap('coolwarm')
        goal_map = (cm(goal_map)[:, :, :3] * 255).astype(np.uint8)
        # img = cv2.addWeighted(goal_map, alpha, img, 1 - alpha, 0, img)
        
        return goal_map

