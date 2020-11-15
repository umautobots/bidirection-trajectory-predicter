import os

from yacs.config import CfgNode as CN

_C = CN()

_C.USE_WANDB = False
_C.PROJECT = 'pedestrian_behavior'
_C.CKPT_DIR = 'data/JAAD/checkpoints'
_C.OUT_DIR = 'data/JAAD/outputs'
_C.DEVICE = 'cuda'
_C.METHOD = 'BiTraPNP'
_C.GPU = '0'
_C.VISUALIZE = False
_C.PRINT_INTERVAL = 20
# ------ MODEL ---
_C.MODEL = CN()
_C.MODEL.INPUT_LEN = 15 # for 30 fps, 15 is 0.5 second
_C.MODEL.PRED_LEN = 15 # for 30 fps, 15 is 0.5 second
_C.MODEL.GLOBAL_EMBED_SIZE = 8
_C.MODEL.LOCAL_EMBED_SIZE = 128
_C.MODEL.GLOBAL_HIDDEN_SIZE = 8
_C.MODEL.LOCAL_HIDDEN_SIZE = 16
_C.MODEL.GLOBAL_INPUT_DIM = 4
_C.MODEL.LOCAL_INPUT_DIM = 50
_C.MODEL.DEC_OUTPUT_DIM = 4
_C.MODEL.DEC_INPUT_SIZE = 512 # the actual input size to the decoder GRU, it's the concatenation of all separate inputs
_C.MODEL.dt = 0.4

# ----- FOL -----
_C.MODEL.WITH_FLOW = False # whether use flow in the model
_C.MODEL.IMG_SIZE = (256,256)
_C.MODEL.ENC_CONCAT_TYPE = 'average'
_C.MODEL.INPUT_EMBED_SIZE = 256
_C.MODEL.FLOW_EMBED_SIZE = 256
_C.MODEL.FLOW_HIDDEN_SIZE = 256
_C.MODEL.ENC_HIDDEN_SIZE = 256
_C.MODEL.DEC_HIDDEN_SIZE = 256
_C.MODEL.GOAL_HIDDEN_SIZE = 64
_C.MODEL.DROPOUT = 0.0
_C.MODEL.PRIOR_DROPOUT = 0.0

# ------ GOAL -----
_C.MODEL.BEST_OF_MANY = False
_C.MODEL.K = 20
_C.MODEL.LATENT_DIST = 'gaussian'
_C.MODEL.LATENT_DIM = 32 # size of Z, can be number of components of GMM when Z is categorical
_C.MODEL.DEC_WITH_Z = False
_C.MODEL.Z_CLIP = False
_C.MODEL.REVERSE_LOSS = True # whether to do reverse integration to get loss
_C.MODEL.KL_MIN = 0.07
# ----- DATASET -----
_C.DATASET = CN()
_C.DATASET.NAME = 'JAAD'
_C.DATASET.ETH_CONFIG = ''
_C.DATASET.ROOT = '/mnt/workspace/datasets/JAAD/'
_C.DATASET.TRAJECTORY_PATH = '/mnt/workspace/datasets/JAAD/trajectories/'
_C.DATASET.FPS = 30
_C.DATASET.BBOX_TYPE = 'cxcywh' # bbox is in cxcywh format
_C.DATASET.NORMALIZE = 'zero-one' # normalize to 0-1
_C.DATASET.MIN_BBOX = [0,0,0,0] # the min of cxcywh or x1x2y1y2
_C.DATASET.MAX_BBOX = [1920, 1080, 1920, 1080] # the max of cxcywh or x1x2y1y2
_C.DATASET.AUGMENT = True
# ---- DATALOADER -----
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4

# ------ TEST ----- 
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 128
_C.TEST.KDE_BATCH_SIZE = 24
# ------ SOLVER ----
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 30
_C.SOLVER.BATCH_SIZE = 512
_C.SOLVER.LR = 0.001
_C.SOLVER.scheduler = 'exp'
_C.SOLVER.GAMMA = 0.999

