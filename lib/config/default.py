# ------------------------------------------------------------------------------
# Modified from HRNet-Human-Pose-Estimation 
# (https://github.com/HRNet/HRNet-Human-Pose-Estimation)
# Copyright (c) Microsoft
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.SAVE_FREQ = 1
_C.AUTO_RESUME = False
_C.PIN_MEMORY = False
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.LANDMARKS = 2
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
_C.DATASET_COFW = CN()
_C.DATASET_COFW.DATASET = 'COFW'
_C.DATASET_COFW.TRAINSET = '/root/datasets_d/FaceAlignment/ocfw/COFW_train_color.mat'
_C.DATASET_COFW.TESTSET = '/root/datasets_d/FaceAlignment/ocfw/COFW_test_color.mat'
_C.DATASET_COFW.DATA_DIR = '/root/datasets_d/FaceAlignment/ocfw'
_C.DATASET_COFW.FLIP = True
_C.DATASET_COFW.SCALE_FACTOR = 0.25
_C.DATASET_COFW.ROT_FACTOR = 30
_C.DATASET_COFW.NUM_LANDMARKS = 29
_C.DATASET_COFW.LANDMARK_INDEX =[]

_C.DATASET_WFLW = CN()
_C.DATASET_WFLW.DATASET = 'WFLW'
_C.DATASET_WFLW.ROOT = '/root/dataset/wflw/WFLW_images'
_C.DATASET_WFLW.DATA_DIR = '/root/dataset/wflw'
_C.DATASET_WFLW.TRAINSET = '/root/dataset/wflw/face_landmarks_wflw_train.csv'
_C.DATASET_WFLW.TESTSET = '/root/dataset/wflw/face_landmarks_wflw_test.csv'
_C.DATASET_WFLW.FLIP = True
_C.DATASET_WFLW.SCALE_FACTOR = 0.25
_C.DATASET_WFLW.ROT_FACTOR = 30
_C.DATASET_WFLW.NUM_LANDMARKS = 98
_C.DATASET_WFLW.LANDMARK_INDEX = []

_C.DATASET_300W = CN()
_C.DATASET_300W.DATASET = '300W'
_C.DATASET_300W.TRAINSET = '/root/dataset/300W/train.tsv'
_C.DATASET_300W.TESTSET = '/root/dataset/300W/test_ibug.tsv'
_C.DATASET_300W.DATA_DIR = '/root/dataset/300W'
_C.DATASET_300W.FLIP = True
_C.DATASET_300W.SCALE_FACTOR = 0.25
_C.DATASET_300W.ROT_FACTOR = 30
_C.DATASET_300W.NUM_LANDMARKS = 68
_C.DATASET_300W.LANDMARK_INDEX = []


_C.DATASET_AFLW = CN()
_C.DATASET_AFLW.DATASET = 'AFLW'
_C.DATASET_AFLW.TRAINSET = '/root/datasets_d/FaceAlignment/AFLW/names_gtboxes_pts_train.txt'
_C.DATASET_AFLW.TESTSET = '/root/datasets_d/FaceAlignment/AFLW/names_gtboxes_pts_test.txt'
_C.DATASET_AFLW.DATA_DIR = '/root/datasets_d/FaceAlignment/AFLW/matlab_get_aflw'
_C.DATASET_AFLW.FLIP = True
_C.DATASET_AFLW.SCALE_FACTOR = 0.25
_C.DATASET_AFLW.ROT_FACTOR = 30
_C.DATASET_AFLW.NUM_LANDMARKS = 68
_C.DATASET_AFLW.LANDMARK_INDEX = []

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_BACKBONE = _C.TRAIN.LR
_C.TRAIN.CLIP_MAX_NORM = 0.0
_C.TRAIN.LARGEST_NUM = 20000
_C.TRAIN.EXPERT_NUM=16
_C.TRAIN.TOP_K= 8

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()
_C.TEST.SHUFFLE = True
_C.TEST.NUM_POINTS = 68

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
