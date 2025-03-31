from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from lib.config import config as cfg
from lib.config import update_config
from lib.models.matcher import build_matcher
from lib.core.loss import SetCriterion
from lib.core.function import validate
from lib.utils.utils import create_logger, model_key_helper

from lib.dataset import get_dataset
from lib.dataset import COFW, WFLW, Face300W, AFLW
from torch.utils.data import DataLoader
import lib.models


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', default='../wflw.yaml', help='experiment configuration filename', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('lib.models.'+cfg.MODEL.NAME+'.get_face_alignment_net')(
        cfg, is_train=False
    )

    # if args.modelDir:
    #     logger.info('=> loading model from {}'.format(args.modelDir))
    #     state = torch.load(args.modelDir)
    # else:
    #     model_state_file = os.path.join(
    #         final_output_dir, 'final_state.pth'
    #     )
    #     logger.info('=> loading model from {}'.format(model_state_file))
    model_state_file='/home/a/hsk/expert/12/LLPT-LOSS2/tools/output/pose_transformer/wflw/checkpoint_0.04413130928523121.pth'
    state = torch.load(model_state_file)

    last_epoch = state['epoch']
    best_nme = state['best_nme']
    if 'best_state_dict' in state.keys():
        state = state['best_state_dict']
    model.load_state_dict(state['state_dict'].module.state_dict())

    # define loss function (criterion) and optimizer
    matcher = build_matcher(cfg.MODEL.LANDMARKS)
    weight_dict = {'loss_ce': 1, 'loss_kpts': cfg.MODEL.EXTRA.KPT_LOSS_COEF}
    criterion = SetCriterion(cfg.MODEL.LANDMARKS, matcher, weight_dict, cfg.MODEL.EXTRA.EOS_COEF, [
                                'labels', 'kpts', 'cardinality']).cuda()

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    gpus = list(cfg.GPUS)

    val_loader_300W = DataLoader(
        dataset=Face300W(cfg, is_train=False),
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    val_loader_AFLW = DataLoader(
        dataset=AFLW(cfg, is_train=False),
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    val_loader_COFW = DataLoader(
        dataset=COFW(cfg, is_train=False),
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    val_loader_WFLW = DataLoader(
        dataset=WFLW(cfg, is_train=False),
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    start_time=time.time()
    val_loaders_300W = {'300W': val_loader_300W}
    val_loaders_COFW = {'COFW': val_loader_COFW}
    val_loaders_AFLW = {'AFLW': val_loader_AFLW}
    val_loaders_WFLW = {'WFLW': val_loader_WFLW}

    # evaluate on validation set
    validate(cfg, val_loaders_300W, model, criterion,epoch=44,
             start_time=start_time, writer_dict=None)#ibug:io:0.0449,ip:0.0649
    # validate(cfg, val_loaders_COFW, model, criterion,epoch=44,
    #          start_time=start_time, writer_dict=None)#io:0.0341,ip:0.0491
    # validate(cfg, val_loaders_AFLW, model, criterion,epoch=44,
    #          start_time=start_time, writer_dict=None)#io:0.0163,ip:0.0732
    # validate(cfg, val_loaders_WFLW, model, criterion,epoch=44,
    #          start_time=start_time, writer_dict=None)#io:0.0459,ip:0.0660