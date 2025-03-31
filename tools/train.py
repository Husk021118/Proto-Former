# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Han Liming
# ------------------------------------------------------------------------------

import os

import pprint
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib
from lib.config import config, update_config
from lib.dataset import COFW, WFLW, Face300W, AFLW
from lib.core import function
from lib.utils import utils
from lib.models.matcher import build_matcher
from lib.core.loss import SetCriterion
import time
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', default='../wflw.yaml', help='experiment configuration filename', type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(config, args)

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))
    
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = eval('lib.models.' +config.MODEL.NAME+'.get_face_alignment_net')(
        config, is_train=True
    )

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    
    weight_dict = {'loss_ce': 1, 'loss_kpts': config.MODEL.EXTRA.KPT_LOSS_COEF}
    if config.MODEL.EXTRA.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(config.MODEL.EXTRA.DEC_LAYERS - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        loss_gating_dict = {'loss_gating': 1}
        weight_dict.update(loss_gating_dict)

    # matcher_cofw = build_matcher(config.DATASET_COFW.NUM_LANDMARKS)
    # criterion_cofw = SetCriterion(config.DATASET_COFW.NUM_LANDMARKS, matcher_cofw, weight_dict,
    #                               config.MODEL.EXTRA.EOS_COEF,
    #                               ['labels', 'kpts', 'cardinality']).cuda()
    # matcher_wflw = build_matcher(config.DATASET_WFLW.NUM_LANDMARKS)
    # criterion_wflw = SetCriterion(config.DATASET_WFLW.NUM_LANDMARKS, matcher_wflw, weight_dict,
    #                               config.MODEL.EXTRA.EOS_COEF,
    #                               ['labels', 'kpts', 'cardinality']).cuda()
    # matcher_300w = build_matcher(config.DATASET_300W.NUM_LANDMARKS)
    # criterion_300w = SetCriterion(config.DATASET_300W.NUM_LANDMARKS, matcher_300w, weight_dict,
    #                               config.MODEL.EXTRA.EOS_COEF,
    #                               ['labels', 'kpts', 'cardinality']).cuda()
    # criterion = {'cofw': criterion_cofw, 'wflw': criterion_wflw, '300w': criterion_300w}

    matcher = build_matcher(config.MODEL.LANDMARKS)
    criterion = SetCriterion(config.MODEL.LANDMARKS, matcher, weight_dict, config.MODEL.EXTRA.EOS_COEF, ['labels', 'kpts', 'cardinality']).cuda()

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    optimizer = utils.get_optimizer(config, model)
    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, config.TRAIN.CHECKPOINT)
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_nme = checkpoint['best_nme']
            model.module.load_state_dict(checkpoint['state_dict'].module.state_dict())
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.TRAIN.LR_STEP,config.TRAIN.LR_FACTOR, last_epoch-1)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.TRAIN.END_EPOCH, eta_min=1e-6)

    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )

    cofw_train_loader = DataLoader(
        dataset=COFW(config,is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)
    wflw_train_loader = DataLoader(
        dataset=WFLW(config, is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)
    face300w_train_loader = DataLoader(
        dataset=Face300W(config, is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    aflw_train_loader = DataLoader(
        dataset=AFLW(config, is_train=True),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    # train_loaders = {'WFLW': wflw_train_loader, '300W': face300w_train_loader, 'COFW': cofw_train_loader,}
    train_loaders = {'AFLW': aflw_train_loader, 'WFLW': wflw_train_loader, '300W': face300w_train_loader,
                     'COFW': cofw_train_loader, }


    val_loader = DataLoader(
        dataset=Face300W(config, is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    val_loaders = {'300W': val_loader}

    start_time = time.time()
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        #
        tmp = lr_scheduler.get_last_lr()
        msg = 'the learning rate of the {} epoch is {}'.format(epoch, tmp)
        logger.info(msg)
        function.train(config, train_loaders, model, criterion, optimizer, epoch, start_time, writer_dict)
        lr_scheduler.step()

        # evaluate
        if epoch % 1 == 0:
            nme, predictions = function.validate(config, val_loaders, model, criterion, epoch, start_time, writer_dict)
            is_best = nme < best_nme
            best_nme = min(nme, best_nme)

            print("best:", is_best)
            if (best_nme < 0.0455) and is_best:
                utils.save_checkpoint(
                    {"state_dict": model,
                     "epoch": epoch + 1,
                     "best_nme": best_nme,
                     "optimizer": optimizer.state_dict(),
                     }, predictions, is_best, final_output_dir, 'checkpoint_{}.pth'.format(best_nme))
                logger.info('=> saving checkpoint to {}'.format(final_output_dir))

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    logger.info('========================')
    logger.info(f'|| Best NME: {best_nme} ||')
    logger.info('========================')

    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

if __name__ == '__main__':
    main()











