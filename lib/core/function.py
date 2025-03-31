# ------------------------------------------------------------------------------
# Modified from HRNet-Human-Pose-Estimation 
# (https://github.com/HRNet/HRNet-Human-Pose-Estimation)
# Copyright (c) Microsoft
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import random

import numpy as np
import torch
from torch.utils import data

from lib.core.evaluate import get_transformer_coords, compute_nme, compute_nme_io
from lib.core.inference import get_final_preds_match
from torchvision import transforms
from PIL import Image
from ..utils.vis import save_debug_images
from itertools import cycle
import datetime


logger = logging.getLogger(__name__)


def train(config, train_loaders, model, criterion, optimizer, epoch, start_time, writer_dict):
    """
    Params:
        train_loaders: {'cofw': cofw_train_loader, 'wflw': wflw_train_loader, '300w': face300w_train_loader}
        criterions: {'cofw': criterion_cofw, 'wflw': criterion_wflw, '300w': criterion_300w}
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    nme_count = 0
    nme_batch_sum = 0

    end = time.time()
    max_norm = config.TRAIN.CLIP_MAX_NORM
    enum_train_loaders = {}

    cycle_key = ''
    tmp = 0
    # for key in train_loaders.keys():
    #     if len(train_loaders[key])>tmp:
    #         tmp = len(train_loaders[key])
    #         cycle_key = key
        # enum_train_loaders[key] = enumerate(train_loaders[key])
    data_iter = [iter(train_loaders["AFLW"]), iter(train_loaders["WFLW"]),iter(train_loaders["300W"]), iter(train_loaders["COFW"])]

    # for i, data in enumerate(zip(train_loaders['AFLW'], cycle(train_loaders['WFLW']), cycle(train_loaders['300W']),
    #                              cycle(train_loaders['COFW']))):
    n=config.TRAIN.LARGEST_NUM
    k=len(train_loaders.keys())
    b=config.TRAIN.BATCH_SIZE_PER_GPU
    iter_num=n//b
    for i in range(iter_num):
        tmp_loss = []
        for ii in range(len(train_loaders.keys())):
            input, target, target_weight, meta = next(data_iter[ii])
            input.shape
            # print(dt_name)
            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            outputs = model(input)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            # output = outputs[dataset]
            loss_dict, pred = criterion(outputs, target, target_weight, config)
            pred *= config.MODEL.IMAGE_SIZE[0]
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k]
                           for k in loss_dict.keys() if k in weight_dict)

            preds = get_transformer_coords(pred, meta, config.MODEL.IMAGE_SIZE)

            nme_batch = compute_nme(preds, meta)
            nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
            nme_count = nme_count + preds.size(0)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            # sum(i for i in decoder_loss_list).backward()
            # if max_norm > 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)

            prefix = '{}_{}'.format(os.path.join(config.OUTPUT_DIR, 'train'), i)
            # save_debug_images(config, input, meta, target, pred, output, prefix)

            if (i+1) % config.PRINT_FREQ == 0:
                if ii==3:
                    et = (time.time() - start_time)
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    tmp_loss.append(loss.item())
                    msg = '[{0}]\t' \
                          'Epoch: [{1}][{2}/{3}]\t' \
                          'AFLW_L: {AFLW_L:.5f} WFLW_L: {WFLW_L:.5f}  300W_L: {W300_L:.5f} COFW_L: {COFW_L:.5f}  \t' \
                          'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                        .format(et,
                                epoch, i + 1, iter_num, AFLW_L=tmp_loss[0], WFLW_L=tmp_loss[1],
                                W300_L=tmp_loss[2], COFW_L=tmp_loss[3], loss=losses)
                    logger.info(msg)
             

                    if writer_dict:
                        writer = writer_dict['writer']
                        global_steps = writer_dict['train_global_steps']
                        writer.add_scalar('train_loss', losses.val, global_steps)
                        # writer.add_scalar('heatmap_loss', heatmap_losses.val, global_steps)
                        # writer.add_scalar('coordinate_loss', coordinate_losses.val, global_steps)
                        writer_dict['train_global_steps'] = global_steps + 1
                else:
                    tmp_loss.append(loss.item())

    et = (time.time() - start_time)
    et = str(datetime.timedelta(seconds=et))[:-7]
    nme = nme_batch_sum / nme_count
    msg = '{} Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} nme_count:{}' \
        .format(et ,epoch, batch_time.avg, losses.avg, nme, nme_count)
    logger.info(msg)


def validate(config, val_loaders, model, criterion, epoch, start_time, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.TEST.NUM_POINTS

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    nme_batch_ip = 0
    nme_batch_io = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()
    image_size = 512

    for dt_name, val_loader in val_loaders.items():
        predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))
        # criterion = criterions[dataset]
        criterion.eval()
        with torch.no_grad():
            for i, (input, target, target_weight, meta) in enumerate(val_loader):
                # measure data time
                data_time.update(time.time() - end)
                num_images = input.size(0)
                outputs = model(input)  # [-1]
                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                # output = outputs[dt_name]

                loss_dict, pred_ = criterion(outputs, target, target_weight, config)
                pred_ *= image_size
                weight_dict = criterion.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k]
                           for k in loss_dict.keys() if k in weight_dict)

                # preds = get_transformer_coords(pred_, meta, [256,256])
                # preds_loss0 = get_transformer_coords(meta['tpts'],meta,[256,256])
                num_joints = target.shape[-2]
                preds, _, pred = get_final_preds_match(config, outputs, num_joints, meta['center'], meta['scale'], meta['rotate'])
                # del outputs
                if config.TEST.SHUFFLE:
                    input_flipped = torch.flip(input, [3, ]).clone()
                    outputs_flipped = model(input_flipped)  # [-1]
                    preds_flipped, _, _ = get_final_preds_match(config, outputs_flipped, num_joints, meta['center'], meta['scale'], meta['rotate'], True)
                    # preds_flipped = get_transformer_coords(outputs_flipped['pred_coords'].detach().cpu()*image_size,
                    #                                        meta, [256, 256])
                    preds_mean = (preds + preds_flipped) / 2
                    # del outputs_flipped

                # NME
                nme_temp = compute_nme(preds, meta)
                # nme_temp_loss0 = compute_nme(preds_loss0, meta)
                if config.TEST.SHUFFLE:
                    nme_temp_ip = compute_nme(preds_mean, meta)
                    nme_temp_io = compute_nme_io(preds_mean, meta)
                    nme_batch_ip += np.sum(nme_temp_ip)
                    nme_batch_io += np.sum(nme_temp_io)
                # nme_temp = nme_temp_loss0
                # preds=preds_loss0

                # Failure Rate under different threshold
                failure_008 = (nme_temp > 0.08).sum()
                failure_010 = (nme_temp > 0.10).sum()
                count_failure_008 += failure_008
                count_failure_010 += failure_010

                nme_batch_sum += np.sum(nme_temp)
                # nme_batch_loss0 += np.sum(nme_temp_loss0)
                nme_count = nme_count + preds.shape[0]

                losses.update(loss.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                prefix = '{}_{}'.format(os.path.join(config.OUTPUT_DIR, 'validate'), i)
                # save_debug_images(config, input, meta, target, pred, output, prefix)

    nme = nme_batch_sum / nme_count
    nme_batch_ip = nme_batch_ip / nme_count
    nme_batch_io = nme_batch_io / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    et = (time.time() - start_time)
    et = str(datetime.timedelta(seconds=et))[:-7]

    msg = '[{}] Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]: {:.4f} nme_ip: {:.4f} nme_io: {:.4f}'.format(et, epoch, batch_time.avg, losses.avg, nme,
                                                               failure_008_rate, failure_010_rate, nme_batch_ip,
                                                               nme_batch_io)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme_batch_io, predictions


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 20:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
