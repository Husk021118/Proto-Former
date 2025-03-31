# ------------------------------------------------------------------------------
# Modified from HRNet-Human-Pose-Estimation 
# (https://github.com/HRNet/HRNet-Human-Pose-Estimation)
# Copyright (c) Microsoft
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import math

from ..utils.transforms import transform_preds, transform_pixel_float, get_transform

def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def compute_nme_io(preds, meta):

    targets = meta['pts']
    if isinstance(preds,torch.Tensor):
        preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)
    return rmse


def compute_nme(preds, meta):

    targets = meta['pts']
    if isinstance(preds,torch.Tensor):
        preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interpupil = np.linalg.norm(pts_gt[7, ] - pts_gt[10, ])
        elif L == 29:  # cofw
            interpupil = np.linalg.norm(pts_gt[16, ] - pts_gt[17, ])
        elif L == 68:  # 300w
            lcenter = (
                              pts_gt[36, :] + pts_gt[37, :] +
                              pts_gt[38, :] + pts_gt[39, :] +
                              pts_gt[40, :] + pts_gt[41, :]
                      ) / 6
            rcenter = (
                              pts_gt[42, :] + pts_gt[43, :] +
                              pts_gt[44, :] + pts_gt[45, :] +
                              pts_gt[46, :] + pts_gt[47, :]
                      ) / 6
            interpupil = np.linalg.norm(lcenter - rcenter)
        elif L == 98:
            interpupil = np.linalg.norm(pts_gt[96, ] - pts_gt[97, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interpupil * L)

    return rmse


def decode_preds(output, center, scale, res):
    coords = get_preds(output)  # float type

    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds

def decode_preds_from_coordinate(coords, center, scale, res):

    preds = coords.clone()
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)
    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds


def decode_preds_softargmax(output, center, scale, res):
    coords = get_preds(output)  # float type
    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))

            if (px - 2) > 0 and (px + 2) < res[1] and (py - 2) >0 and (py + 2) < res[0]:
                patch = hm[py-2:py+3,px-2:px+3]
                patch_softmax = torch.exp(patch) / torch.sum(torch.exp(patch))
                diff_x = torch.sum(torch.sum(patch_softmax,dim=0)*torch.Tensor([0,1,2,3,4])) - 2
                diff_y = torch.sum(torch.sum(patch_softmax,dim=1)*torch.Tensor([0,1,2,3,4])) - 2
                diff = torch.Tensor([diff_x,diff_y])
                coords[n][p] += diff

    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds


def get_originCoords(preds,meta):
    '''preds: b*c*2'''
    '''return origin_coords  b*c*2'''
    center = meta['center'].cuda(non_blocking=True)
    scale = meta['scale'].cuda(non_blocking=True)

    origin_coords = preds.clone()
    for i in range(preds.size(0)):
        w = 200 * scale[i]
        h = 200 * scale[i]
        ul = center[i] -torch.tensor((0.5*w,0.5*h)).cuda(non_blocking=True)
        normalize_ul = torch.tensor((-1,-1)).cuda(non_blocking=True)
        coords = preds[i]
        for p in range(coords.size(0)):
            coords[p, 0:2] = ul+(coords[p]-normalize_ul)/2*w
        origin_coords[i] = coords
    return origin_coords

def get_coordinate(xx,yy):
    """
    :param xx: batchsize*num_points*64
    :param yy: batchsize*num_points*64
    :return:coords batch*size*num_points*2
    """
    coords_x = torch.argmax(xx,2).unsqueeze(2)
    coords_y = torch.argmax(yy,2).unsqueeze(2)
    coords = torch.cat((coords_x,coords_y),2)
    return coords

def get_transformer_coords(preds,meta,output_size):
    b,c,_ = preds.size()
    coords = torch.zeros((b,c,2))
    for i in range(b):
        center = meta['center'][i]
        scale = meta['scale'][i]
        rotate = meta['rotate'][i]
        t = get_transform(center, scale, output_size, rot = rotate)
        t = np.linalg.inv(t)
        for j in range(c):
            pt = preds[i,j,:]
            new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
            new_pt = np.dot(t, new_pt)
            coords[i,j,:] = torch.from_numpy(new_pt[:2])
    return coords