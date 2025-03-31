# ------------------------------------------------------------------------------
# Modified from HRNet-Human-Pose-Estimation 
# (https://github.com/HRNet/HRNet-Human-Pose-Estimation)
# Copyright (c) Microsoft
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

import numpy as np
import torchvision
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ..core.inference import get_max_preds


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def generate_target(joints, heatmap_shape, target_type='gaussian', sigma=3):
    '''
    :param joints:  [bs, num_joints, 3]
    :param heatmap_shape: (w, h)
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    num_boxes, num_joints, _ = joints.shape
    target_weight = (joints[..., 2] > 0).float()

    device = joints.device

    map_w, map_h = heatmap_shape

    assert target_type == 'gaussian', 'Only support gaussian map now!'

    target = torch.zeros(num_boxes, num_joints, map_h,
                         map_w).float().to(device)

    # normalized [x, y] * [w, h]
    joints_loc = torch.round(
        joints[..., :2] * joints.new([map_w, map_h])).int()
    mu_x = joints_loc[..., 0]
    mu_y = joints_loc[..., 1]

    if target_type == 'gaussian':
        tmp_size = sigma * 3

        left = mu_x - tmp_size  # size: [num_box, num_kpts]
        right = mu_x + tmp_size + 1
        up = mu_y - tmp_size
        down = mu_y + tmp_size + 1

        # heatmap range
        img_x_min = torch.clamp(left, min=0)
        img_x_max = torch.clamp(right, max=map_w)
        img_y_min = torch.clamp(up, min=0)
        img_y_max = torch.clamp(down, max=map_h)

        # usable gaussian range
        gx_min = torch.clamp(-left, min=0)
        gx_max = img_x_max - left
        gy_min = torch.clamp(-up, min=0)
        gy_max = img_y_max - up

        is_out_bound = (left >= map_w) | (
            up >= map_h) | (right < 0) | (down < 0)

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = torch.arange(0, size, 1).float().to(device)
        y = x[:, None]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        for i in range(num_boxes):
            for j in range(num_joints):
                if is_out_bound[i, j]:
                    continue
                if target_weight[i, j] > 0:
                    target[i, j, img_y_min[i, j]: img_y_max[i, j],
                           img_x_min[i, j]: img_x_max[i, j]] = g[gy_min[i, j]: gy_max[i, j], gx_min[i, j]: gx_max[i, j]]
    return target, target_weight


def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    # if isinstance(output, dict):
    #     pred_coords = output['pred_coords'].detach().cpu()
    #     pred_logits = output['pred_logits'].detach().cpu()
    #     logit, ind = torch.max(F.softmax(pred_logits, dim=-1), dim=-1)
    #     joints = torch.cat([pred_coords, logit[..., None]], dim=-1)
    #     targets = generate_target(joints, config.MODEL.HEATMAP_SIZE, sigma=2)[0]
    #     bs, qs, h, w = targets.shape
    #     heatmaps = targets.new_zeros(bs, 18, h, w)
    #     for b in range(bs):
    #         for q in range(qs):
    #             heatmaps[b, ind[b, q]] += targets[b, q]
    #     save_batch_heatmaps(input, heatmaps, '{}_hm_pred.jpg'.format(prefix))

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['tpts'], meta['pts'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['pts'],
            '{}_pred.jpg'.format(prefix)
        )
    # if config.DEBUG.SAVE_HEATMAPS_GT:
    #     save_batch_heatmaps(
    #         input, target, '{}_hm_gt.jpg'.format(prefix)
    #     )
    # if config.DEBUG.SAVE_HEATMAPS_PRED:
    #     save_batch_heatmaps(
    #         input, output, '{}_hm_pred.jpg'.format(prefix)
    #     )


_300w=[ 1, 3, 5, 8, 11, 15, 19, 22, 24,  # left chin (0-8) 24 chin center
                         26, 30, 33, 37, 40, 42, 45, 48,  # right chin (9-16)
                         49, 50, 51, 52, 53, 58, 59, 60, 61, 62,  # brow (17-26)
                         67, 68, 69, 70, 71, 72, 73, 74, 75,   # nose 9 (27-35)
                         78, 80, 83, 84, 86, 88, 90, 91, 94, 96, 98, 100, # eyes 12 (36-47)
                         102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, # mouth 20 (48-67)
                         124]
cofw=[ 49, 62, 53, 58, 51, 56, 60, 64,  # brow 8 (0-7)
                    78, 96, 84, 90, 81, 87, 93, 99,  # eyes 8 (8-15)
                    122, 123, #pupil 2 (16, 17)
                    76, 77, 70, 73,  # //nose 4 (18-21)
                    102, 108, 105, 116, 120, 111,  # mouth 6 (22-27)
                    24, # chin 1  (28)
                    124]
aflw=[ 49, 51, 53, 58, 60, 62,  # brow 6 (0-5)
                    78, 122, 84, 90, 123, 96,  # eyes 6 (6-11)
                    76, 70, 77,  # nose 3 (12-14)
                    102, 120, 108,  # mouth 3 (15-17)
                    24, # chin 1 (18)
                    124] #背景
wflw=[ 0, 2, 4, 6, 7, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 23, 24,  # left chin (0-16) 24 chin center
                    25, 27, 28, 29, 31, 32, 34, 35, 36, 38, 39, 41, 43, 44, 46, 47,  # right chin (17-32)
                    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,  # brow (33-50)
                    67, 68, 69, 70, 71, 72, 73, 74, 75,  # nose 9 (51-59)
                    78, 79, 81, 82, 84, 85, 87, 89, 90, 92, 93, 95, 96, 97, 99, 101,  # eyes 16 (60-75)
                    102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, # mouth 20 (76-95)
                    122, 123, # pupil 2 (96-97) background
                    124] #背景
def draw_cosin_heatpmaps(query):#[6,8,156,256]
    """
       计算每层归一化后的query向量之间的余弦相似性并绘制相似性矩阵图

       参数:
       - query_tensor: 输入张量，形状为 [6, 8, 156, 256]
       """

    num_decoder, bs, l, dim =query.shape
    # 存储每层的归一化向量
    normalized_vectors = []
    cosin_vectors=[]

    # 遍历6层
    for i in range(num_decoder):
        # 获取第i层的 query 张量，形状为 [8, 156, 256]
        layer_query = query[i]
        # 对每一层的query进行归一化处理，得到 [256] 的向量
        landmark_indices=_300w
        select_query=[]
        for ii in range(68):
            select_query.append(layer_query[:,landmark_indices[ii],:])
        select_query=torch.stack(select_query)#[68,8,256]
        select_query_mean=select_query.mean(axis=(1))#[68,256]
        cosin_vectors.append(cosine_similarity(select_query_mean.cpu().numpy(),select_query_mean.cpu().numpy()))

    for i in range(len(cosin_vectors)):
    # 绘制相似性矩阵热图
        plt.figure(figsize=(8, 6))
        sns.heatmap(cosin_vectors[i], cmap='vlag', center=0, vmin=-1, vmax=1,
                    xticklabels=[f'Layer {i + 1}' for i in range(6)],
                    yticklabels=[f'Layer {i + 1}' for i in range(6)],
                    cbar_kws={"label": "Cosine Similarity"})
        plt.title("Cosine Similarity Across Layers (Normalized Query)")
        plt.show()
        plt.close()
