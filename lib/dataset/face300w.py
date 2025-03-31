# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel


class Face300W(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        # # specify annotation file for dataset
        # if is_train:
        #     self.csv_file = cfg.DATASET_300W.TRAINSET
        # else:
        #     self.csv_file = cfg.DATASET_300W.TESTSET
        #
        # self.is_train = is_train
        # self.transform = transform
        # self.input_size = cfg.MODEL.IMAGE_SIZE
        # # self.output_size = cfg.MODEL.HEATMAP_SIZE
        # self.output_size = cfg.MODEL.IMAGE_SIZE
        # self.sigma = cfg.MODEL.SIGMA
        # self.scale_factor = cfg.DATASET_300W.SCALE_FACTOR
        # self.rot_factor = cfg.DATASET_300W.ROT_FACTOR
        # self.label_type = cfg.MODEL.TARGET_TYPE
        # self.flip = cfg.DATASET_300W.FLIP
        self.data_root = cfg.DATASET_300W.DATA_DIR
        #
        # # load annotations
        # self.landmarks_frame = pd.read_csv(self.csv_file, sep="\t")
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET_300W.TRAINSET
            landmarks_frame = pd.read_csv(self.csv_file, sep="\t")
            # 扩展训练集
            # 使用pdconcat将landmarks_frameDataFrame复制多次以扩展数据集。扩展倍数是根据配置文件中最大的训练样本数
            # cfg.TRAIN.LARGEST_NUM和原始数据集大小计算得出的。ignore_index = True确保扩展后的DataFrame索引是连续的

            extended_df = pd.concat([landmarks_frame] * (cfg.TRAIN.LARGEST_NUM // len(landmarks_frame) + 1),
                                    ignore_index=True)
            self.landmarks_frame = extended_df.iloc[:cfg.TRAIN.LARGEST_NUM]
        else:
            self.csv_file = cfg.DATASET_300W.TESTSET
            self.landmarks_frame = pd.read_csv(self.csv_file, sep="\t")

        self.is_train = is_train
        self.transform = transform
        self.input_size = cfg.MODEL.IMAGE_SIZE
        # self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.output_size = cfg.MODEL.IMAGE_SIZE
        self.scale_factor = cfg.DATASET_300W.SCALE_FACTOR
        self.rot_factor = cfg.DATASET_300W.ROT_FACTOR
        self.flip = cfg.DATASET_300W.FLIP
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        d = 'trainset/' if self.is_train else 'ibug/'

        image_path = self.data_root + self.landmarks_frame.iloc[idx, 0]
        scale = self.landmarks_frame.iloc[idx, 3]

        center_w = self.landmarks_frame.iloc[idx, 4]
        center_h = self.landmarks_frame.iloc[idx, 5]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 2]
        # pts = self.landmarks_frame.iloc[idx, 4:].values
        pts = np.array(list(map(float, pts.split(","))), dtype=np.float32).reshape(-1, 2)

        scale *= 1.25
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='300W')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        # target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)
                # target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                #                             label_type=self.label_type)

        # ---------------------- update coord target -------------------------------
        target = tpts[:, 0:2] / self.input_size[0]

        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)

        target_weight = np.ones((nparts, 1), dtype=np.float32)
        target_weight = torch.from_numpy(target_weight)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'rotate': r, 'pts': torch.Tensor(pts), 'tpts': tpts,
                'img_pth': image_path}

        return img, target, target_weight, meta


if __name__ == '__main__':

    pass
