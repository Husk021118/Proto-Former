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
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from lib.utils.utils import show_preds
from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

from lib.utils.vis import save_batch_image_with_joints


class AFLW(data.Dataset):
    """AFLW
    """
    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET_AFLW.TRAINSET
        else:
            self.csv_file = cfg.DATASET_AFLW.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET_AFLW.DATA_DIR
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.IMAGE_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET_AFLW.SCALE_FACTOR
        self.rot_factor = cfg.DATASET_AFLW.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET_AFLW.FLIP
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])
        scale = self.landmarks_frame.iloc[idx, 1]
        box_size = self.landmarks_frame.iloc[idx, 2]

        center_w = self.landmarks_frame.iloc[idx, 3]
        center_h = self.landmarks_frame.iloc[idx, 4]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 5:].values
        pts = pts.astype('float').reshape(-1, 2)

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
                pts = fliplr_joints(pts, width=img.shape[1], dataset='AFLW')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)
        # show_preds(img, tpts)
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = tpts[:, 0:2] / self.input_size[0]
        target = target.astype(np.float32)
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)
        target_weight = np.ones((nparts, 1), dtype=np.float32)
        target_weight = torch.from_numpy(target_weight)
        # save_batch_image_with_joints(torch.Tensor(img).unsqueeze(0), tpts.unsqueeze(0), torch.Tensor(pts).unsqueeze(0),
        #                              'crop.jpg', nrow=1)
        meta = {'index': idx, 'center': center, 'scale': scale, 'rotate': r,
                'pts': torch.Tensor(pts), 'tpts': tpts, 'box_size': box_size}

        return img, target, target_weight, meta


if __name__ == '__main__':

    pass
