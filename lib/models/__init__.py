# ------------------------------------------------------------------------------
# Modified from HRNet-Human-Pose-Estimation 
# (https://github.com/HRNet/HRNet-Human-Pose-Estimation)
# Copyright (c) Microsoft
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings

import lib.models.pose_transformer
# import lib.models.deformable_pose_transformer
# try:
#     import lib.models.deformable_pose_transformer
# except ModuleNotFoundError as ex:
#     warnings.warn(f'{ex}. deformable_pose_transformer will be unavailable')
