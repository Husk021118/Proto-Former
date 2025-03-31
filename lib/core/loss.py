# ------------------------------------------------------------------------------
# Modified from HRNet-Human-Pose-Estimation 
# (https://github.com/HRNet/HRNet-Human-Pose-Estimation)
# Copyright (c) Microsoft
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothL1Loss(nn.Module):
    def __init__(self, scale=0.01):
        super(SmoothL1Loss, self).__init__()
        self.scale = scale

    def __repr__(self):
        return "SmoothL1Loss()"

    def forward(self, output, groundtruth):
        """
            input:  b x n x 2
            output: b x n x 1 => 1
        """
        delta_2 = (output - groundtruth).pow(2).sum(dim=-1, keepdim=False)
        #delta = delta_2.clamp(min=1e-12).sqrt()
        delta = delta_2.sqrt()
        loss = torch.where(\
                delta_2 < self.scale * self.scale, \
                0.5 / self.scale * delta_2, \
                delta - 0.5 * self.scale)
        return loss.mean()


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    
    @torch.no_grad()
    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        if target.numel() == 0:
            return [torch.zeros([], device=output.device)]
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def loss_labels(self, outputs, targets, indices, num_joints, landmark_index, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'][..., landmark_index] #[bs, 256, 128]

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        target_classes_o = src_idx[1].to(src_logits.device)
        
        target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2]-1,
                                    dtype=torch.int64, device=src_logits.device)
        # default to no-kpt class, for matched ones, set to 0, ..., 16
        target_classes[tgt_idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - self.accuracy(src_logits[tgt_idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_joints, landmark_index):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        tgt_lengths = pred_logits.new_ones(pred_logits.shape[0]) * (pred_logits.shape[2]-1)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_kpts(self, outputs, targets, indices, num_joints, landmark_index, weights):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # loss = SmoothL1Loss()

        assert 'pred_coords' in outputs
        # match gt --> pred
        src_idx = self._get_src_permutation_idx(indices)  # always (0, 1, 2, .., 16)
        tgt_idx = self._get_tgt_permutation_idx(indices)  # must be in range(0, 100)

        target_kpts = targets[src_idx]
        weights = weights[src_idx]
        src_kpts = outputs['pred_coords'][tgt_idx]
        # src_kpts = targets
        # weights = weights
        # target_kpts = outputs['pred_coords']

        loss_bbox = F.l1_loss(src_kpts, target_kpts, reduction='none') * weights

        losses = {'loss_kpts': loss_bbox.mean() * num_joints}

        return losses

    def calculate_cosine_similarity_loss(outputs, targets, indices, num_joints, landmark_index, expert_scores_list):
        """
        计算批次内所有 expert_scores 的平均余弦相似度损失，使得分数分布在批次内更为接近
        """
        loss = 0
        for i in range(len(expert_scores_list)):
            expert_scores = expert_scores_list[i]
            B = expert_scores.shape[0]  # 批次大小
            cosine_loss = 0.0
            for i in range(B):
                for j in range(i + 1, B):
                    # 计算样本 i 和样本 j 的余弦相似度
                    similarity = F.cosine_similarity(expert_scores[i], expert_scores[j], dim=0)
                    # 期望相似度接近 1，因此使用 1 - similarity 作为损失
                    cosine_loss += (1 - similarity)
            loss = loss + cosine_loss
            # 归一化损失，使得损失值独立于批次大小
            # cosine_loss = cosine_loss / (B * (B - 1) / 2)

        losses = {'loss_gating': loss}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_joints, landmark_index, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'kpts': self.loss_kpts,
            'gating_weight': self.calculate_cosine_similarity_loss,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_joints, landmark_index, **kwargs)

    def forward(self, outputs, targets, target_weights, config):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, num_joints, landmark_index = self.matcher(outputs_without_aux, targets, config)
        # indices = 2*[torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67])]

        idx = self._get_tgt_permutation_idx(indices)
        src_kpts = outputs['pred_coords'][idx].view(-1, num_joints, 2)
        pred = src_kpts * target_weights

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == 'kpts':
                losses.update(self.get_loss(loss, outputs, targets, indices, num_joints, landmark_index,weights=target_weights))
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_joints, landmark_index))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, num_joints, landmark_index = self.matcher(aux_outputs, targets, config)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    elif loss == 'kpts':
                        kwargs = {'weights': target_weights}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_joints, landmark_index, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'gating_weight' in outputs:
            gating_weights_list = outputs['gating_weight']
            l_dict = self.get_loss('gating_weight', None, None, None, None, gating_weights_list)
            l_dict = {k: v for k, v in l_dict.items()}
            losses.update(l_dict)

        return losses, pred.detach().cpu()
        # return losses, outputs['pred_coords'].detach().cpu()