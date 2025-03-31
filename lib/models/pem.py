# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

# from detectron2.config import configurable
# from detectron2.layers import Conv2d
# from detectron2.utils.registry import Registry

from lib.models.positional_encoding import PositionEmbeddingSine



class LocalRepresentation(nn.Module):
    """
    Local Representation module for generating feature vectors from input features.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors (default: 256).

    Attributes:
        to_query_3x3 (nn.Conv2d): 3x3 depth-wise convolutional layer for local feature extraction.
        bn (nn.BatchNorm2d): Batch normalization layer.
        out (nn.Linear): Linear transformation layer.
        d_model (int): The dimensionality of the input and output feature vectors.

    Methods:
        forward(self, x): Forward pass through the LocalRepresentation module.
    """
    def __init__(self, d_model=256,dim=1024):
        super().__init__()

        self.to_query_3x3 = nn.Conv2d(dim, 256, 3, groups=d_model, padding=1)
        self.bn = nn.BatchNorm2d(dim)
        self.out = nn.Linear(d_model, d_model)

        self.d_model = d_model

    def forward(self, x):
        # Retrieve input tensor shape
        B, C, H, W = x.shape#[8,1024,32,32]

        # Apply pre-normalisation followed by 3x3 local convolution to extract local features
        x = self.bn(x)#[8,1024,32,32]
        x_3x3 = self.to_query_3x3(x)

        # Reshape the local features and permute dimensions for linear transformation
        return self.out(x_3x3.view(B, self.d_model, H*W).permute(0, 2, 1))


class PEM_CA(nn.Module):
    """
    Prototype-based Masked Cross-Attention module.

    This module implements a variant of the cross-attention mechanism for use in segmentation heads.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors (default: 256).
        nhead (int): The number of attention heads (default: 8).

    Attributes:
        to_query (LocalRepresentation): Module for converting input to query representations.
        to_key (nn.Sequential): Sequential module for transforming input to key representations.
        proj (nn.Linear): Linear transformation layer.
        final (nn.Linear): Final linear transformation layer.
        alpha (nn.Parameter): Parameter for scaling in the attention mechanism.
        num_heads (int): Number of attention heads.

    Methods:
        with_pos_embed(self, tensor, pos): Adds positional embeddings to the input tensor.
        most_similar_tokens(self, x, q, mask=None): Finds the most similar tokens based on content-based attention.
        forward(self, q, x, memory_mask, pos, query_pos): Forward pass through the PEM_CA module.
    """

    def __init__(self, d_model=256, nhead=8,dim=1024):
        super().__init__()

        self.feature_proj = LocalRepresentation(d_model,dim)
        self.query_proj = nn.Sequential(nn.LayerNorm(d_model),
                                    nn.Linear(d_model, d_model))

        self.proj = nn.Linear(d_model, d_model)
        self.final = nn.Linear(d_model, d_model)

        self.alpha = nn.Parameter(torch.ones(1, 1, d_model))
        self.num_heads = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def most_similar_tokens(self, x, q, mask=None):
        # Retrieve input tensors shapes
        B, N, C = x.shape
        Q, D = q.shape[1], C // self.num_heads

        # Reshape tensors in multi-head fashion
        x = x.view(B, N, self.num_heads, D).permute(0, 2, 1, 3)#[8bs,8heads,1024,32]
        q = q.view(B, Q, self.num_heads, D).permute(0, 2, 1, 3)#[8,8,1024,32]

        # Compute similarity scores between features and queries
        sim = torch.einsum('bhnc, bhqc -> bhnq', x, q)

        # Apply mask to similarity scores if provided
        if mask is not None:
            mask = (mask.flatten(2).permute(0, 2, 1).detach() < 0.0).bool()
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            mask[torch.all(mask.sum(2) == mask.shape[2], dim=2)] = False
            sim.masked_fill_(mask, float('-inf'))

        # Find indices of most similar tokens
        most_similar_indices = torch.argmax(sim, dim=2)#[8,8,156]

        # Gather most similar tokens
        return torch.gather(x, 2, most_similar_indices.unsqueeze(-1).expand(-1, -1, -1, D)).permute(0, 2, 1, 3).reshape(B, Q, C)

    def forward(self, tgt, memory, memory_mask, pos, query_pos):#memory[8,1024,32,32] pos[8,1024,32,32] query_pos=[156,8,256] tgt=[156,8,256]
        res = tgt#[156,8,256]

        # Add positional embeddings to input tensors
        memory= self.with_pos_embed(memory, pos)
        tgt=self.with_pos_embed(tgt, query_pos)
        # Project input tensors
        memory = self.feature_proj(memory)  # BxDxHxW
        tgt = self.query_proj(tgt.permute(1, 0, 2))  # BxQxD 8,156,256

        # Normalize input tensors
        memory = torch.nn.functional.normalize(memory, dim=-1)
        tgt = torch.nn.functional.normalize(tgt, dim=-1)

        # Find the most similar feature token to each query
        memory = self.most_similar_tokens(memory, tgt, memory_mask)  # BxQxD选出最相似的tokens8,156,256

        # Perform attention mechanism with projection and scaling
        out = nn.functional.normalize(self.proj(memory * tgt), dim=1) * self.alpha + memory  # BxQxD

        # Final linear transformation
        out = self.final(out)  # BxQxD

        return out.permute(1, 0, 2) + res

