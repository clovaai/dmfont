"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from functools import partial
import torch.nn as nn
import torch
from .modules import ConvBlock, ResBlock, GCBlock, SAFFNBlock


class ComponentEncoder(nn.Module):
    """ Component image decomposer
    Encode the glyph into each component-wise features
    """
    def __init__(self, C_in, C, norm='none', activ='relu', pad_type='reflect',
                 sa=None, n_comp_types=3):
        super().__init__()
        self.n_heads = n_comp_types

        ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
        ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type)
        SAFFNBlk = partial(SAFFNBlock, **sa)

        self.shared = nn.ModuleList([
            ConvBlk(C_in, C, 3, 1, 1, norm='none', activ='none'),  # 128x128
            ConvBlk(C*1, C*2, 3, 1, 1, downsample=True), # 64x64
            GCBlock(C*2),
            ConvBlk(C*2, C*4, 3, 1, 1, downsample=True), # 32x32
            SAFFNBlk(C*4, size=32, rel_pos=True),
        ])

        self.heads = nn.ModuleList([
            nn.ModuleList([
                ResBlk(C*4, C*4, 3, 1),
                SAFFNBlk(C*4, size=32, rel_pos=False),
                ResBlk(C*4, C*4, 3, 1),
                ResBlk(C*4, C*8, 3, 1, downsample=True), # 16x16
                SAFFNBlk(C*8, size=16, rel_pos=False),
                ResBlk(C*8, C*8)
            ]) for _ in range(self.n_heads)
        ])

        # skip featuremap indices. Since shared featuremap is idx=0,
        # featuremap index starts from 1 in each head.
        self.skip_layers = [3]

        # final output shape
        self.final_shape = (C*8, 16, 16)

    def forward(self, x):
        # foward in shared
        for layer in self.shared:
            x = layer(x)

        # forward in heads
        feats = [x]
        xs = [x] * self.n_heads
        n_layers = len(self.heads[0])
        for layer_idx in range(n_layers):
            for head_idx, head in enumerate(self.heads):
                layer = head[layer_idx]
                xs[head_idx] = layer(xs[head_idx])

            comp_feature = torch.stack(xs, dim=1)  # [B, n_comp_types, C, H, W]
            feats.append(comp_feature)

        return feats

    def filter_skips(self, feats):
        if self.skip_layers is None:
            return None

        return [feats[i] for i in self.skip_layers]
