"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from functools import partial
import torch.nn as nn
from .modules import ResBlock, Flatten


class AuxClassifier(nn.Module):
    def __init__(self, C, C_out, norm='BN', activ='relu', pad_type='reflect',
                 conv_dropout=0., clf_dropout=0.):
        super().__init__()
        ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type, dropout=conv_dropout)
        self.layers = nn.ModuleList([
            ResBlk(C, C*2, 3, 1, downsample=True),
            ResBlk(C*2, C*2, 3, 1),
            nn.AdaptiveAvgPool2d(1),
            Flatten(1),
            nn.Dropout(clf_dropout),
            nn.Linear(C*2, C_out)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
