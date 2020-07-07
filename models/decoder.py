"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from functools import partial
import torch
import torch.nn as nn
from .modules import ConvBlock, ResBlock, HourGlass


class Integrator(nn.Module):
    """Integrate component type-wise features"""
    def __init__(self, C, n_comps=3, norm='none', activ='none', C_in=None):
        super().__init__()
        C_in = (C_in or C) * n_comps
        self.integrate_layer = ConvBlock(C_in, C, 1, 1, 0, norm=norm, activ=activ)

    def forward(self, comps):
        """
        Args:
            comps [B, n_comps, C, H, W]: component features
        """
        inputs = comps.flatten(1, 2)
        out = self.integrate_layer(inputs)

        return out


class Decoder(nn.Module):
    def __init__(self, C, C_out, size, norm='IN', activ='relu', pad_type='reflect', n_comp_types=3):
        super().__init__()

        ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
        ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type)
        # Hourglass block downsamples the featuremap to 1x1 where IN makes trouble.
        HGBlk = partial(HourGlass, size=size, norm='BN', activ=activ, pad_type=pad_type)

        IntegrateBlk = partial(
            Integrator, norm='none', activ='none', n_comps=n_comp_types
        )

        self.layers = nn.ModuleList([
            IntegrateBlk(C*8),
            HGBlk(C*8, C*16, n_downs=4),
            ResBlk(C*8, C*8, 3, 1),
            ResBlk(C*8, C*8, 3, 1),
            ConvBlk(C*8, C*4, 3, 1, 1, upsample=True),   # 32x32
            ConvBlk(C*12, C*8, 3, 1, 1),   # enc-skip
            ConvBlk(C*8, C*8, 3, 1, 1),
            ConvBlk(C*8, C*4, 3, 1, 1),
            ConvBlk(C*4, C*2, 3, 1, 1, upsample=True),   # 64x64
            ConvBlk(C*2, C*1, 3, 1, 1, upsample=True),   # 128x128
            ConvBlk(C*1, C_out, 3, 1, 1)
        ])

        self.skip_indices = [5]
        self.skip_layers = nn.ModuleList([IntegrateBlk(C*8, C_in=C*4)])

        self.out = nn.Tanh()

    def forward(self, comps, skips=None):
        """
        Args:
            comps [B, n_comps, C, H, W]: component features
            skips: skip features
        """
        if skips is not None:
            assert len(skips) == 1
            skip_idx = self.skip_indices[0]
            skip_layer = self.skip_layers[0]
            skip_feat = skips[0]

        x = comps
        for i, layer in enumerate(self.layers):
            if i == skip_idx:
                skip_feat = skip_layer(skip_feat)  # integrate skip features
                x = torch.cat([x, skip_feat], dim=1)

            x = layer(x)

        return self.out(x)
