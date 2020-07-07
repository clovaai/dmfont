"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from functools import partial
import torch
import torch.nn as nn
from .modules import ResBlock, ConvBlock, w_norm_dispatch, activ_dispatch


class MultitaskDiscriminator(nn.Module):
    def __init__(self, C, n_fonts, n_chars, use_rx=True, w_norm='spectral', activ='none'):
        super().__init__()
        self.use_rx = use_rx
        self.activ = activ_dispatch(activ)()
        w_norm = w_norm_dispatch(w_norm)
        self.font_emb = w_norm(nn.Embedding(n_fonts, C))
        self.char_emb = w_norm(nn.Embedding(n_chars, C))
        if use_rx:
            self.rx = w_norm(nn.Conv2d(C, 1, kernel_size=1, padding=0))

    def forward(self, x, font_indices, char_indices):
        """
        Args:
            x: [B, C, H, W]
            font_indices: [B]
            char_indices: [B]

        Return:
            [rx_logit, font_logit, char_logit]; [B, 1, H, W]
        """
        x = self.activ(x)
        font_emb = self.font_emb(font_indices)  # [B, C]
        char_emb = self.char_emb(char_indices)  # [B, (3), C]

        if hasattr(self, "rx"):
            rx_out = self.rx(x) # [B, 1, H, W]
            ret = [rx_out]
        else:
            ret = [torch.as_tensor(0.0)] # dummy

        font_out = torch.einsum('bchw,bc->bhw', x, font_emb).unsqueeze(1)
        char_out = torch.einsum('bchw,bc->bhw', x, char_emb).unsqueeze(1)

        ret += [font_out, char_out]

        return ret


class Discriminator(nn.Module):
    def __init__(self, C, n_fonts, n_chars, activ='relu', gap_activ='relu', w_norm='spectral',
                 use_rx=False, pad_type='reflect'):
        super().__init__()
        ConvBlk = partial(ConvBlock, w_norm=w_norm, activ=activ, pad_type=pad_type)
        ResBlk = partial(ResBlock, w_norm=w_norm, activ=activ, pad_type=pad_type)
        feats = [
            ConvBlk(1, C, stride=2, activ='none'), # 64x64 (stirde==2)
            ResBlk(C*1, C*2, downsample=True),    # 32x32
            ResBlk(C*2, C*4, downsample=True),    # 16x16
            ResBlk(C*4, C*8, downsample=True),    # 8x8
            ResBlk(C*8, C*16, downsample=False),  # 8x8
            ResBlk(C*16, C*32, downsample=False), # 8x8
            ResBlk(C*32, C*32, downsample=False), # 8x8
        ]

        self.feats = nn.ModuleList(feats)
        gap_activ = activ_dispatch(gap_activ)
        self.gap = nn.Sequential(
            gap_activ(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.projD = MultitaskDiscriminator(C*32, n_fonts, n_chars, use_rx=use_rx, w_norm=w_norm)

    def forward(self, x, font_indices, char_indices, out_feats=False):
        feats = []
        for layer in self.feats:
            x = layer(x)
            feats.append(x)

        x = self.gap(x) # final features
        ret = self.projD(x, font_indices, char_indices)
        if out_feats:
            ret.append(feats)

        return ret

    @property
    def use_rx(self):
        return self.projD.use_rx
