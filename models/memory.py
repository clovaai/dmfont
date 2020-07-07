"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import numpy as np
import torch
import torch.nn as nn
from models.modules import split_dim, ConvBlock
import datasets.kor_decompose as kor
import datasets.thai_decompose as thai


def comp_id_to_addr(ids, language):
    """ Component id to memory address converter

    Args:
        ids [B, 3 or 4], torch.tensor: [B, 3] -> kor, [B, 4] -> thai.
    """
    ids = ids.clone()
    if language == 'kor':
        ids[:, 1] += kor.N_CHO
        ids[:, 2] += kor.N_CHO + kor.N_JUNG
    elif language == 'thai':
        ids[:, 1] += thai.N_CONSONANTS
        ids[:, 2] += thai.N_CONSONANTS + thai.N_UPPERS
        ids[:, 3] += thai.N_CONSONANTS + thai.N_UPPERS + thai.N_HIGHESTS
    else:
        raise ValueError(language)

    return ids


class DynamicMemory(nn.Module):
    # NOTE the dynamic memory can be accelerated by using torch tensor instead of python dict.
    def __init__(self):
        super().__init__()
        self.memory = {}
        self.reset()

    def write(self, style_ids, comp_addrs, comp_feats):
        """ Batch write

        Args:
            style_ids: [B]
            comp_addrs: [B, 3]
            comp_feats: [B, 3, mem_shape]
        """
        assert len(style_ids) == len(comp_addrs) == len(comp_feats), "Input sizes are different"

        # batch iter
        for style_id, comp_addrs_per_char, comp_feats_per_char in zip(style_ids,
                                                                      comp_addrs,
                                                                      comp_feats):
            # comp iter
            for comp_addr, comp_feat in zip(comp_addrs_per_char, comp_feats_per_char):
                self.write_point(style_id, comp_addr, comp_feat)

    def read(self, style_ids, comp_addrs, reduction='mean'):
        """ Batch read

        Args:
            style_ids: [B]
            comp_addrs: [B, 3]
            reduction: reduction method if multiple features exist in sample memory address:
                       ['mean' (default), 'first', 'rand', 'none']
        """
        out = []
        for style_id, comp_addrs_per_char in zip(style_ids, comp_addrs):
            char_feats = []
            for comp_addr in comp_addrs_per_char:
                comp_feat = self.read_point(style_id, comp_addr, reduction)
                char_feats.append(comp_feat)

            char_feats = torch.stack(char_feats)  # [3, mem_shape]
            out.append(char_feats)

        out = torch.stack(out)  # [B, 3, mem_shape]
        return out

    def write_point(self, style_id, comp_addr, data):
        self.memory.setdefault(style_id.item(), {}) \
                   .setdefault(comp_addr.item(), []) \
                   .append(data)

    def read_point(self, style_id, comp_addr, reduction='mean'):
        """ Point read """
        comp_feats = self.memory[style_id.item()][comp_addr.item()]
        return self.reduce_features(comp_feats, reduction)

    def reduce_features(self, feats, reduction='mean'):
        if len(feats) == 1:
            return feats[0]

        if reduction == 'mean':
            return torch.stack(feats).mean(dim=0)
        elif reduction == 'first':
            return feats[0]
        elif reduction == 'rand':
            return np.random.choice(feats)
        elif reduction == 'none':
            return feats
        else:
            raise ValueError(reduction)

    def reset(self):
        self.memory = {}

    def reset_batch(self, style_ids, comp_addrs):
        for style_id, comp_addrs_per_char in zip(style_ids, comp_addrs):
            for comp_addr in comp_addrs_per_char:
                self.reset_point(style_id, comp_addr)

    def reset_point(self, style_id, comp_addr):
        self.memory[style_id.item()].pop(comp_addr.item())


class PersistentMemory(nn.Module):
    def __init__(self, n_comps, mem_shape):
        """
        Args:
            mem_shape: (C, H, W) tuple (3-elem)
        """
        super().__init__()
        self.shape = mem_shape

        self.bias = nn.Parameter(torch.randn(n_comps, *mem_shape))
        C = mem_shape[0]
        self.hypernet = nn.Sequential(
            ConvBlock(C, C),
            ConvBlock(C, C),
            ConvBlock(C, C)
        )

    def read(self, comp_addrs):
        b = self.bias[comp_addrs]  # [B, 3, mem_shape]

        return b

    def forward(self, x, comp_addrs):
        """
        Args:
            x: [B, 3, *mem_shape]
            comp_addr: [B, 3]
        """
        b = self.read(comp_addrs)  # [B, 3, *mem_shape] * 2

        B = b.size(0)
        b = b.flatten(0, 1)
        b = self.hypernet(b)
        b = split_dim(b, 0, B)

        return x + b


class Memory(nn.Module):
    # n_components: # of total comopnents. 19 + 21 + 28 = 68 in kr.
    STYLE_ADDR = -1

    def __init__(self, mem_shape, n_comps, persistent, language):
        """
        Args:
            mem_shape (tuple [3]):
                memory shape in (C, H, W) tuple, which is same as encoded feature shape
            n_comps: # of total components, which identify persistent memory size
        """
        super().__init__()
        self.dynamic_memory = DynamicMemory()
        self.mem_shape = mem_shape
        self.persistent = persistent
        self.language = language
        if persistent:
            self.persistent_memory = PersistentMemory(n_comps, mem_shape)

    def write(self, style_ids, comp_ids, comp_feats):
        """ Write data into dynamic memory """
        comp_addrs = comp_id_to_addr(comp_ids, self.language)
        self.dynamic_memory.write(style_ids, comp_addrs, comp_feats)

    def read(self, style_ids, comp_ids):
        """ Read data from memory (dynamic w/ or w/o persistent)

        Args:
            comp_ids [B, 3]
        """
        # memory shape = [B, 3, mem_shape]
        comp_addrs = comp_id_to_addr(comp_ids, self.language)
        mem = self.dynamic_memory.read(style_ids, comp_addrs)
        if self.persistent:
            mem = self.persistent_memory(mem, comp_addrs)

        return mem

    def reset_style(self, style_ids):
        style_addrs = self.get_style_addr(len(style_ids))
        self.dynamic_memory.reset_batch(style_ids, style_addrs)

    def write_style(self, style_ids, style_codes):
        style_addrs = self.get_style_addr(len(style_ids))
        self.dynamic_memory.write(style_ids, style_addrs, style_codes.unsqueeze(1))

    def read_style(self, style_ids):
        style_addrs = self.get_style_addr(len(style_ids))
        return self.dynamic_memory.read(style_ids, style_addrs).squeeze(1)

    def get_style_addr(self, N):
        return torch.full([N, 1], self.STYLE_ADDR, dtype=torch.long)

    def reset_dynamic(self):
        """ Reset dynamic memory """
        self.dynamic_memory.reset()
