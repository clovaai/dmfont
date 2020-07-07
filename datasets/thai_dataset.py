"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from . import thai_decompose as thai
from .data_utils import rev_dict, get_fonts, get_union_chars


def product4_no_dup(consonants, uppers, highests, lowers):
    for i, co in enumerate(consonants):
        for j, up in enumerate(uppers):
            if i == j:
                continue
            for k, hi in enumerate(highests):
                if k in (i, j):
                    continue
                for l, lo in enumerate(lowers):
                    if l in (i, j, k):
                        continue
                    yield (co, up, hi, lo)


class MAStyleFirstDataset(Dataset):
    def __init__(self, data, fonts, chars, n_sample_min, n_sample_max,
                 f_mult=800, transform=None, content_font=None):
        self.data = data
        self.n_sample_min = n_sample_min
        self.n_sample_max = n_sample_max
        self.f_mult = f_mult
        self.transform = transform
        self.content_font = content_font

        self.fonts = fonts
        self.chars = chars
        self.n_fonts = len(fonts)
        self.n_chars = len(self.chars)
        self.font2idx = rev_dict(self.fonts)
        self.char2idx = rev_dict(self.chars)

        self.n_avails = self.n_fonts * self.n_chars
        # for compatibility
        self.avails = {
            fname: chars
            for fname in self.fonts
        }

    def sample_style_chars(self, n_styles):
        # Sampling performance can be improved but which is not the bottleneck, at least now.
        while True:
            # sample without replacement => fullcomb
            consonants = np.random.choice(thai.CONSONANTS, n_styles, replace=False)
            uppers = np.random.choice(thai.UPPERS, n_styles, replace=False)
            highests = np.random.choice(thai.HIGHESTS, n_styles, replace=False)
            lowers = np.random.choice(thai.LOWERS, n_styles, replace=False)

            style_chars = []
            style_ords = []
            for c, u, h, l in zip(consonants, uppers, highests, lowers):
                char = thai.compose(c, u, h, l)
                if char not in self.chars:
                    break

                style_chars.append(char)
                style_ords.append((c, u, h, l))
            else:
                # for-loop is not broken -> all chars available
                break

        components = (consonants, uppers, highests, lowers)
        return style_chars, style_ords, components

    def sample_trg_chars(self, components, style_chars):
        combinations = list(product4_no_dup(*components))
        np.random.shuffle(combinations)
        trg_chars = []
        trg_ords = []
        for c, u, h, l in combinations:
            char = thai.compose(c, u, h, l)
            # exclude style chars from target chars
            if char in style_chars or char not in self.chars:
                continue

            trg_chars.append(char)
            trg_ords.append((c, u, h, l))

            if len(trg_chars) >= self.n_sample_max:
                break

        return trg_chars, trg_ords

    def __getitem__(self, index):
        font_idx = index % self.n_fonts
        font_name = self.fonts[font_idx]

        n_styles = 4
        while True:
            # 1. sample style components
            style_chars, style_ords, components = self.sample_style_chars(n_styles)

            # 2. sample targets from style components
            trg_chars, trg_ords = self.sample_trg_chars(components, style_chars)
            if len(trg_chars) >= self.n_sample_min:
                break

        # 3. setup rest
        style_char_ids = [self.char2idx[ch] for ch in style_chars]
        style_comp_ids = thai.ord2idx_2d(style_ords)
        style_imgs = torch.cat([self.data.get(font_name, char) for char in style_chars])
        n_trgs = len(trg_chars)
        trg_char_ids = [self.char2idx[ch] for ch in trg_chars]
        trg_comp_ids = thai.ord2idx_2d(trg_ords)
        trg_imgs = torch.cat([self.data.get(font_name, char) for char in trg_chars])

        font_idx = torch.as_tensor(font_idx)
        ret = (
            font_idx.repeat(n_styles),
            torch.as_tensor(style_char_ids),
            torch.as_tensor(style_comp_ids),
            style_imgs,
            font_idx.repeat(n_trgs),
            torch.as_tensor(trg_char_ids),
            torch.as_tensor(trg_comp_ids),
            trg_imgs
        )

        if self.content_font:
            content_imgs = torch.cat([
                self.data.get(self.content_font, char, transform=self.transform)
                for char in trg_chars
            ])
            ret += (content_imgs,)

        return ret

    def __len__(self):
        return self.n_fonts * self.f_mult

    @staticmethod
    def collate_fn(batch):
        (style_ids, style_char_ids, style_comp_ids, style_imgs,
         trg_ids, trg_char_ids, trg_comp_ids, trg_imgs, *left) = zip(*batch)

        ret = (
            torch.cat(style_ids),
            torch.cat(style_char_ids),
            torch.cat(style_comp_ids),
            torch.cat(style_imgs).unsqueeze_(1),
            torch.cat(trg_ids),
            torch.cat(trg_char_ids),
            torch.cat(trg_comp_ids),
            torch.cat(trg_imgs).unsqueeze_(1)
        )

        if left:
            assert len(left) == 1
            content_imgs = left[0]
            ret += (
                torch.cat(content_imgs).unsqueeze_(1),
            )

        return ret


class MATargetFirstDataset(Dataset):
    def __init__(self, target_fc, style_avails, style_data, n_max_match=4, transform=None,
                 ret_targets=False, first_shuffle=False, content_font=None):
        """ TargetFirstDataset can use out-of-avails target chars,
            so long as its components could be represented in avail chars.

        Args:
            target_fc[font_name] = target_chars
            style_avails[font_name] = avail_style_chars
            style_data: style_data getter
            n_max_match: maximum-allowed matches between style char and target char.
                         n_max_match=4 indicates that style_char == target_char is possible.
            transform: image transform. If not given, use data.transform as default.
            ret_targets: return target images also
            first_shuffle: shuffle item list
        """
        self.target_fc = target_fc
        self.style_avails = style_avails
        self.style_avail_comps_list = {
            fname: [thai.decompose_ords(char) for char in char_list]
            for fname, char_list in style_avails.items()
        }
        self.n_max_match = n_max_match
        self.transform = transform
        self.ret_targets = ret_targets
        self.content_font = content_font

        self.style_data = style_data
        self.fcs = [
            (font_name, char)
            for font_name, char_list in target_fc.items()
            for char in char_list
        ]
        if first_shuffle:
            np.random.shuffle(self.fcs)
        self.fonts = get_fonts(self.target_fc)
        self.chars = get_union_chars(self.target_fc)
        self.font2idx = rev_dict(self.fonts)

    def sample_style_char(self, font_name, trg_char):
        """ sample style char from target char within avail style chars """
        def is_allowed_matches(arr1, arr2):
            """ check # of matched ids
            return count(arr1 == arr2) <= self.n_max_match
            """
            if self.n_max_match >= 4:
                return True

            n_matched = sum(v1 == v2 for v1, v2 in zip(arr1, arr2))

            return n_matched <= self.n_max_match

        trg_comp_ords = thai.decompose_ords(trg_char)
        style_chars = []
        style_comps_list = []
        for i, _ in enumerate(trg_comp_ords):
            avail_comps_list = list(
                filter(
                    lambda comp_ords: comp_ords[i] == trg_comp_ords[i] \
                            and is_allowed_matches(comp_ords, trg_comp_ords),
                    self.style_avail_comps_list[font_name]
                )
            )
            style_comp_ords = random.choice(avail_comps_list)
            style_char = thai.compose(*style_comp_ords)

            style_chars.append(style_char)
            style_comps_list.append(style_comp_ords)

        return style_chars, thai.ord2idx_2d(style_comps_list)

    def __getitem__(self, index):
        font_name, trg_char = self.fcs[index]
        font_idx = self.font2idx[font_name]

        style_chars, style_comp_ids = self.sample_style_char(font_name, trg_char)
        style_imgs = torch.cat([
            self.style_data.get(font_name, char, transform=self.transform)
            for char in style_chars
        ])

        trg_comp_ords = [thai.decompose_ords(trg_char)]
        trg_comp_ids = thai.ord2idx_2d(trg_comp_ords)

        n_styles = len(style_chars)
        font_idx = torch.as_tensor(font_idx)

        style_ids = font_idx.repeat(n_styles)
        trg_ids = font_idx.repeat(1)

        content_img = self.style_data.get(self.content_font, trg_char, transform=self.transform)

        ret = (
            style_ids,
            torch.as_tensor(style_comp_ids),
            style_imgs,
            trg_ids,
            torch.as_tensor(trg_comp_ids),
            content_img
        )

        if self.ret_targets:
            trg_img = self.style_data.get(font_name, trg_char, transform=self.transform)
            ret += (trg_img, )

        return ret

    def __len__(self):
        return len(self.fcs)

    @staticmethod
    def collate_fn(batch):
        style_ids, style_comp_ids, style_imgs, trg_ids, trg_comp_ids, content_imgs, *left = \
            list(zip(*batch))

        ret = (
            torch.cat(style_ids),
            torch.cat(style_comp_ids),
            torch.cat(style_imgs).unsqueeze_(1),
            torch.cat(trg_ids),
            torch.cat(trg_comp_ids),
            torch.cat(content_imgs).unsqueeze_(1)
        )

        if left:
            assert len(left) == 1
            trg_imgs = left[0]
            ret += torch.cat(trg_imgs).unsqueeze_(1),

        return ret


def get_ma_dataset(hdf5_data, avail_fonts, avail_chars=None, transform=None, **kwargs):
    if not avail_chars:
        avail_chars = list(thai.complete_chars())
    dset = MAStyleFirstDataset(hdf5_data, avail_fonts, avail_chars, transform=transform, **kwargs)

    return dset, MAStyleFirstDataset.collate_fn


def get_ma_val_dataset(hdf5_data, fonts, chars, style_avails, n_max_match, transform, **kwargs):
    target_fc = {font_name: chars for font_name in fonts}
    dset = MATargetFirstDataset(
        target_fc, style_avails, hdf5_data, n_max_match, transform=transform, **kwargs
    )

    return dset, MATargetFirstDataset.collate_fn
