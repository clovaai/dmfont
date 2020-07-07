"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import random
from itertools import product

import numpy as np
import torch
from torch.utils.data import Dataset

from .kor_decompose import decompose, compose, COMPLETE_SET
from .samplers import StyleSampler
from .data_utils import rev_dict, sample, get_fonts, get_union_chars


class MAStyleFirstDataset(Dataset):
    """ Sampling style chars first and then generating target chars
        by combination of style components
    """
    def __init__(self, data, style_sampler, n_sample_min=1, n_sample_max=999,
                 f_mult=800, transform=None, content_font=None):
        """
        Args:
            style_sampler: style sampler with data source. avail fonts and avail chars are
                           determined by `style_sampler.avails`.
            n_sample_min: minimum # of target chars per 3 style chars.
            n_sample_max: maximum # of target chars per 3 style chars.
            f_mult: #fonts multiplier for full-batch
            transform: image transform. If not given, use data.transform as default.
        """
        self.data = data
        self.style_sampler = style_sampler
        self.avails = style_sampler.avails
        self.R = style_sampler.R
        self.n_sample_min = n_sample_min
        self.n_sample_max = n_sample_max
        self.f_mult = f_mult
        self.transform = transform
        self.content_font = content_font

        self.fonts = get_fonts(self.avails)
        self.chars = get_union_chars(self.avails)
        self.n_fonts = len(self.fonts)
        self.n_chars = len(self.chars)
        self.font2idx = rev_dict(self.fonts)
        self.char2idx = rev_dict(self.chars)
        self.n_avails = sum(len(chars) for chars in self.avails.values())

    def get_avail_chars(self, font_name, style_chars):
        avail_chars = set(self.avails[font_name])
        avail_chars = avail_chars - set(style_chars)

        return avail_chars

    def get_component_combinations(self, components, avail_chars, style_comp_ids=None):
        """ Generate all component combinations within avail_chars
        Args:
            style_comp_ids: style component ids for excluding duplication (if needed)
        """
        trg_comp_ids = []
        for cho, jung, jong in product(*components):
            char = compose(cho, jung, jong)
            if char not in avail_chars:
                continue

            ids = np.asarray([cho, jung, jong])
            # exclude duplicated components:
            # exclude target chars which has duplicate source style char
            # e.g.) src ["성", "공", "해"] => trg ["송"]  (duplicated)
            if ((style_comp_ids == ids).sum(axis=1) >= 2).any():
                continue

            trg_comp_ids.append(ids)

        return trg_comp_ids

    def check_and_sample(self, trg_comp_ids):
        n_sample = len(trg_comp_ids)
        if n_sample > self.n_sample_max:
            trg_comp_ids = sample(trg_comp_ids, self.n_sample_max)
        elif n_sample < self.n_sample_min:
            return None

        return trg_comp_ids

    def __getitem__(self, index):
        font_idx = index % self.n_fonts
        font_name = self.fonts[font_idx]
        while True:
            ####################################################
            # 1. sample styles
            ####################################################
            style_imgs, style_chars = self.style_sampler.get(font_name, ret_values=True)
            style_comp_ids = [decompose(char) for char in style_chars]
            chos, jungs, jongs = list(map(set, zip(*style_comp_ids)))

            # fullcomb
            if not (len(chos) == len(jungs) == len(jongs) == self.R):
                continue

            style_comp_ids = np.asarray(style_comp_ids)

            ####################################################
            # 2. sample targets from style components
            ####################################################
            avail_chars = self.get_avail_chars(font_name, style_chars)
            trg_comp_ids = self.get_component_combinations(
                (chos, jungs, jongs), avail_chars, style_comp_ids
            )
            trg_comp_ids = np.asarray(trg_comp_ids)
            trg_comp_ids = self.check_and_sample(trg_comp_ids)
            if trg_comp_ids is None:
                continue

            ####################################################
            # 3. setup chars, font_ids, char_ids and images
            ####################################################
            trg_chars = [compose(*comp_id) for comp_id in trg_comp_ids]
            trg_imgs = torch.cat([
                self.data.get(font_name, char, transform=self.transform)
                for char in trg_chars
            ])

            style_char_ids = [self.char2idx[ch] for ch in style_chars]
            trg_char_ids = [self.char2idx[ch] for ch in trg_chars]

            n_styles = len(style_chars)
            n_trgs = len(trg_chars)

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
    """
    MAStyleFirstDatset samples source style characters first and then determines target characters.
    In contrast, MATargetFirstDataset samples target characters first and then
    determines source style characters.
    """
    def __init__(self, target_fc, style_avails, style_data, n_max_match=3, transform=None,
                 ret_targets=False, first_shuffle=False, content_font=None):
        """ TargetFirstDataset can use out-of-avails target chars,
            so long as its components could be represented in avail chars.

        Args:
            target_fc[font_name] = target_chars
            style_avails[font_name] = avail_style_chars
            style_data: style_data getter
            n_max_match: maximum-allowed matches between style char and target char.
                         n_max_match=3 indicates that style_char == target_char is possible.
            transform: image transform. If not given, use data.transform as default.
            ret_targets: return target images also
            first_shuffle: shuffle item list
        """
        self.target_fc = target_fc
        self.style_avails = style_avails
        self.style_avail_comps_list = {
            fname: [decompose(char) for char in char_list]
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
        self.font2idx = {font_name: i for i, font_name in enumerate(self.target_fc.keys())}

    def sample_style_char(self, font_name, trg_char):
        """ sample style char from target char within avail style chars """
        def is_allowed_matches(arr1, arr2):
            """ check # of matched ids
            return count(arr1 == arr2) <= self.n_max_match
            """
            if self.n_max_match >= 3:
                return True

            n_matched = sum(v1 == v2 for v1, v2 in zip(arr1, arr2))

            return n_matched <= self.n_max_match

        trg_comp_ids = decompose(trg_char)
        style_chars = []
        style_comps_list = []
        for i, _ in enumerate(trg_comp_ids):
            avail_comps_list = list(
                filter(
                    lambda comp_ids: comp_ids[i] == trg_comp_ids[i] \
                            and is_allowed_matches(comp_ids, trg_comp_ids),
                    self.style_avail_comps_list[font_name]
                )
            )
            style_comp_ids = random.choice(avail_comps_list)
            style_char = compose(*style_comp_ids)

            style_chars.append(style_char)
            style_comps_list.append(style_comp_ids)

        return style_chars, style_comps_list

    def __getitem__(self, index):
        font_name, trg_char = self.fcs[index]
        font_idx = self.font2idx[font_name]

        style_chars, style_comp_ids = self.sample_style_char(font_name, trg_char)
        style_imgs = torch.cat([
            self.style_data.get(font_name, char, transform=self.transform)
            for char in style_chars
        ])

        trg_comp_ids = [decompose(trg_char)]

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
            ret += (torch.cat(trg_imgs).unsqueeze_(1), )

        return ret


def filter_complete_chars(chars):
    return sorted(set(chars) & COMPLETE_SET)


def get_ma_dataset(hdf5_data, avail_fonts, avail_chars=None, transform=None, **kwargs):
    if avail_chars:
        avail_chars = set(avail_chars)
    R_style = 3

    avails = {}
    for fname in avail_fonts:
        chars = hdf5_data.get_avail_chars(fname)
        if avail_chars:
            chars = set(chars) & avail_chars
        avails[fname] = filter_complete_chars(chars)

    style_sampler = StyleSampler(R_style, avails, hdf5_data)
    dset = MAStyleFirstDataset(hdf5_data, style_sampler, transform=transform, **kwargs)

    return dset, MAStyleFirstDataset.collate_fn


def get_ma_val_dataset(hdf5_data, fonts, chars, style_avails, n_max_match, transform, **kwargs):
    target_fc = {font_name: chars for font_name in fonts}
    dset = MATargetFirstDataset(
        target_fc, style_avails, hdf5_data, n_max_match, transform=transform, **kwargs
    )

    return dset, MATargetFirstDataset.collate_fn
