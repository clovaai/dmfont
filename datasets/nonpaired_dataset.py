"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
from torch.utils.data import Dataset

from . import kor_decompose as kor
from . import thai_decompose as thai


class EncodeDataset(Dataset):
    def __init__(self, font_name, chars, data, language, style_id=0, transform=None):
        self.fname = font_name
        self.data = data
        self.chars = chars
        self.style_id = style_id
        self.transform = transform
        self.language = language

        if language == 'kor':
            self.decompose = kor.decompose
        elif language == 'thai':
            self.decompose = thai.decompose_ids
        else:
            raise ValueError(language)

    def __getitem__(self, index):
        style_char = self.chars[index]
        style_comp_ids = self.decompose(style_char)
        style_img = self.data.get(self.fname, style_char, transform=self.transform)

        return (
            self.style_id,
            torch.as_tensor(style_comp_ids),
            style_img
        )

    def __len__(self):
        return len(self.chars)


class DecodeDataset(Dataset):
    def __init__(self, chars, language, style_id=0):
        """
        Args:
            chars: target characters
            language
            style_id: Use different style id for different reference style set
        """
        self.chars = chars
        self.language = language
        self.style_id = style_id

        if language == 'kor':
            self.decompose = kor.decompose
        elif language == 'thai':
            raise NotImplementedError()
        else:
            raise ValueError(language)

    def __getitem__(self, index):
        char = self.chars[index]
        trg_comp_ids = self.decompose(char)

        return (
            self.style_id,
            torch.as_tensor(trg_comp_ids)
        )

    def __len__(self):
        return len(self.chars)
