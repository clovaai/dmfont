"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
from .data_utils import sample, uniform_indices


class Sampler:
    def __init__(self, R, avails, data):
        self.R = R
        self.avails = avails
        self.data = data

    def get_item(self, key, value):
        raise NotImplementedError()

    def get(self, key, ex_values=None, ret_values=False):
        """ Random sampling """
        values = self.avails[key]
        values = sample(values, self.R, ex_values)
        images = torch.cat([self.get_item(key, value) for value in values])

        if ret_values:
            return images, values

        return images

    def get_uniform(self, key, indices=None, st=None):
        values = self.avails[key]
        if indices is None:
            indices = uniform_indices(len(values), self.R, st)

        images, vals = [], []
        for idx in indices:
            value = values[idx]
            img = self.get_item(key, value)
            vals.append(value)
            images.append(img)

        return torch.cat(images), vals


class StyleSampler(Sampler):
    """ Return various content (char) but single style (font): which represent style """
    def get_item(self, font_name, char):
        return self.data.get(font_name, char)


class ContentSampler(Sampler):
    """ Return various style (font) but single content (char): which represent content """
    def get_item(self, char, font_name):
        return self.data.get(font_name, char)
