"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import h5py as h5
from torchvision import transforms

from . import thai_decompose as thai


class FCData:
    """ FontChar data
    Data module provides image access by font_name and char in get method
    """
    def get(self, font_name, char):
        raise NotImplementedError()


class HDF5Data(FCData):
    def __init__(self, hdf5_paths, transform=None, language='kor'):
        """
        cmap = {
            font_name: {
                char: idx,
                ...
            },
            ...
        }
        """
        self.paths = hdf5_paths
        self.transform = transform or transforms.ToTensor()
        self.language = language

        self.fn2path = {}
        self.cmap = {}
        self.n_items = 0

        # chars: union of all chars
        for path in hdf5_paths:
            with h5.File(path, 'r') as f:
                font_name = f['dataset'].attrs['font_name']
                self.fn2path[font_name] = path
                # [:] for batch read
                char2idx = self.make_char2idx(f['dataset']['chars'][:])
                self.cmap[font_name] = char2idx

                self.n_items += len(char2idx)

        # indexing
        self.fonts = list(self.cmap.keys())

    def make_char2idx(self, chars):
        """ Generate char2idx map
        Args:
            chars [N] or [N, 4]
                for kor: [N]
                for thai: [N, 4]
        """
        if self.language == 'kor':
            # chars: 1d array [N]
            char2idx = {
                chr(ch): i
                for i, ch in enumerate(chars)
            }
        elif self.language == 'thai':
            # chars: 2d array [N, 4]
            char2idx = {
                thai.compose(*ch): i
                for i, ch in enumerate(chars)
            }
        else:
            raise ValueError(self.language)

        return char2idx

    def is_avail(self, font_name, char):
        if font_name not in self.cmap:
            return False

        return char in self.cmap[font_name]

    def get(self, font_name, char, default=None, transform=None):
        """
        Args:
            default: if not available (font_name, char) is given, return default.
            transform: image transform. If not given, use self.transform.
        """
        if default is not None:
            if not self.is_avail(font_name, char):
                return default

        path = self.fn2path[font_name]

        with h5.File(path, 'r') as f:
            cidx = self.cmap[font_name][char]
            image = f['dataset']['images'][cidx]

        transform = transform or self.transform

        return transform(image)

    def get_from_reffont(self, char):
        """ get character image from reference font """
        if self.language == 'kor':
            font_name = "D2Coding-Ver1.3.2-20180524.ttf"
        elif self.language == 'thai':
            font_name = "NotoSansThai-Regular.ttf"

        return self.get(font_name, char)

    def get_avail_chars(self, font_name):
        return list(self.cmap[font_name].keys())
