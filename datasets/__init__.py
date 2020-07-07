"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from . import kor_dataset
from . import thai_dataset
from .fcdata import HDF5Data
from .data_utils import (
    cyclize, sample, uniform_sample,
    get_fonts_unionchars, get_union_chars, get_intersection_chars
)


def get_ma_dataset(*args, language=None, **kwargs):
    if language == 'kor':
        return kor_dataset.get_ma_dataset(*args, **kwargs)
    elif language == 'thai':
        return thai_dataset.get_ma_dataset(*args, **kwargs)
    else:
        raise ValueError(language)


def get_ma_val_dataset(*args, language=None, **kwargs):
    if language == 'kor':
        return kor_dataset.get_ma_val_dataset(*args, **kwargs)
    elif language == 'thai':
        return thai_dataset.get_ma_val_dataset(*args, **kwargs)
    else:
        raise ValueError(language)
