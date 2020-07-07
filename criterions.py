"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch.nn.functional as F


def hinge_d_loss(real, fake):
    return F.relu(1. - real).mean() + F.relu(1. + fake).mean()


def hinge_g_loss(real, fake):
    return -fake.mean()
