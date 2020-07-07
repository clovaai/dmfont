"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
from torchvision import utils as tv_utils
from PIL import Image


def make_merged_grid(batchs, merge_dim, **kwargs):
    """ Generate grid for each batch and integrate them with sep bar

    Args:
        merge_dim: 1 => height-wise merge, 2=> width-wise merge.
    """
    sep_bar_size = 12
    out = []
    for batch in batchs:
        grid = to_grid(batch, 'torch', **kwargs).cpu()
        shape = list(grid.size())
        shape[merge_dim] = sep_bar_size
        sep_bar = torch.zeros(*shape)
        out += [grid, sep_bar]

    # remove last sep_bar
    return torch.cat(out[:-1], dim=merge_dim)


def make_comparable_grid(*batches, nrow):
    assert all(len(batches[0]) == len(batch) for batch in batches[1:])
    N = len(batches[0])

    grids = []
    for i in range(0, N, nrow):
        rows = [batch[i:i+nrow] for batch in batches]
        row = torch.cat(rows)
        grid = to_grid(row, 'torch', nrow=nrow)
        grids.append(grid)

        C, _H, W = grid.shape
        sep_bar = torch.zeros(C, 10, W)
        grids.append(sep_bar)

    return torch.cat(grids[:-1], dim=1)


def normalize(tensor, eps=1e-5):
    """ Normalize tensor to [0, 1] """
    # eps=1e-5 is same as make_grid in torchvision.
    minv, maxv = tensor.min(), tensor.max()
    tensor = (tensor - minv) / (maxv - minv + eps)

    return tensor


def to_grid(tensor, to, **kwargs):
    """ Integrated functions of make_grid and save_image

    Possible conversions:
        torch: torch tensor [0, 1]
        numpy: numpy ndarr [0, 255]
        pil: PIL image
    """
    to = to.lower()

    grid = tv_utils.make_grid(tensor, **kwargs, normalize=True)
    if to == 'torch':
        return grid

    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    if to == 'numpy':
        return ndarr

    im = Image.fromarray(ndarr)
    if to == 'pil':
        return im

    raise ValueError("Not supported target format `{}`".format(to))


def save_tensor_to_image(tensor, filepath, scale=None):
    """ Save torch tensor to filepath
    Same as torchvision.save_image; only scale factor is difference.
    """
    tensor = normalize(tensor)
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    if ndarr.shape[-1] == 1:
        ndarr = ndarr.squeeze(-1)
    im = Image.fromarray(ndarr)
    if scale:
        size = tuple(map(lambda v: int(v*scale), im.size))
        im = im.resize(size, resample=Image.BILINEAR)
    im.save(filepath)


def textboxes(chars, data):
    images = [data.get_from_reffont(char) for char in chars]
    return images
