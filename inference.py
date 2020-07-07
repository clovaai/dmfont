"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
from torch.utils.data import DataLoader
from datasets import get_ma_val_dataset
from datasets.nonpaired_dataset import EncodeDataset, DecodeDataset


def infer(gen, loader):
    outs = []
    for style_ids, style_comp_ids, style_imgs, trg_ids, trg_comp_ids, content_imgs \
            in loader:
        style_ids = style_ids.cuda()
        style_comp_ids = style_comp_ids.cuda()
        style_imgs = style_imgs.cuda()
        trg_ids = trg_ids.cuda()
        trg_comp_ids = trg_comp_ids.cuda()

        gen.encode_write(style_ids, style_comp_ids, style_imgs)
        out = gen.read_decode(trg_ids, trg_comp_ids)

        outs.append(out.detach().cpu())

    return torch.cat(outs)  # [B, 1, 128, 128]; B = #fonts * #chars


def get_val_loader(data, fonts, chars, style_avails, transform, content_font, language,
                   B=32, n_max_match=3, n_workers=2):
    val_dset, collate_fn = get_ma_val_dataset(
        data, fonts, chars, style_avails, n_max_match, transform=transform,
        content_font=content_font, language=language
    )
    loader = DataLoader(val_dset, batch_size=B, shuffle=False,
                        num_workers=n_workers, collate_fn=collate_fn)

    return loader


def infer_2stage(gen, encode_loader, decode_loader, reset_memory=True):
    """ 2-stage infer; encode first, decode second """
    # stage 1. encode
    if reset_memory:
        gen.reset_dynamic_memory()

    for style_ids, style_comp_ids, style_imgs in encode_loader:
        style_ids = style_ids.cuda()
        style_comp_ids = style_comp_ids.cuda()
        style_imgs = style_imgs.cuda()

        gen.encode_write(style_ids, style_comp_ids, style_imgs, reset_dynamic_memory=False)

    # stage 2. decode
    outs = []
    for trg_ids, trg_comp_ids in decode_loader:
        trg_ids = trg_ids.cuda()
        trg_comp_ids = trg_comp_ids.cuda()

        out = gen.read_decode(trg_ids, trg_comp_ids)

        outs.append(out.detach().cpu())

    return torch.cat(outs)


def get_val_encode_loader(data, font_name, encode_chars, language, transform, B=32, num_workers=2,
                          style_id=0):
    encode_dset = EncodeDataset(
        font_name, encode_chars, data, language=language, transform=transform, style_id=style_id
    )
    loader = DataLoader(encode_dset, batch_size=B, shuffle=False, num_workers=num_workers)

    return loader


def get_val_decode_loader(chars, language, B=32, num_workers=2, style_id=0):
    decode_dset = DecodeDataset(chars, language=language, style_id=style_id)
    loader = DataLoader(decode_dset, batch_size=B, shuffle=False, num_workers=num_workers)

    return loader
