"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import sys
import json
from pathlib import Path
import argparse
import random

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import numpy as np
from sconf import Config, dump_args

from logger import Logger
from models import MACore, Discriminator, AuxClassifier
from models.modules import weights_init
from datasets import HDF5Data, get_ma_dataset, get_ma_val_dataset
import datasets.kor_decompose as kor
import datasets.thai_decompose as thai
import utils
from trainer import Trainer, load_checkpoint
from evaluator import Evaluator


def get_dset_loader(data, avail_fonts, avail_chars, transform, shuffle, cfg, content_font=None):
    dset, collate_fn = get_ma_dataset(
        data,
        avail_fonts,
        avail_chars=avail_chars,
        transform=transform,
        **cfg.get('dset_args', {}),
        content_font=content_font,
        language=cfg['language']
    )
    loader = DataLoader(dset, batch_size=cfg['batch_size'], shuffle=shuffle,
                        num_workers=cfg['n_workers'], collate_fn=collate_fn)

    return dset, loader


def get_val_dset_loader(data, avail_fonts, avail_chars, trn_avail_chars, transform,
                        batch_size, n_workers=2, n_max_match=3, content_font=None, language=None):
    style_avails = {
        font_name: trn_avail_chars for font_name in avail_fonts
    }
    dset, collate_fn = get_ma_val_dataset(
        data,
        avail_fonts,
        avail_chars,
        style_avails,
        n_max_match=n_max_match,
        transform=transform,
        ret_targets=True,
        first_shuffle=True,
        content_font=content_font,
        language=language
    )
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False,
                        num_workers=n_workers, collate_fn=collate_fn)

    return dset, loader


def setup_args_and_config():
    parser = argparse.ArgumentParser('MaHFG')
    parser.add_argument("name")
    parser.add_argument("config_paths", nargs="+")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--log_lv", default='info')
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--tb-image", default=False, action="store_true",
                        help="Write image log to tensorboard")
    parser.add_argument("--deterministic", default=False, action="store_true")

    args, left_argv = parser.parse_known_args()
    assert not args.name.endswith(".yaml")

    cfg = Config(*args.config_paths, colorize_modified_item=True)
    cfg.argv_update(left_argv)

    if args.debug:
        cfg['print_freq'] = 1
        cfg['tb_freq'] = 1
        cfg['max_iter'] = 10
        #  cfg['save'] = 'last'
        cfg['val_freq'] = 5
        cfg['save_freq'] = 10
        args.name += "_debug"
        args.tb_image = True
        args.log_lv = 'debug'

    cfg['data_dir'] = Path(cfg['data_dir'])

    assert cfg['save_freq'] % cfg['val_freq'] == 0

    return args, cfg


def setup_language_dependent(cfg):
    if cfg['language'] == 'kor':
        content_font = "NanumBarunpenR.ttf"
        n_comp_types = 3  # cho, jung, jong
        n_comps = kor.N_COMPONENTS
    elif cfg['language'] == 'thai':
        content_font = "NotoSansThai-Regular.ttf"
        n_comp_types = 4  # consonant, upper, highest, lower
        n_comps = thai.N_COMPONENTS
    else:
        raise ValueError(cfg['language'])

    return content_font, n_comp_types, n_comps


def setup_data(cfg, val_transform):
    """ setup data, meta_data, and check cross-validation flag

    Return (tuple): (data, meta_data)
        data (HDF5Data)
        meta_data (dict)
    """
    hdf5_paths = list(cfg['data_dir'].glob("*.hdf5"))
    hdf5_data = HDF5Data(hdf5_paths, val_transform, language=cfg['language'])

    # setup meta data
    meta = json.load(open(cfg['data_meta']))

    return hdf5_data, meta


def setup_cv_dset_loader(hdf5_data, meta, val_transform, n_comp_types, content_font, cfg):
    trn_chars = meta['train']['chars']
    batch_size = cfg['batch_size'] * 3
    n_workers = cfg['n_workers']
    n_max_match = n_comp_types  # for validation dset
    # seen fonts, unseen chars -> same as original unseen validation
    sfuc_dset, sfuc_loader = get_val_dset_loader(
        hdf5_data, meta['train']['fonts'], meta['valid']['chars'], trn_chars, val_transform,
        batch_size, n_workers, n_max_match, content_font, cfg['language']
    )
    # unseen fonts, seen chars
    ufsc_dset, ufsc_loader = get_val_dset_loader(
        hdf5_data, meta['valid']['fonts'], meta['train']['chars'], trn_chars, val_transform,
        batch_size, n_workers, n_max_match, content_font, cfg['language']
    )
    # unseen fonts, unseen chars
    ufuc_dset, ufuc_loader = get_val_dset_loader(
        hdf5_data, meta['valid']['fonts'], meta['valid']['chars'], trn_chars, val_transform,
        batch_size, n_workers, n_max_match, content_font, cfg['language']
    )
    # setup val_loaders
    val_loaders = {
        "SeenFonts-UnseenChars": sfuc_loader,
        "UnseenFonts-SeenChars": ufsc_loader,
        "UnseenFonts-UnseenChars": ufuc_loader
    }

    return val_loaders


def main():
    ############################
    # argument setup
    ############################
    args, cfg = setup_args_and_config()

    if args.show:
        print("### Run Argv:\n> {}".format(' '.join(sys.argv)))
        print("### Run Arguments:")
        s = dump_args(args)
        print(s + '\n')
        print("### Configs:")
        print(cfg.dumps())
        sys.exit()

    timestamp = utils.timestamp()
    unique_name = "{}_{}".format(timestamp, args.name)
    cfg['unique_name'] = unique_name  # for save directory
    cfg['name'] = args.name

    utils.makedirs('logs')
    utils.makedirs(Path('checkpoints', unique_name))

    # logger
    logger_path = Path('logs', f"{unique_name}.log")
    logger = Logger.get(file_path=logger_path, level=args.log_lv, colorize=True)

    # writer
    image_scale = 0.6
    writer_path = Path('runs', unique_name)
    if args.tb_image:
        writer = utils.TBWriter(writer_path, scale=image_scale)
    else:
        image_path = Path('images', unique_name)
        writer = utils.TBDiskWriter(writer_path, image_path, scale=image_scale)

    # log default informations
    args_str = dump_args(args)
    logger.info("Run Argv:\n> {}".format(' '.join(sys.argv)))
    logger.info("Args:\n{}".format(args_str))
    logger.info("Configs:\n{}".format(cfg.dumps()))
    logger.info("Unique name: {}".format(unique_name))

    # seed
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])

    if args.deterministic:
        #  https://discuss.pytorch.org/t/how-to-get-deterministic-behavior/18177/16
        #  https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cfg['n_workers'] = 0
        logger.info("#" * 80)
        logger.info("# Deterministic option is activated !")
        logger.info("#" * 80)
    else:
        torch.backends.cudnn.benchmark = True

    ############################
    # setup dataset & loader
    ############################
    logger.info("Get dataset ...")

    # setup language dependent values
    content_font, n_comp_types, n_comps = setup_language_dependent(cfg)

    # setup transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # setup data
    hdf5_data, meta = setup_data(cfg, transform)

    # setup dataset
    trn_dset, loader = get_dset_loader(
        hdf5_data, meta['train']['fonts'], meta['train']['chars'], transform, True, cfg,
        content_font=content_font
    )

    logger.info("### Training dataset ###")
    logger.info("# of avail fonts = {}".format(trn_dset.n_fonts))
    logger.info(f"Total {len(loader)} iterations per epochs")
    logger.info("# of avail items = {}".format(trn_dset.n_avails))
    logger.info(f"#fonts = {trn_dset.n_fonts}, #chars = {trn_dset.n_chars}")

    val_loaders = setup_cv_dset_loader(
        hdf5_data, meta, transform, n_comp_types, content_font, cfg
    )
    sfuc_loader = val_loaders['SeenFonts-UnseenChars']
    sfuc_dset = sfuc_loader.dataset
    ufsc_loader = val_loaders['UnseenFonts-SeenChars']
    ufsc_dset = ufsc_loader.dataset
    ufuc_loader = val_loaders['UnseenFonts-UnseenChars']
    ufuc_dset = ufuc_loader.dataset

    logger.info("### Cross-validation datasets ###")
    logger.info(
        "Seen fonts, Unseen chars | "
        "#items = {}, #fonts = {}, #chars = {}, #steps = {}".format(
            len(sfuc_dset), len(sfuc_dset.fonts), len(sfuc_dset.chars), len(sfuc_loader)))
    logger.info(
        "Unseen fonts, Seen chars | "
        "#items = {}, #fonts = {}, #chars = {}, #steps = {}".format(
            len(ufsc_dset), len(ufsc_dset.fonts), len(ufsc_dset.chars), len(ufsc_loader)))
    logger.info(
        "Unseen fonts, Unseen chars | "
        "#items = {}, #fonts = {}, #chars = {}, #steps = {}".format(
            len(ufuc_dset), len(ufuc_dset.fonts), len(ufuc_dset.chars), len(ufuc_loader)))

    ############################
    # build model
    ############################
    logger.info("Build model ...")
    # generator
    g_kwargs = cfg.get('g_args', {})
    gen = MACore(
        1, cfg['C'], 1, **g_kwargs, n_comps=n_comps, n_comp_types=n_comp_types,
        language=cfg['language']
    )
    gen.cuda()
    gen.apply(weights_init(cfg['init']))

    d_kwargs = cfg.get('d_args', {})
    disc = Discriminator(cfg['C'], trn_dset.n_fonts, trn_dset.n_chars, **d_kwargs)
    disc.cuda()
    disc.apply(weights_init(cfg['init']))

    if cfg['ac_w'] > 0.:
        C = gen.mem_shape[0]
        aux_clf = AuxClassifier(C, n_comps, **cfg['ac_args'])
        aux_clf.cuda()
        aux_clf.apply(weights_init(cfg['init']))
    else:
        aux_clf = None
        assert cfg['ac_gen_w'] == 0., "ac_gen loss is only available with ac loss"

    # setup optimizer
    g_optim = optim.Adam(gen.parameters(), lr=cfg['g_lr'], betas=cfg['adam_betas'])
    d_optim = optim.Adam(disc.parameters(), lr=cfg['d_lr'], betas=cfg['adam_betas'])
    ac_optim = optim.Adam(aux_clf.parameters(), lr=cfg['g_lr'], betas=cfg['adam_betas']) \
               if aux_clf is not None else None

    # resume checkpoint
    st_step = 1
    if args.resume:
        st_step, loss = load_checkpoint(args.resume, gen, disc, aux_clf, g_optim, d_optim, ac_optim)
        logger.info("Resumed checkpoint from {} (Step {}, Loss {:7.3f})".format(
            args.resume, st_step-1, loss))

    ############################
    # setup validation
    ############################
    evaluator = Evaluator(
        hdf5_data, trn_dset.avails, logger, writer, cfg['batch_size'],
        content_font=content_font, transform=transform, language=cfg['language'],
        val_loaders=val_loaders, meta=meta
    )
    if args.debug:
        evaluator.n_cv_batches = 10
        logger.info("Change CV batches to 10 for debugging")

    ############################
    # start training
    ############################
    trainer = Trainer(
        gen, disc, g_optim, d_optim, aux_clf, ac_optim,
        writer, logger, evaluator, cfg
    )
    trainer.train(loader, st_step)


if __name__ == "__main__":
    main()
