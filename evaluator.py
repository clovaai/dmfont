"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from itertools import chain
from pathlib import Path
import json
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from sconf import Config

import utils
from logger import Logger

from models import MACore
from datasets import uniform_sample
from datasets import kor_decompose as kor
from datasets import thai_decompose as thai
from inference import (
    infer, get_val_loader,
    infer_2stage, get_val_encode_loader, get_val_decode_loader
)
from ssim import SSIM, MSSSIM


def torch_eval(val_fn):
    @torch.no_grad()
    def decorated(self, gen, *args, **kwargs):
        gen.eval()
        ret = val_fn(self, gen, *args, **kwargs)
        gen.train()

        return ret

    return decorated


class Evaluator:
    """DMFont evaluator.
    The evaluator provides pixel-level evaluation and glyphs generation
    from the reference style samples.
    """
    def __init__(self, data, trn_avails, logger, writer, batch_size, transform,
                 content_font, language, meta, val_loaders, n_workers=2):
        self.data = data
        self.logger = logger
        self.writer = writer
        self.batch_size = batch_size
        self.transform = transform
        self.n_workers = n_workers
        self.unify_resize_method = True

        self.trn_avails = trn_avails
        self.val_loaders = val_loaders
        self.content_font = content_font
        self.language = language
        if self.language == 'kor':
            self.n_comp_types = 3
        elif self.language == 'thai':
            self.n_comp_types = 4
        else:
            raise ValueError()

        # setup cross-validation
        self.SSIM = SSIM().cuda()
        weights = [0.25, 0.3, 0.3, 0.15]
        self.MSSSIM = MSSSIM(weights=weights).cuda()

        n_batches = [len(loader) for loader in self.val_loaders.values()]
        self.n_cv_batches = min(n_batches)
        self.logger.info("# of cross-validation batches = {}".format(self.n_cv_batches))

        # the number of chars/fonts for CV visualization
        n_chars = 16
        n_fonts = 16
        seen_chars = uniform_sample(meta['train']['chars'], n_chars//2)
        unseen_chars = uniform_sample(meta['valid']['chars'], n_chars//2)
        unseen_fonts = uniform_sample(meta['valid']['fonts'], n_fonts)

        self.cv_comparable_fonts = unseen_fonts
        self.cv_comparable_chars = seen_chars + unseen_chars

        allchars = meta['train']['chars'] + meta['valid']['chars']
        self.cv_comparable_avails = {
            font: allchars
            for font in self.cv_comparable_fonts
        }

    def validation(self, gen, step, extra_tag=''):
        self.comparable_validset_validation(gen, step, True, 'comparable_val'+extra_tag)

        plot_dic = {}
        for tag, loader in self.val_loaders.items():
            tag = tag + extra_tag
            l1, ssim, msssim = self.cross_validation(
                gen, step, loader, tag, n_batches=self.n_cv_batches
            )
            plot_dic[f'val/{tag}/l1'] = l1
            plot_dic[f'val/{tag}/ssim'] = ssim
            plot_dic[f'val/{tag}/ms-ssim'] = msssim if not np.isnan(msssim) else 0.
        self.writer.add_scalars(plot_dic, step)

        return plot_dic

    @torch_eval
    def comparable_validset_validation(self, gen, step, compare_inputs=False, tag='comparable_val'):
        """Comparable validation on validation set from CV"""
        comparable_grid = self.comparable_validation(
            gen, self.cv_comparable_avails, self.cv_comparable_fonts, self.cv_comparable_chars,
            n_max_match=1, compare_inputs=compare_inputs
        )

        self.writer.add_image(tag, comparable_grid, global_step=step)

    @torch_eval
    def comparable_validation(self, gen, style_avails, target_fonts, target_chars, n_max_match=3,
                              compare_inputs=False):
        """Compare horizontally for target fonts and chars"""
        # infer
        loader = get_val_loader(
            self.data, target_fonts, target_chars, style_avails,
            B=self.batch_size, n_max_match=n_max_match, transform=self.transform,
            content_font=self.content_font, language=self.language, n_workers=self.n_workers
        )
        out = infer(gen, loader)  # [B, 1, 128, 128]

        # ref original chars
        refs = self.get_charimages(target_fonts, target_chars)

        compare_batches = [refs, out]
        if compare_inputs:
            compare_batches += self.get_inputimages(loader)

        nrow = len(target_chars)
        comparable_grid = utils.make_comparable_grid(*compare_batches, nrow=nrow)

        return comparable_grid

    @torch_eval
    def cross_validation(self, gen, step, loader, tag, n_batches, n_log=64, save_dir=None):
        """Validation using splitted cross-validation set
        Args:
            n_log: # of images to log
            save_dir: if given, images are saved to save_dir
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        outs = []
        trgs = []
        n_accum = 0

        losses = utils.AverageMeters("l1", "ssim", "msssim")
        for i, (style_ids, style_comp_ids, style_imgs,
                trg_ids, trg_comp_ids, content_imgs, trg_imgs) in enumerate(loader):
            if i == n_batches:
                break

            style_ids = style_ids.cuda()
            style_comp_ids = style_comp_ids.cuda()
            style_imgs = style_imgs.cuda()
            trg_ids = trg_ids.cuda()
            trg_comp_ids = trg_comp_ids.cuda()
            trg_imgs = trg_imgs.cuda()

            gen.encode_write(style_ids, style_comp_ids, style_imgs)
            out = gen.read_decode(trg_ids, trg_comp_ids)
            B = len(out)

            # log images
            if n_accum < n_log:
                trgs.append(trg_imgs)
                outs.append(out)
                n_accum += B

                if n_accum >= n_log:
                    # log results
                    outs = torch.cat(outs)[:n_log]
                    trgs = torch.cat(trgs)[:n_log]
                    self.merge_and_log_image(tag, outs, trgs, step)

            l1, ssim, msssim = self.get_pixel_losses(out, trg_imgs, self.unify_resize_method)
            losses.updates({
                "l1": l1.item(),
                "ssim": ssim.item(),
                "msssim": msssim.item()
            }, B)

            # save images
            if save_dir:
                font_ids = trg_ids.detach().cpu().numpy()
                images = out.detach().cpu()  # [B, 1, 128, 128]
                char_comp_ids = trg_comp_ids.detach().cpu().numpy()  # [B, n_comp_types]
                for font_id, image, comp_ids in zip(font_ids, images, char_comp_ids):
                    font_name = loader.dataset.fonts[font_id]  # name.ttf
                    font_name = Path(font_name).stem  # remove ext
                    (save_dir / font_name).mkdir(parents=True, exist_ok=True)
                    if self.language == 'kor':
                        char = kor.compose(*comp_ids)
                    elif self.language == 'thai':
                        char = thai.compose_ids(*comp_ids)

                    uni = "".join([f'{ord(each):04X}' for each in char])
                    path = save_dir / font_name / "{}_{}.png".format(font_name, uni)
                    utils.save_tensor_to_image(image, path)

        self.logger.info(
            "  [Valid] {tag:30s} | Step {step:7d}  L1 {L.l1.avg:7.4f}  SSIM {L.ssim.avg:7.4f}"
            "  MSSSIM {L.msssim.avg:7.4f}"
            .format(tag=tag, step=step, L=losses))

        return losses.l1.avg, losses.ssim.avg, losses.msssim.avg

    def get_pixel_losses(self, out, trg_imgs, unify):
        """
        Args:
            out: generated images
            trg_imgs: target GT images
            unify: if True is given, unify glyph size and resize method before evaluation.
                This option give us the fair evaluation setting, which is used in the paper.
        """
        def unify_resize_method(img):
            # Unify various glyph size and resize method for fair evaluation
            size = img.size(-1)
            if size == 128:
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize([64, 64]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
                img = torch.stack([transform(_img) for _img in img.cpu()]).cuda()

            img = F.interpolate(img, scale_factor=2.0, mode='bicubic', align_corners=True)
            return img

        if unify:
            out = unify_resize_method(out)
            trg_imgs = unify_resize_method(trg_imgs)

        l1 = F.l1_loss(out, trg_imgs)
        ssim = self.SSIM(out, trg_imgs)
        msssim = self.MSSSIM(out, trg_imgs)

        return l1, ssim, msssim

    @torch_eval
    def handwritten_validation_2stage(self, gen, step, fonts, style_chars, target_chars,
                                      comparable=False, save_dir=None, tag='hw_validation_2stage'):
        """2-stage handwritten validation
        Args:
            fonts: [font_name1, font_name2, ...]
            save_dir: if given, do not write image grid, instead save every image into save_dir
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        outs = []
        for font_name in tqdm(fonts):
            encode_loader = get_val_encode_loader(
                self.data, font_name, style_chars, self.language, self.transform
            )
            decode_loader = get_val_decode_loader(target_chars, self.language)
            out = infer_2stage(gen, encode_loader, decode_loader)
            outs.append(out)

            if save_dir:
                for char, glyph in zip(target_chars, out):
                    uni = "".join([f'{ord(each):04X}' for each in char])
                    path = save_dir / font_name / "{}_{}.png".format(font_name, uni)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    utils.save_tensor_to_image(glyph, path)

        if save_dir:  # do not write grid
            return

        out = torch.cat(outs)
        if comparable:
            # ref original chars
            refs = self.get_charimages(fonts, target_chars)

            nrow = len(target_chars)
            grid = utils.make_comparable_grid(refs, out, nrow=nrow)
        else:
            grid = utils.to_grid(out, 'torch', nrow=len(target_chars))

        tag = tag + target_chars[:4]
        self.writer.add_image(tag, grid, global_step=step)

    def get_inputimages(self, val_loader):
        # integrate style images
        inputs = []
        for style_ids, style_comp_ids, style_imgs, trg_ids, trg_comp_ids, content_imgs \
                in val_loader:
            inputs.append(style_imgs)

        inputs = torch.cat(inputs)
        shape = inputs.shape
        inputs = inputs.view(shape[0]//self.n_comp_types, self.n_comp_types, *shape[1:])
        batches = [inputs[:, i] for i in range(self.n_comp_types)]

        return batches

    def get_charimages(self, fonts, chars, empty_header=False, as_tensor=True):
        """ get char images from self.data
        Return:
            2d list of charimages or 5d tensor:
            [
                [charimage1, charimage2, ...] (font1),
                ...
            ]
            or
            Tensor [n_fonts, n_chars, 1, 128, 128]
        """
        empty_box = torch.ones(1, 128, 128)
        charimages = [
            [self.data.get(font_name, char, empty_box) for char in chars]
            for font_name in fonts
        ]

        if empty_header:
            header = [empty_box for _ in chars]
            charimages.insert(0, header)

        if as_tensor:
            charimages = torch.stack(list(chain.from_iterable(charimages)))

        return charimages

    def merge_and_log_image(self, name, out, target, step):
        """ Merge out and target into 2-column grid and log it """
        merge = utils.make_merged_grid([out, target], merge_dim=2)
        self.writer.add_image(name, merge, global_step=step)


def eval_ckpt():
    from train import (
        setup_language_dependent, setup_data, setup_cv_dset_loader,
        get_dset_loader
    )

    logger = Logger.get()

    parser = argparse.ArgumentParser('MaHFG-eval')
    parser.add_argument(
        "name", help="name is used for directory name of the user-study generation results"
    )
    parser.add_argument("resume")
    parser.add_argument("img_dir")
    parser.add_argument("config_paths", nargs="+")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument(
        "--mode", default="eval",
        help="eval (default) / cv-save / user-study / user-study-save. "
             "`eval` generates comparable grid and computes pixel-level CV scores. "
             "`cv-save` generates and saves all target characters in CV. "
             "`user-study` generates comparable grid for the ramdomly sampled target characters. "
             "`user-study-save` generates and saves all target characters in user-study."
    )
    parser.add_argument("--deterministic", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    args, left_argv = parser.parse_known_args()

    cfg = Config(*args.config_paths)
    cfg.argv_update(left_argv)

    torch.backends.cudnn.benchmark = True

    cfg['data_dir'] = Path(cfg['data_dir'])

    if args.show:
        exit()

    # seed
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])

    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cfg['n_workers'] = 0
        logger.info("#" * 80)
        logger.info("# Deterministic option is activated !")
        logger.info("# Deterministic evaluator only ensure the deterministic cross-validation")
        logger.info("#" * 80)
    else:
        torch.backends.cudnn.benchmark = True

    if args.mode.startswith('mix'):
        assert cfg['g_args']['style_enc']['use'], \
                "Style mixing is only available with style encoder model"

    #####################################
    # Dataset
    ####################################
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

    val_loaders = setup_cv_dset_loader(
        hdf5_data, meta, transform, n_comp_types, content_font, cfg
    )

    #####################################
    # Model
    ####################################
    # setup generator only
    g_kwargs = cfg.get('g_args', {})
    gen = MACore(
        1, cfg['C'], 1, **g_kwargs, n_comps=n_comps, n_comp_types=n_comp_types,
        language=cfg['language']
    )
    gen.cuda()

    ckpt = torch.load(args.resume)
    logger.info("Use EMA generator as default")
    gen.load_state_dict(ckpt['generator_ema'])

    step = ckpt['epoch']
    loss = ckpt['loss']

    logger.info("Resumed checkpoint from {} (Step {}, Loss {:7.3f})".format(
        args.resume, step, loss))

    writer = utils.DiskWriter(args.img_dir, 0.6)

    evaluator = Evaluator(
        hdf5_data, trn_dset.avails, logger, writer, cfg['batch_size'],
        content_font=content_font, transform=transform, language=cfg['language'],
        val_loaders=val_loaders, meta=meta
    )
    evaluator.n_cv_batches = -1
    logger.info("Update n_cv_batches = -1 to evaluate about full data")
    if args.debug:
        evaluator.n_cv_batches = 10
        logger.info("!!! DEBUG MODE: n_cv_batches = 10 !!!")

    if args.mode == 'eval':
        logger.info("Start validation ...")
        dic = evaluator.validation(gen, step)
        logger.info("Validation is done. Result images are saved to {}".format(args.img_dir))
    elif args.mode.startswith('user-study'):
        meta = json.load(open('meta/kor-unrefined.json'))
        target_chars = meta['target_chars']
        style_chars = meta['style_chars']
        fonts = meta['fonts']

        if args.mode == 'user-study':
            sampled_target_chars = uniform_sample(target_chars, 20)
            logger.info("Start generation kor-unrefined ...")
            logger.info("Sampled chars = {}".format(sampled_target_chars))

            evaluator.handwritten_validation_2stage(
                gen, step, fonts, style_chars, sampled_target_chars,
                comparable=True, tag='userstudy-{}'.format(args.name)
            )
        elif args.mode == 'user-study-save':
            logger.info("Start generation & saving kor-unrefined ...")
            save_dir = Path(args.img_dir) / "{}-{}".format(args.name, step)
            evaluator.handwritten_validation_2stage(
                gen, step, fonts, style_chars, target_chars,
                comparable=True, save_dir=save_dir
            )
        logger.info("Validation is done. Result images are saved to {}".format(args.img_dir))
    elif args.mode == 'cv-save':
        save_dir = Path(args.img_dir) / "cv_images_{}".format(step)
        logger.info("Save CV results to {} ...".format(save_dir))
        utils.rm(save_dir)
        for tag, loader in val_loaders.items():
            l1, ssim, msssim = evaluator.cross_validation(
                gen, step, loader, tag, n_batches=evaluator.n_cv_batches, save_dir=(save_dir / tag)
            )
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    eval_ckpt()
