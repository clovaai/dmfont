"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from datasets import cyclize
from models.memory import comp_id_to_addr
from criterions import hinge_g_loss, hinge_d_loss


def has_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True

    return False


def load_checkpoint(path, gen, disc, aux_clf, g_optim, d_optim, ac_optim):
    ckpt = torch.load(path)
    gen.load_state_dict(ckpt['generator'])
    g_optim.load_state_dict(ckpt['optimizer'])

    if disc is not None:
        disc.load_state_dict(ckpt['discriminator'])
        d_optim.load_state_dict(ckpt['d_optimizer'])

    if aux_clf is not None:
        aux_clf.load_state_dict(ckpt['aux_clf'])
        ac_optim.load_state_dict(ckpt['ac_optimizer'])

    # NOTE epoch is step
    st_epoch = ckpt['epoch'] + 1
    loss = ckpt['loss']

    return st_epoch, loss


class Trainer:
    def __init__(self, gen, disc, g_optim, d_optim, aux_clf, ac_optim,
                 writer, logger, evaluator, cfg):
        self.gen = gen
        self.gen_ema = copy.deepcopy(self.gen)
        self.is_bn_gen = has_bn(self.gen)
        self.disc = disc
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.aux_clf = aux_clf
        self.ac_optim = ac_optim
        self.writer = writer
        self.logger = logger
        self.evaluator = evaluator
        self.cfg = cfg
        self.step = 1
        self.language = cfg['language']

        self.g_losses = {}
        self.d_losses = {}
        self.ac_losses = {}

    def clear_losses(self):
        """ Integrate & clear loss dict """
        # g losses
        loss_dic = {k: v.item() for k, v in self.g_losses.items()}
        loss_dic['g_total'] = sum(loss_dic.values())
        # d losses
        loss_dic.update({k: v.item() for k, v in self.d_losses.items()})
        # ac losses
        loss_dic.update({k: v.item() for k, v in self.ac_losses.items()})

        self.g_losses = {}
        self.d_losses = {}
        self.ac_losses = {}

        return loss_dic

    def accum_g(self, decay=0.999):
        par1 = dict(self.gen_ema.named_parameters())
        par2 = dict(self.gen.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

    def sync_g_ema(self, style_ids, style_comp_ids, style_imgs, trg_ids, trg_comp_ids):
        """ update running stats for BN & update max singular value for SN """
        org_train_mode = self.gen_ema.training
        with torch.no_grad():
            self.gen_ema.train()
            self.gen_ema.encode_write(style_ids, style_comp_ids, style_imgs)
            self.gen_ema.read_decode(trg_ids, trg_comp_ids)
        self.gen_ema.train(org_train_mode)

    def train(self, loader, st_step=1, val=None):
        val = val or {}
        self.gen.train()
        self.disc.train()

        # loss stats
        losses = utils.AverageMeters("g_total", "pixel", "disc", "gen", "fm", "ac", "ac_gen")
        # discriminator stats
        discs = utils.AverageMeters("real", "fake",
                                    "real_font", "real_char", "fake_font", "fake_char",
                                    "real_acc", "fake_acc", "real_font_acc", "real_char_acc",
                                    "fake_font_acc", "fake_char_acc")
        # etc stats
        stats = utils.AverageMeters("B_style", "B_target", "ac_acc", "ac_gen_acc")

        self.step = st_step
        self.clear_losses()

        self.logger.info("Start training ...")
        for (style_ids, style_char_ids, style_comp_ids, style_imgs,
             trg_ids, trg_char_ids, trg_comp_ids, trg_imgs, *content_imgs) in cyclize(loader):
            B = trg_imgs.size(0)
            stats.updates({
                "B_style": style_imgs.size(0),
                "B_target": B
            })

            style_ids = style_ids.cuda()
            #  style_char_ids = style_char_ids.cuda()
            style_comp_ids = style_comp_ids.cuda()
            style_imgs = style_imgs.cuda()
            trg_ids = trg_ids.cuda()
            trg_char_ids = trg_char_ids.cuda()
            trg_comp_ids = trg_comp_ids.cuda()
            trg_imgs = trg_imgs.cuda()

            # infer
            comp_feats = self.gen.encode_write(style_ids, style_comp_ids, style_imgs)
            out = self.gen.read_decode(trg_ids, trg_comp_ids)

            # D loss
            real, real_font, real_char, real_feats = self.disc(
                trg_imgs, trg_ids, trg_char_ids, out_feats=True
            )
            fake, fake_font, fake_char = self.disc(out.detach(), trg_ids, trg_char_ids)
            self.add_gan_d_loss(real, real_font, real_char, fake, fake_font, fake_char)

            self.d_optim.zero_grad()
            self.d_backward()
            self.d_optim.step()

            # G loss
            fake, fake_font, fake_char, fake_feats = self.disc(
                out, trg_ids, trg_char_ids, out_feats=True
            )
            self.add_gan_g_loss(real, real_font, real_char, fake, fake_font, fake_char)

            # feature matching loss
            self.add_fm_loss(real_feats, fake_feats)

            # disc stats
            racc = lambda x: (x > 0.).float().mean().item()
            facc = lambda x: (x < 0.).float().mean().item()
            discs.updates({
                "real": real.mean().item(),
                "fake": fake.mean().item(),
                "real_font": real_font.mean().item(),
                "real_char": real_char.mean().item(),
                "fake_font": fake_font.mean().item(),
                "fake_char": fake_char.mean().item(),

                'real_acc': racc(real),
                'fake_acc': facc(fake),
                'real_font_acc': racc(real_font),
                'real_char_acc': racc(real_char),
                'fake_font_acc': facc(fake_font),
                'fake_char_acc': facc(fake_char)
            }, B)

            # pixel loss
            self.add_pixel_loss(out, trg_imgs)

            self.g_optim.zero_grad()
            # NOTE ac loss generates & leaves grads to G.
            # so g_optim.zero_grad() should place in front of ac loss and
            # g_backward() should follow ac loss.
            if self.aux_clf is not None:
                self.add_ac_losses_and_update_stats(
                    comp_feats, style_comp_ids, out, trg_comp_ids, stats
                )

                self.ac_optim.zero_grad()
                self.ac_backward(retain_graph=True)
                self.ac_optim.step()

            self.g_backward()
            self.g_optim.step()

            loss_dic = self.clear_losses()
            losses.updates(loss_dic, B)

            # generator EMA
            self.accum_g()
            if self.is_bn_gen:
                self.sync_g_ema(style_ids, style_comp_ids, style_imgs, trg_ids, trg_comp_ids)

            # after step
            if self.step % self.cfg['tb_freq'] == 0:
                self.plot(losses, discs, stats)

            if self.step % self.cfg['print_freq'] == 0:
                self.log(losses, discs, stats)
                losses.resets()
                discs.resets()
                stats.resets()

            if self.step % self.cfg['val_freq'] == 0:
                epoch = self.step / len(loader)
                self.logger.info("Validation at Epoch = {:.3f}".format(epoch))
                self.evaluator.merge_and_log_image('d1', out, trg_imgs, self.step)
                self.evaluator.validation(self.gen, self.step)

                # if non-BN generator, sync max singular value of spectral norm.
                if not self.is_bn_gen:
                    self.sync_g_ema(style_ids, style_comp_ids, style_imgs, trg_ids, trg_comp_ids)
                self.evaluator.validation(self.gen_ema, self.step, extra_tag='_EMA')

                # save freq == val freq
                self.save(
                    loss_dic['g_total'], self.cfg['save'],
                    self.cfg.get('save_freq', self.cfg['val_freq'])
                )

            if self.step >= self.cfg['max_iter']:
                self.logger.info("Iteration finished.")
                break

            self.step += 1

    def add_pixel_loss(self, out, target):
        loss = F.l1_loss(out, target, reduction='mean') * self.cfg['pixel_w']
        self.g_losses['pixel'] = loss

        return loss

    def add_gan_g_loss(self, real, real_font, real_char, fake, fake_font, fake_char):
        if self.cfg['gan_w'] == 0.:
            return 0.

        g_loss = hinge_g_loss(real_font.detach(), fake_font) + \
                 hinge_g_loss(real_char.detach(), fake_char)
        if self.disc.use_rx:
            g_loss += hinge_g_loss(real.detach(), fake)
        g_loss *= self.cfg['gan_w']

        self.g_losses['gen'] = g_loss

        return g_loss

    def add_gan_d_loss(self, real, real_font, real_char, fake, fake_font, fake_char):
        if self.cfg['gan_w'] == 0.:
            return 0.

        d_loss = hinge_d_loss(real_font, fake_font) + \
                 hinge_d_loss(real_char, fake_char)
        if self.disc.use_rx:
            d_loss += hinge_d_loss(real, fake)
        d_loss *= self.cfg['gan_w']

        self.d_losses['disc'] = d_loss

        return d_loss

    def add_fm_loss(self, real_feats, fake_feats):
        if self.cfg['fm_w'] == 0.:
            return 0.

        fm_loss = 0.
        for real_f, fake_f in zip(real_feats, fake_feats):
            fm_loss += F.l1_loss(real_f.detach(), fake_f)
        fm_loss = fm_loss / len(real_feats) * self.cfg['fm_w']

        self.g_losses['fm'] = fm_loss

        return fm_loss

    def add_ac_losses_and_update_stats(self, comp_feats, style_comp_ids, generated,
                                       trg_comp_ids, stats):
        # 1. ac(enc(x)) loss
        loss, acc = self.infer_ac(comp_feats, style_comp_ids)
        self.ac_losses['ac'] = loss * self.cfg['ac_w']
        stats.ac_acc.update(acc, style_comp_ids.numel())

        # 2. ac(enc(fake)) loss
        # Freeze second encoder to prevent cheating by encoder
        with utils.temporary_freeze(self.gen.component_encoder):
            feats = self.gen.component_encoder(generated)

        gen_comp_feats = feats[-1]

        loss, acc = self.infer_ac(gen_comp_feats, trg_comp_ids)
        self.ac_losses['ac_gen'] = loss * self.cfg['ac_w']
        stats.ac_gen_acc.update(acc, trg_comp_ids.numel())

    def infer_ac(self, comp_feats, comp_ids):
        """ Auxiliary classifier loss on style or output features """
        comp_addrs = comp_id_to_addr(comp_ids, self.language)

        comp_feats_flat = comp_feats.flatten(0, 1)
        comp_addrs_flat = comp_addrs.flatten(0, 1)

        aux_out = self.aux_clf(comp_feats_flat)
        loss = F.cross_entropy(aux_out, comp_addrs_flat)

        acc = utils.accuracy(aux_out, comp_addrs_flat)

        return loss, acc

    def d_backward(self):
        with utils.temporary_freeze(self.gen):
            d_loss = sum(self.d_losses.values())
            d_loss.backward()

    def g_backward(self):
        with utils.temporary_freeze(self.disc):
            g_loss = sum(self.g_losses.values())
            g_loss.backward()

    def ac_backward(self, retain_graph):
        if self.aux_clf is None:
            return

        org_grads = utils.freeze(self.gen.memory.persistent_memory)

        if 'ac' in self.ac_losses:
            self.ac_losses['ac'].backward(retain_graph=retain_graph)

        if 'ac_gen' in self.ac_losses:
            with utils.temporary_freeze(self.aux_clf):
                self.ac_losses['ac_gen'].backward(retain_graph=retain_graph)

        utils.unfreeze(self.gen.memory.persistent_memory, org_grads)

    def save(self, cur_loss, method, save_freq=None):
        """
        Args:
            method: all / last
                all: save checkpoint by step
                last: save checkpoint to 'last.pth'
                all-last: save checkpoint by step per save_freq and
                          save checkpoint to 'last.pth' always
        """
        if method not in ['all', 'last', 'all-last']:
            return

        step_save = False
        last_save = False
        if method == 'all' or (method == 'all-last' and self.step % save_freq == 0):
            step_save = True
        if method in ('last', 'all-last'):
            last_save = True
        assert step_save or last_save

        save_dic = {
            'generator': self.gen.state_dict(),
            'generator_ema': self.gen_ema.state_dict(),
            'discriminator': self.disc.state_dict(),
            'd_optimizer': self.d_optim.state_dict(),
            'optimizer': self.g_optim.state_dict(),
            'epoch': self.step,
            'loss': cur_loss
        }

        if self.aux_clf is not None:
            save_dic['aux_clf'] = self.aux_clf.state_dict()
            save_dic['ac_optimizer'] = self.ac_optim.state_dict()

        ckpt_dir = Path("checkpoints", self.cfg['unique_name'])
        step_ckpt_name = "{:06d}-{}.pth".format(self.step, self.cfg['name'])
        last_ckpt_name = "last.pth"
        step_ckpt_path = ckpt_dir / step_ckpt_name
        last_ckpt_path = ckpt_dir / last_ckpt_name

        log = ""
        if step_save:
            torch.save(save_dic, str(step_ckpt_path))
            log = "Checkpoint is saved to {}".format(step_ckpt_path)

            if last_save:
                utils.rm(last_ckpt_path)
                last_ckpt_path.symlink_to(step_ckpt_path.absolute())
                log += " w/ {} symlink".format(last_ckpt_name)

        if not step_save and last_save:
            utils.rm(last_ckpt_path)
            torch.save(save_dic, str(last_ckpt_path))
            log = "Checkpoint is saved to {}".format(last_ckpt_path)

        self.logger.info("{}\n".format(log))

    def plot(self, losses, discs, stats):
        tag_scalar_dic = {
            'train/g_total_loss': losses.g_total.val,
            'train/pixel_loss': losses.pixel.val,

            'train/d_loss': losses.disc.val,
            'train/g_loss': losses.gen.val,
            'train/d_real_font': discs.real_font.val,
            'train/d_real_char': discs.real_char.val,
            'train/d_fake_font': discs.fake_font.val,
            'train/d_fake_char': discs.fake_char.val,

            'train/d_real_font_acc': discs.real_font_acc.val,
            'train/d_real_char_acc': discs.real_char_acc.val,
            'train/d_fake_font_acc': discs.fake_font_acc.val,
            'train/d_fake_char_acc': discs.fake_char_acc.val
        }

        if self.disc.use_rx:
            tag_scalar_dic.update({
                'train/d_real': discs.real.val,
                'train/d_fake': discs.fake.val,
                'train/d_real_acc': discs.real_acc.val,
                'train/d_fake_acc': discs.fake_acc.val
            })
        if self.cfg['fm_w'] > 0.:
            tag_scalar_dic['train/feature_matching'] = losses.fm.val

        if self.aux_clf is not None:
            tag_scalar_dic.update({
                'train/ac_loss': losses.ac.val,
                'train/ac_acc': stats.ac_acc.val,
                'train/ac_gen_loss': losses.ac_gen.val,
                'train/ac_gen_acc': stats.ac_gen_acc.val
            })

        self.writer.add_scalars(tag_scalar_dic, self.step)

    def log(self, losses, discs, stats):
        self.logger.info(
            "  Step {step:7d}: L1 {L.pixel.avg:7.4f}  D {L.disc.avg:7.3f}  G {L.gen.avg:7.3f}"
            "  FM {L.fm.avg:7.3f}  AC {S.ac_acc.avg:5.1%}  AC_gen {S.ac_gen_acc.avg:5.1%}"
            "  R {D.real_acc.avg:7.3f}  F {D.fake_acc.avg:7.3f}"
            "  R_font {D.real_font_acc.avg:7.3f}  F_font {D.fake_font_acc.avg:7.3f}"
            "  R_char {D.real_char_acc.avg:7.3f}  F_char {D.fake_char_acc.avg:7.3f}"
            "  B_stl {S.B_style.avg:5.1f}  B_trg {S.B_target.avg:5.1f}"
            .format(step=self.step, L=losses, D=discs, S=stats))
