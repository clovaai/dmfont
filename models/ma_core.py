"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch.nn as nn
from .comp_encoder import ComponentEncoder
from .decoder import Decoder
from .memory import Memory


class MACore(nn.Module):
    """ Memory-augmented HFG """
    def __init__(self, C_in, C, C_out, comp_enc, dec, n_comps, n_comp_types, language):
        """
        Args:
            C_in: 1
            C: unit of channel size
            C_out: 1

            comp_enc: component encoder configs
            dec: decoder configs

            n_comps: # of total component instances.
            n_comp_types: # of component types. kor=3, thai=4.
        """
        super().__init__()
        self.component_encoder = ComponentEncoder(
            C_in, C, **comp_enc, n_comp_types=n_comp_types
        )
        self.mem_shape = self.component_encoder.final_shape
        self.memory = Memory(self.mem_shape, n_comps, persistent=True, language=language)

        # setup skip memory
        if self.component_encoder.skip_layers is not None:
            # use dynamic memory only
            self.skip_memory = Memory(self.mem_shape, n_comps, persistent=False, language=language)

            skip_layers = self.component_encoder.skip_layers
            assert skip_layers is None or len(skip_layers) == 1, "Only supports #skip_layers <= 1"

        self.decoder = Decoder(
            C, C_out, self.mem_shape[-1], **dec, n_comp_types=n_comp_types
        )

    def reset_dynamic_memory(self):
        self.memory.reset_dynamic()
        if hasattr(self, 'skip_memory'):
            self.skip_memory.reset_dynamic()

    def encode_write(self, style_ids, comp_ids, style_imgs, reset_dynamic_memory=True):
        """ Encode feature from input data and write it to memory
        Args:
            # batch size B can be different with infer phase
            style_ids [B]: style index
            comp_ids [B, n_comp_types]: component ids of style chars
            style_imgs [B, 1, 128, 128]: eq_fonts
        """
        if reset_dynamic_memory:
            # reset dynamic memory before write
            self.reset_dynamic_memory()

        # encode & write component feature
        feats = self.component_encoder(style_imgs)  # [B, n_comp_types, C, H, W]
        comp_feats = feats[-1]
        skips = self.component_encoder.filter_skips(feats)  # filter skip features
        self.memory.write(style_ids, comp_ids, comp_feats)
        if hasattr(self, 'skip_memory'):
            self.skip_memory.write(style_ids, comp_ids, skips[0])

        return comp_feats

    def read_decode(self, target_style_ids, target_comp_ids):
        """ Read feature from memory and decode it
        Args:
            # batch size B can be different with write phase
            target_style_ids: [B]
            target_comp_ids: [B, n_comp_types]
        """
        # read memory w/ or w/o persistent memory
        # [B, n_comp_types, C, H, W]
        comp_feats = self.memory.read(target_style_ids, target_comp_ids)

        skips = None
        if hasattr(self, 'skip_memory'):
            skip_feats = self.skip_memory.read(target_style_ids, target_comp_ids)
            skips = [skip_feats]

        out = self.decoder(comp_feats, skips)

        return out
