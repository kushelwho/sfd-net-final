"""SFD-Net: full model assembly."""

import torch
import torch.nn as nn

from .model_parts import (
    StemBlock, FrequencyBlock, SpatialBlock, CrossModulation, FusionBlock,
)


class SFDNet(nn.Module):
    """
    Stem -> [Spatial | Frequency] -> CrossModulation -> Fusion -> clean image.

    forward(x)                        -> (B, 3, H, W)
    forward(x, return_intermediates)  -> (output, spatial_up, freq_up)
    """

    def __init__(self, stem_channels=32):
        super().__init__()
        C = stem_channels
        self.stem           = StemBlock()
        self.spatial_branch = SpatialBlock()
        self.freq_branch    = FrequencyBlock(C)
        self.cross_mod      = CrossModulation(C)
        self.fusion         = FusionBlock(C)

    def forward(self, x, return_intermediates=False):
        input_size = x.shape[2:]
        feat = self.stem(x)
        s = self.spatial_branch(feat)
        f = self.freq_branch(feat)
        s, f = self.cross_mod(s, f)
        out, s_up, f_up = self.fusion(s, f, output_size=input_size)

        if return_intermediates:
            return out, s_up, f_up
        return out
