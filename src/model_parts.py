"""
Building blocks for SFD-Net.

Pipeline: Input -> Stem -> [Spatial | Frequency] -> CrossModulation -> Fusion -> Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class StemBlock(nn.Module):
    """Downsample 2x before the dual-stream split. (B,3,H,W) -> (B,32,H/2,W/2)"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.stem(x)


class FrequencyBlock(nn.Module):
    """
    FFT-based haze suppressor. Learns a per-channel amplitude mask on the
    magnitude spectrum via a 1x1 conv (resolution-agnostic).
    """

    def __init__(self, channels):
        super().__init__()
        self.mag_scale = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x):
        fft = torch.fft.rfft2(x, norm='ortho')
        mag, phase = torch.abs(fft), torch.angle(fft)

        mag = mag * torch.sigmoid(self.mag_scale(mag))

        return torch.fft.irfft2(torch.polar(mag, phase), s=x.shape[-2:], norm='ortho')


class SpatialBlock(nn.Module):
    """3-level U-Net with a dilated bottleneck. Preserves fine spatial detail."""

    def __init__(self):
        super().__init__()
        self.enc1  = conv_bn_relu(32, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2  = conv_bn_relu(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_bn_relu(128, 128, dilation=2, padding=2)

        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = conv_bn_relu(256, 64)
        self.up1  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = conv_bn_relu(128, 32)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        bn = self.bottleneck(self.pool2(e2))

        d2 = self.dec2(torch.cat([self.up2(bn), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return d1


class CrossModulation(nn.Module):
    """
    Channel-attention alignment: each stream's attention re-weights the other.
    Keeps spatial and frequency branches semantically consistent before fusion.
    """

    def __init__(self, channels):
        super().__init__()
        self.attn_from_spatial = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels), nn.Sigmoid(),
        )
        self.attn_from_freq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels), nn.Sigmoid(),
        )

    def forward(self, spatial, freq):
        a_s = self.attn_from_spatial(spatial).unsqueeze(-1).unsqueeze(-1)
        a_f = self.attn_from_freq(freq).unsqueeze(-1).unsqueeze(-1)
        return spatial * a_f, freq * a_s


class FusionBlock(nn.Module):
    """
    Soft gate: upsample 2x, learn alpha, blend streams, project to RGB.
    Returns (output, spatial_up, freq_up) — last two are for visualization.
    """

    def __init__(self, channels=32):
        super().__init__()
        self.up        = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.gate_conv = nn.Conv2d(channels * 2, channels, 1, bias=True)
        self.out_conv  = nn.Conv2d(channels, 3, 1, bias=True)

    def forward(self, spatial, freq):
        s_up = self.up(spatial)
        f_up = self.up(freq)

        alpha = torch.sigmoid(self.gate_conv(torch.cat([s_up, f_up], 1)))
        fused = alpha * s_up + (1 - alpha) * f_up
        return torch.sigmoid(self.out_conv(fused)), s_up, f_up
