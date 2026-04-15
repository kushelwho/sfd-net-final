"""
Loss functions for SFD-Net.

- SaliencyCharbonnierLoss: pixel-wise Charbonnier weighted by saliency map
- PerceptualLoss: VGG16 relu2_2 feature L1 (frozen)
- CombinedLoss: charb + 0.04 * perceptual
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SaliencyCharbonnierLoss(nn.Module):
    """Charbonnier with saliency weighting. Foreground gets 6x penalty."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, saliency):
        per_pixel = torch.sqrt((pred - target) ** 2 + self.eps ** 2)
        weight = 1.0 + 5.0 * saliency
        return (per_pixel * weight).mean()


class PerceptualLoss(nn.Module):
    """L1 on VGG16 relu2_2 features. Weights are frozen."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features.children())[:10])
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        norm = lambda x: (x - self.mean) / self.std
        return torch.mean(torch.abs(
            self.features(norm(pred)) - self.features(norm(target.detach()))
        ))


class CombinedLoss(nn.Module):
    """Returns (total, charb, perc) as separate tensors for logging."""

    def __init__(self, charb_eps=1e-3, perc_weight=0.04):
        super().__init__()
        self.charb = SaliencyCharbonnierLoss(charb_eps)
        self.perc  = PerceptualLoss()
        self.perc_w = perc_weight

    def forward(self, pred, target, saliency):
        lc = self.charb(pred, target, saliency)
        lp = self.perc(pred, target)
        return lc + self.perc_w * lp, lc, lp
