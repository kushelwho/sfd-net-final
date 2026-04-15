"""Metrics, logging, and image conversion helpers."""

import os
import csv

import cv2
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def tensor_to_numpy(t):
    """(C,H,W) float [0,1] -> (H,W,C) uint8."""
    img = t.detach().cpu().clamp(0, 1).numpy()
    img = (img * 255).astype(np.uint8)
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    return img


def compute_psnr(pred, target):
    if pred.ndim == 3:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    return float(np.mean([
        _psnr(tensor_to_numpy(t), tensor_to_numpy(p), data_range=255)
        for p, t in zip(pred, target)
    ]))


def compute_ssim(pred, target):
    if pred.ndim == 3:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    return float(np.mean([
        _ssim(tensor_to_numpy(t), tensor_to_numpy(p), channel_axis=2, data_range=255)
        for p, t in zip(pred, target)
    ]))


def save_image(tensor, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(tensor_to_numpy(tensor), cv2.COLOR_RGB2BGR))


class MetricsLogger:
    HEADER = ['epoch', 'train_loss', 'val_psnr', 'val_ssim']

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(self.HEADER)

    def log(self, epoch, train_loss, val_psnr=0.0, val_ssim=0.0):
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, f'{train_loss:.6f}', f'{val_psnr:.4f}', f'{val_ssim:.4f}',
            ])
