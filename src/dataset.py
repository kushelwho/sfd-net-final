"""
DehazingDataset — loads (hazy, clear, saliency) triplets.

Handles ITS (.png) and OTS (.jpg) naming conventions:
  ITS hazy: 1_1_0.90179.png  -> clear ID '1'  -> clear/1.png
  OTS hazy: 0025_0.8_0.04.jpg -> clear ID '0025' -> clear/0025.jpg

90/10 train/val split, OTS val capped at 1000 images.
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from sklearn.model_selection import train_test_split


OTS_MAX_VAL = 1_000

DOMAIN_CFG = {
    'its': {'hazy_subdir': 'hazy', 'clear_subdir': 'clear',
            'hazy_ext': '.png', 'clear_ext': '.png'},
    'ots': {'hazy_subdir': 'hazy', 'clear_subdir': 'clear',
            'hazy_ext': '.jpg', 'clear_ext': '.jpg'},
}


def get_clear_stem(hazy_filename):
    """'1_1_0.90179.png' -> '1', '0025_0.8_0.04.jpg' -> '0025'"""
    return os.path.splitext(hazy_filename)[0].split('_')[0]


def _shared_transform(split):
    extra = {'clear': 'image', 'saliency': 'image'}
    if split == 'train':
        return A.Compose([
            A.SmallestMaxSize(256),
            A.RandomCrop(256, 256),
            A.HorizontalFlip(p=0.5),
        ], additional_targets=extra)
    return A.Compose([A.Resize(480, 640)], additional_targets=extra)


def _hazy_augment():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    ])


class DehazingDataset(Dataset):

    def __init__(self, domain, split, data_root, saliency_root):
        assert domain in ('its', 'ots') and split in ('train', 'val')

        cfg = DOMAIN_CFG[domain]
        self.hazy_dir     = os.path.join(data_root, cfg['hazy_subdir'])
        self.clear_dir    = os.path.join(data_root, cfg['clear_subdir'])
        self.saliency_dir = saliency_root
        self.clear_ext    = cfg['clear_ext']

        all_files = sorted(f for f in os.listdir(self.hazy_dir)
                           if f.lower().endswith(cfg['hazy_ext']))
        if not all_files:
            raise RuntimeError(f"No hazy images in {self.hazy_dir}")

        train_f, val_f = train_test_split(all_files, test_size=0.1, random_state=42)
        if domain == 'ots' and len(val_f) > OTS_MAX_VAL:
            val_f = random.Random(42).sample(val_f, OTS_MAX_VAL)

        self.files = train_f if split == 'train' else val_f
        self.shared_tfm = _shared_transform(split)
        self.hazy_tfm   = _hazy_augment() if split == 'train' else None

        print(f"[DehazingDataset] {domain}/{split}: {len(self.files):,} images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hazy_fname = self.files[idx]
        stem       = get_clear_stem(hazy_fname)

        hazy  = cv2.cvtColor(cv2.imread(os.path.join(self.hazy_dir, hazy_fname)), cv2.COLOR_BGR2RGB)
        clear = cv2.cvtColor(cv2.imread(os.path.join(self.clear_dir, stem + self.clear_ext)), cv2.COLOR_BGR2RGB)

        sal_path = os.path.join(self.saliency_dir, stem + '.png')
        sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(sal_path) \
              else np.full(clear.shape[:2], 128, dtype=np.uint8)

        t = self.shared_tfm(image=hazy, clear=clear, saliency=sal)
        hazy, clear, sal = t['image'], t['clear'], t['saliency']

        if self.hazy_tfm:
            hazy = self.hazy_tfm(image=hazy)['image']

        to_t = lambda x: torch.from_numpy(x.astype(np.float32) / 255.0)
        return to_t(hazy).permute(2, 0, 1), to_t(clear).permute(2, 0, 1), to_t(sal).unsqueeze(0)
