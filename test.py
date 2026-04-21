#!/usr/bin/env python3
"""
Test SFD-Net on the SOTS (Synthetic Objective Testing Set) benchmark.

SOTS is the standard test set from the RESIDE dataset. It has two subsets:
  - indoor  (500 images) — used to evaluate models trained on ITS
  - outdoor (500 images) — used to evaluate models trained on OTS

Expected SOTS directory layout:
    SOTS/indoor/hazy/       hazy images  (e.g. 1_1_0.90179.png)
    SOTS/indoor/clear/      ground truth (e.g. 1.png)
    SOTS/outdoor/hazy/      hazy images  (e.g. 0025_0.8_0.04.jpg)
    SOTS/outdoor/clear/     ground truth (e.g. 0025.jpg)

Usage examples:
    # Test ITS-trained model on SOTS indoor
    python test.py \\
        --checkpoint output/outputs_its/best_model.pth \\
        --data-root  data/SOTS/indoor \\
        --output-dir output/test_sots_indoor

    # Test OTS-trained model on SOTS outdoor
    python test.py \\
        --checkpoint output/outputs_ots/best_model.pth \\
        --data-root  data/SOTS/outdoor \\
        --output-dir output/test_sots_outdoor

    # Save dehazed images alongside metrics
    python test.py \\
        --checkpoint output/outputs_its/best_model.pth \\
        --data-root  data/SOTS/indoor \\
        --output-dir output/test_sots_indoor \\
        --save-images
"""

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.sfd_net import SFDNet
from src.utils import (
    AverageMeter, compute_psnr, compute_ssim, tensor_to_numpy, save_image,
)


# ── SOTS dataset ─────────────────────────────────────────────────────────

def _get_clear_stem(hazy_filename):
    """Extract ground-truth ID from a hazy filename.

    Works for both ITS and OTS naming:
        '1_1_0.90179.png'  → '1'
        '0025_0.8_0.04.jpg' → '0025'
    """
    return os.path.splitext(hazy_filename)[0].split('_')[0]


def _find_gt(gt_dir, stem):
    """Find the ground-truth file for a given stem, trying common extensions."""
    for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
        path = os.path.join(gt_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def _load_image(path):
    """Read an image as float32 RGB tensor (C, H, W) in [0, 1]."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def _pad_to_even(tensor):
    """Pad (C, H, W) so both H and W are even (required by stride-2 stem)."""
    _, h, w = tensor.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h),
                                         mode='reflect')
    return tensor, h, w


# ── model loading ────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    """Load SFD-Net from a training checkpoint."""
    model = SFDNet()
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    sd = ckpt['model_state']
    if any(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)

    model = model.to(device).eval()
    epoch = ckpt.get('epoch', '?')
    psnr  = ckpt.get('best_psnr', 0.0)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Trained for {epoch} epoch(s), best val PSNR: {psnr:.2f} dB")
    return model


# ── testing ──────────────────────────────────────────────────────────────

def test(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    hazy_dir = os.path.join(args.data_root, args.hazy_subdir)
    gt_dir   = os.path.join(args.data_root, args.gt_subdir)

    if not os.path.isdir(hazy_dir):
        raise FileNotFoundError(f"Hazy directory not found: {hazy_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    hazy_files = sorted(f for f in os.listdir(hazy_dir)
                        if os.path.splitext(f)[1].lower() in exts)
    if not hazy_files:
        raise RuntimeError(f"No images found in {hazy_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.checkpoint, device)

    print(f"\n{'='*55}")
    print(f"  Test set   : {args.data_root}")
    print(f"  Images     : {len(hazy_files)}")
    print(f"  Device     : {device}")
    print(f"  Save images: {args.save_images}")
    print(f"  Output     : {args.output_dir}")
    print(f"{'='*55}\n")

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    per_image_results = []
    times = []

    with torch.no_grad():
        for fname in tqdm(hazy_files, desc='Testing', dynamic_ncols=True):
            stem = _get_clear_stem(fname)

            gt_path = _find_gt(gt_dir, stem)
            if gt_path is None:
                tqdm.write(f"  WARN: no GT for {fname} (stem={stem}), skipping")
                continue

            hazy_tensor = _load_image(os.path.join(hazy_dir, fname))
            gt_tensor   = _load_image(gt_path)

            padded, orig_h, orig_w = _pad_to_even(hazy_tensor)

            t0 = time.time()
            pred = model(padded.unsqueeze(0).to(device)).squeeze(0)
            elapsed = time.time() - t0
            times.append(elapsed)

            pred = pred[:, :orig_h, :orig_w].cpu()
            gt_tensor = gt_tensor[:, :orig_h, :orig_w]

            p = compute_psnr(pred, gt_tensor)
            s = compute_ssim(pred, gt_tensor)
            psnr_meter.update(p)
            ssim_meter.update(s)

            per_image_results.append({
                'filename': fname, 'psnr': p, 'ssim': s, 'time_s': elapsed,
            })

            if args.save_images:
                save_image(pred, os.path.join(args.output_dir, 'dehazed', fname))

    if not per_image_results:
        print("No images were tested. Check dataset paths.")
        return 0.0, 0.0

    # ── Save per-image CSV ────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'per_image_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['filename', 'psnr', 'ssim', 'time_s'])
        w.writeheader()
        for row in per_image_results:
            w.writerow({
                'filename': row['filename'],
                'psnr': f"{row['psnr']:.4f}",
                'ssim': f"{row['ssim']:.4f}",
                'time_s': f"{row['time_s']:.4f}",
            })

    psnr_vals = [r['psnr'] for r in per_image_results]
    ssim_vals = [r['ssim'] for r in per_image_results]
    n = len(per_image_results)

    _print_comprehensive_report(
        args, psnr_vals, ssim_vals, times, per_image_results, n,
    )

    return psnr_meter.avg, ssim_meter.avg


# ── comprehensive report ─────────────────────────────────────────────────

def _percentile(vals, p):
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (k - f) * (s[c] - s[f])


def _grade(psnr):
    if psnr >= 35: return 'A+'
    if psnr >= 30: return 'A'
    if psnr >= 27: return 'B+'
    if psnr >= 25: return 'B'
    if psnr >= 22: return 'C'
    if psnr >= 20: return 'D'
    return 'F'


def _print_comprehensive_report(args, psnr_vals, ssim_vals, times,
                                per_image_results, n):
    out = args.output_dir
    lines = []

    def p(text=''):
        lines.append(text)
        print(text)

    # ── 1. Basic summary ─────────────────────────────────────────────
    p(f"\n{'='*65}")
    p(f"  SFD-Net — SOTS Test Results")
    p(f"{'='*65}")
    p(f"  Test Set     : {args.data_root}")
    p(f"  Checkpoint   : {args.checkpoint}")
    p(f"  Images tested: {n}")
    p(f"  PSNR (avg)   : {np.mean(psnr_vals):.3f} dB")
    p(f"  SSIM (avg)   : {np.mean(ssim_vals):.4f}")
    p(f"  Avg time/img : {np.mean(times):.4f} s")
    p(f"  Output       : {out}")
    p(f"{'='*65}")

    # ── 2. Detailed statistics ───────────────────────────────────────
    p(f"\n{'='*65}")
    p(f"  DETAILED STATISTICAL SUMMARY")
    p(f"{'='*65}")
    p(f"  PSNR (dB):")
    p(f"    Mean      : {np.mean(psnr_vals):.3f}")
    p(f"    Std       : {np.std(psnr_vals):.3f}")
    p(f"    Min       : {np.min(psnr_vals):.3f}")
    p(f"    Max       : {np.max(psnr_vals):.3f}")
    p(f"    Median    : {np.median(psnr_vals):.3f}")
    p(f"    25th %ile : {_percentile(psnr_vals, 25):.3f}")
    p(f"    75th %ile : {_percentile(psnr_vals, 75):.3f}")
    p(f"    90th %ile : {_percentile(psnr_vals, 90):.3f}")
    p()
    p(f"  SSIM:")
    p(f"    Mean      : {np.mean(ssim_vals):.4f}")
    p(f"    Std       : {np.std(ssim_vals):.4f}")
    p(f"    Min       : {np.min(ssim_vals):.4f}")
    p(f"    Max       : {np.max(ssim_vals):.4f}")
    p(f"    Median    : {np.median(ssim_vals):.4f}")
    p(f"    25th %ile : {_percentile(ssim_vals, 25):.4f}")
    p(f"    75th %ile : {_percentile(ssim_vals, 75):.4f}")
    p(f"    90th %ile : {_percentile(ssim_vals, 90):.4f}")
    p()
    p(f"  Inference Time:")
    p(f"    Mean      : {np.mean(times):.4f} s")
    p(f"    Median    : {np.median(times):.4f} s")
    p(f"    Total     : {np.sum(times):.2f} s")
    p(f"    FPS       : {1.0 / np.mean(times):.1f}")
    p(f"{'='*65}")

    # ── 3. Quality classification matrix ─────────────────────────────
    psnr_bins = [(0, 20, 'Poor (<20 dB)'),
                 (20, 25, 'Fair (20-25 dB)'),
                 (25, 30, 'Good (25-30 dB)'),
                 (30, 35, 'Very Good (30-35 dB)'),
                 (35, 999, 'Excellent (>35 dB)')]
    ssim_bins = [(0, 0.80, 'Poor (<0.80)'),
                 (0.80, 0.90, 'Fair (0.80-0.90)'),
                 (0.90, 0.95, 'Good (0.90-0.95)'),
                 (0.95, 0.98, 'Very Good (0.95-0.98)'),
                 (0.98, 1.01, 'Excellent (>0.98)')]

    p(f"\n{'='*65}")
    p(f"  QUALITY CLASSIFICATION MATRIX  (PSNR rows x SSIM columns)")
    p(f"{'='*65}")

    header = f"  {'PSNR \\ SSIM':<22s}"
    for _, _, sl in ssim_bins:
        short = sl.split('(')[1].rstrip(')')
        header += f" {short:>12s}"
    header += f" {'Total':>7s}"
    p(header)
    p(f"  {'-'*len(header)}")

    for plo, phi, plabel in psnr_bins:
        row_str = f"  {plabel:<22s}"
        row_total = 0
        for slo, shi, _ in ssim_bins:
            count = sum(1 for r in per_image_results
                        if plo <= r['psnr'] < phi and slo <= r['ssim'] < shi)
            row_total += count
            cell = str(count) if count > 0 else '.'
            row_str += f" {cell:>12s}"
        row_str += f" {row_total:>7d}"
        p(row_str)

    p(f"  {'-'*len(header)}")
    total_row = f"  {'Total':<22s}"
    for slo, shi, _ in ssim_bins:
        col_total = sum(1 for r in per_image_results if slo <= r['ssim'] < shi)
        total_row += f" {col_total:>12d}"
    total_row += f" {n:>7d}"
    p(total_row)
    p(f"{'='*65}")

    # ── 4. PSNR quality breakdown ────────────────────────────────────
    p(f"\n{'='*65}")
    p(f"  QUALITY BREAKDOWN")
    p(f"{'='*65}")
    for lo, hi, label in psnr_bins:
        count = sum(1 for v in psnr_vals if lo <= v < hi)
        pct = count / n * 100
        bar = '#' * int(pct / 2)
        p(f"  {label:<24s}: {count:>4d}/{n}  ({pct:5.1f}%)  {bar}")

    p()
    psnr_ge25 = sum(1 for v in psnr_vals if v >= 25)
    psnr_ge30 = sum(1 for v in psnr_vals if v >= 30)
    ssim_ge90 = sum(1 for v in ssim_vals if v >= 0.90)
    ssim_ge95 = sum(1 for v in ssim_vals if v >= 0.95)
    p(f"  Pass rates:")
    p(f"    PSNR >= 25 dB : {psnr_ge25:>4d}/{n}  ({psnr_ge25/n*100:5.1f}%)")
    p(f"    PSNR >= 30 dB : {psnr_ge30:>4d}/{n}  ({psnr_ge30/n*100:5.1f}%)")
    p(f"    SSIM >= 0.90  : {ssim_ge90:>4d}/{n}  ({ssim_ge90/n*100:5.1f}%)")
    p(f"    SSIM >= 0.95  : {ssim_ge95:>4d}/{n}  ({ssim_ge95/n*100:5.1f}%)")
    p(f"{'='*65}")

    # ── 5. Top 10 best & worst ───────────────────────────────────────
    sorted_by_psnr = sorted(per_image_results, key=lambda r: r['psnr'])
    k = min(10, n)

    p(f"\n{'='*65}")
    p(f"  TOP {k} BEST IMAGES (by PSNR)")
    p(f"{'='*65}")
    p(f"  {'#':>4s}  {'Filename':<35s} {'PSNR':>9s} {'SSIM':>8s}")
    p(f"  {'-'*60}")
    for i, r in enumerate(reversed(sorted_by_psnr[-k:]), 1):
        p(f"  {i:>4d}  {r['filename']:<35s} {r['psnr']:>8.3f}  {r['ssim']:>7.4f}")

    p(f"\n{'='*65}")
    p(f"  TOP {k} WORST IMAGES (by PSNR)")
    p(f"{'='*65}")
    p(f"  {'#':>4s}  {'Filename':<35s} {'PSNR':>9s} {'SSIM':>8s}")
    p(f"  {'-'*60}")
    for i, r in enumerate(sorted_by_psnr[:k], 1):
        p(f"  {i:>4d}  {r['filename']:<35s} {r['psnr']:>8.3f}  {r['ssim']:>7.4f}")

    # ── 6. Comparison table ──────────────────────────────────────────
    our_psnr = np.mean(psnr_vals)
    our_ssim = np.mean(ssim_vals)
    literature = [
        ('DCP (He et al.)',    2009, '-',     16.62, 0.8179),
        ('AOD-Net',            2017, '~2K',   19.06, 0.8504),
        ('DehazeNet',          2016, '~8K',   21.14, 0.8472),
        ('GFN',                2018, '~500K', 22.30, 0.8800),
        ('FFA-Net',            2020, '~4.5M', 36.39, 0.9886),
        ('DehazeFormer-B',     2023, '~25M',  40.17, 0.9960),
    ]

    p(f"\n{'='*65}")
    p(f"  COMPARISON WITH EXISTING METHODS")
    p(f"{'='*65}")
    p(f"  {'Method':<20s} {'Year':>5s} {'Params':>8s} {'PSNR':>9s} {'SSIM':>8s}")
    p(f"  {'-'*55}")
    for name, year, params, psnr, ssim in literature:
        p(f"  {name:<20s} {year:>5d} {params:>8s} {psnr:>8.2f}  {ssim:>7.4f}")
    p(f"  {'SFD-Net (Ours)':<20s} {'2024':>5s} {'~431K':>8s} {our_psnr:>8.2f}  {our_ssim:>7.4f}  <--")
    p(f"{'='*65}")

    # ── 7. Efficiency ranking ────────────────────────────────────────
    eff_data = [
        ('AOD-Net',        0.002, 19.06),
        ('DehazeNet',      0.008, 21.14),
        ('GFN',            0.500, 22.30),
        ('SFD-Net (Ours)', 0.431, our_psnr),
        ('FFA-Net',        4.500, 36.39),
        ('DehazeFormer-B', 25.00, 40.17),
    ]
    eff_data.sort(key=lambda x: -x[2] / x[1])

    p(f"\n{'='*65}")
    p(f"  EFFICIENCY RANKING (PSNR per Million Parameters)")
    p(f"{'='*65}")
    p(f"  {'#':>3s}  {'Method':<20s} {'Params':>8s} {'PSNR':>8s} {'PSNR/M':>10s}")
    p(f"  {'-'*55}")
    for i, (name, params, psnr) in enumerate(eff_data, 1):
        eff = psnr / params
        marker = '  <--' if 'Ours' in name else ''
        p(f"  {i:>3d}  {name:<20s} {params:>7.3f}M {psnr:>7.2f} {eff:>9.1f}{marker}")
    p(f"{'='*65}")

    # ── 8. Final report card ─────────────────────────────────────────
    g = _grade(our_psnr)
    p(f"\n{'='*65}")
    p(f"  FINAL REPORT CARD")
    p(f"{'='*65}")
    p(f"  Model          : SFD-Net (~431K parameters)")
    p(f"  Test Set       : {args.data_root}")
    p(f"  Grade          : {g}")
    p(f"  PSNR           : {our_psnr:.3f} dB")
    p(f"  SSIM           : {our_ssim:.4f}")
    p(f"  Images         : {n}")
    p(f"  FPS            : {1.0 / np.mean(times):.1f}")
    p(f"  PSNR >= 25 dB  : {psnr_ge25}/{n} ({psnr_ge25/n*100:.1f}%)")
    p(f"  PSNR >= 30 dB  : {psnr_ge30}/{n} ({psnr_ge30/n*100:.1f}%)")
    p(f"  SSIM >= 0.90   : {ssim_ge90}/{n} ({ssim_ge90/n*100:.1f}%)")
    p(f"  SSIM >= 0.95   : {ssim_ge95}/{n} ({ssim_ge95/n*100:.1f}%)")
    p()
    p(f"  Grade scale: A+ (>=35) A (>=30) B+ (>=27) B (>=25)")
    p(f"               C  (>=22) D (>=20) F  (<20)")
    p(f"{'='*65}")

    # ── Save full report ─────────────────────────────────────────────
    report_path = os.path.join(out, 'test_results.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nFull report saved to {report_path}")
    print(f"Per-image CSV saved to {os.path.join(out, 'per_image_results.csv')}")

    report_json = {
        'test_set': args.data_root,
        'checkpoint': args.checkpoint,
        'n_images': n,
        'psnr_mean': round(float(np.mean(psnr_vals)), 4),
        'psnr_std': round(float(np.std(psnr_vals)), 4),
        'psnr_min': round(float(np.min(psnr_vals)), 4),
        'psnr_max': round(float(np.max(psnr_vals)), 4),
        'psnr_median': round(float(np.median(psnr_vals)), 4),
        'ssim_mean': round(float(np.mean(ssim_vals)), 4),
        'ssim_std': round(float(np.std(ssim_vals)), 4),
        'ssim_min': round(float(np.min(ssim_vals)), 4),
        'ssim_max': round(float(np.max(ssim_vals)), 4),
        'ssim_median': round(float(np.median(ssim_vals)), 4),
        'fps': round(1.0 / np.mean(times), 1),
        'grade': g,
        'psnr_ge_25': psnr_ge25,
        'psnr_ge_30': psnr_ge30,
        'ssim_ge_090': ssim_ge90,
        'ssim_ge_095': ssim_ge95,
    }
    json_path = os.path.join(out, 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(report_json, f, indent=2)
    print(f"Machine-readable results saved to {json_path}")


def main():
    p = argparse.ArgumentParser(
        description="Test SFD-Net on SOTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--checkpoint', required=True,
                   help="Path to trained model checkpoint (.pth)")
    p.add_argument('--data-root', required=True,
                   help="SOTS subset root (must contain hazy/ and clear/ subdirs)")
    p.add_argument('--output-dir', default='output/test_sots',
                   help="Where to save results (default: output/test_sots)")
    p.add_argument('--hazy-subdir', default='hazy',
                   help="Name of hazy images subdirectory (default: hazy)")
    p.add_argument('--gt-subdir', default='clear',
                   help="Name of ground truth subdirectory (default: clear)")
    p.add_argument('--save-images', action='store_true',
                   help="Save dehazed output images")
    p.add_argument('--device', default='cuda',
                   help="'cuda' or 'cpu' (default: cuda)")

    test(p.parse_args())


if __name__ == '__main__':
    main()
