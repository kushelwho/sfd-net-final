#!/usr/bin/env python3
"""
Train SFD-Net on a single domain (ITS or OTS).

Features: AdamW + cosine LR with warmup, early stopping on val PSNR,
checkpoint resume, DataParallel, metrics CSV.

Usage:
    python train.py --domain its --data-root data/ITS --saliency-root saliency/its --output-dir outputs/its
    python train.py --domain ots --data-root data/OTS --saliency-root saliency/ots --output-dir outputs/ots

Typical workflow:
    1. python generate_saliency.py --data-root data/ITS --out-dir saliency/its
    2. python generate_saliency.py --data-root data/OTS --out-dir saliency/ots
    3. python train.py --domain its ...
    4. python train.py --domain ots ...
"""

import argparse
import os
import time
import statistics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.sfd_net import SFDNet
from src.losses import CombinedLoss
from src.dataset import DehazingDataset
from src.utils import AverageMeter, MetricsLogger, compute_psnr, compute_ssim

# ── defaults (override via CLI) ──────────────────────────────────────────

EPOCHS       = {'its': 200, 'ots': 100}
WARMUP       = 5
LR           = 2e-4
LR_MIN       = 1e-6
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 16
VAL_EVERY    = 5
NUM_WORKERS  = 4
PATIENCE     = 10


# ── checkpoint helpers ───────────────────────────────────────────────────

def _state_dict(model):
    m = model.module if isinstance(model, nn.DataParallel) else model
    return m.state_dict()


def save_checkpoint(path, epoch, model, optimizer, scheduler,
                    best_psnr, patience_counter=0):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        'epoch':            epoch,
        'model_state':      _state_dict(model),
        'optimizer_state':  optimizer.state_dict(),
        'scheduler_state':  scheduler.state_dict(),
        'best_psnr':        best_psnr,
        'patience_counter': patience_counter,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state']
    if any(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    raw = model.module if isinstance(model, nn.DataParallel) else model
    raw.load_state_dict(sd)
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    return ckpt['epoch'], ckpt.get('best_psnr', 0.0), ckpt.get('patience_counter', 0)


# ── validation ───────────────────────────────────────────────────────────

def validate(model, loader, device):
    model.eval()
    psnr_m, ssim_m = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for hazy, clear, _ in tqdm(loader, desc='  val', leave=False):
            pred = model(hazy.to(device))
            clear = clear.to(device)
            psnr_m.update(compute_psnr(pred, clear), n=hazy.size(0))
            ssim_m.update(compute_ssim(pred, clear), n=hazy.size(0))
    model.train()
    return psnr_m.avg, ssim_m.avg


# ── training ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    total_epochs = args.epochs or EPOCHS.get(args.domain, 200)
    patience     = args.patience
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Domain  : {args.domain.upper()}")
    print(f"  Epochs  : {total_epochs}")
    print(f"  Patience: {patience} val rounds (= {patience * args.val_every} epochs)")
    print(f"  Device  : {device}")
    print(f"  Output  : {args.output_dir}")
    print(f"{'='*55}\n")

    # data
    train_ds = DehazingDataset(args.domain, 'train', args.data_root, args.saliency_root)
    val_ds   = DehazingDataset(args.domain, 'val',   args.data_root, args.saliency_root)

    pw = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=pw)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False,
                              num_workers=2, pin_memory=True,
                              persistent_workers=False)

    # model
    model = SFDNet()
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = LinearLR(optimizer, start_factor=LR_MIN / args.lr,
                      end_factor=1.0, total_iters=args.warmup)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - args.warmup, eta_min=LR_MIN)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup])

    criterion = CombinedLoss().to(device)

    # resume
    start_epoch = 0
    best_psnr = best_ssim = 0.0
    patience_counter = 0
    batch_times = []
    last_ckpt = os.path.join(args.output_dir, 'last_model.pth')

    if os.path.exists(last_ckpt):
        print(f"Resuming from {last_ckpt}")
        ep, best_psnr, patience_counter = load_checkpoint(
            last_ckpt, model, optimizer, scheduler)
        start_epoch = ep + 1
        print(f"  -> epoch {start_epoch+1}/{total_epochs}  "
              f"best PSNR {best_psnr:.2f} dB  patience {patience_counter}/{patience}\n")

    logger = MetricsLogger(os.path.join(args.output_dir, 'metrics_log.csv'))

    # train loop
    model.train()
    for epoch in range(start_epoch, total_epochs):
        loss_m, charb_m, perc_m = AverageMeter(), AverageMeter(), AverageMeter()

        for hazy, clear, sal in tqdm(train_loader,
                                     desc=f'Epoch {epoch+1:03d}/{total_epochs}',
                                     dynamic_ncols=True):
            t0 = time.time()
            hazy, clear, sal = hazy.to(device), clear.to(device), sal.to(device)

            optimizer.zero_grad()
            total_loss, lc, lp = criterion(model(hazy), clear, sal)
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_times.append(time.time() - t0)
            if len(batch_times) > 500:
                batch_times = batch_times[-500:]

            b = hazy.size(0)
            loss_m.update(total_loss.item(), b)
            charb_m.update(lc.item(), b)
            perc_m.update(lp.item(), b)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        # validate
        val_psnr = val_ssim = 0.0
        do_val = ((epoch + 1) % args.val_every == 0) or (epoch == total_epochs - 1)

        if do_val:
            val_psnr, val_ssim = validate(model, val_loader, device)
            print(f"  Epoch {epoch+1:03d} | "
                  f"Loss {loss_m.avg:.5f} (charb {charb_m.avg:.5f} perc {perc_m.avg:.5f}) | "
                  f"PSNR {val_psnr:.2f}  SSIM {val_ssim:.4f} | LR {lr_now:.2e}")

            if val_psnr > best_psnr:
                best_psnr, best_ssim = val_psnr, val_ssim
                patience_counter = 0
                save_checkpoint(os.path.join(args.output_dir, 'best_model.pth'),
                                epoch, model, optimizer, scheduler,
                                best_psnr, patience_counter)
                print(f"  -> new best ({best_psnr:.2f} dB)")
            else:
                patience_counter += 1
                print(f"  -> no improvement, patience {patience_counter}/{patience}")
        else:
            print(f"  Epoch {epoch+1:03d} | Loss {loss_m.avg:.5f} | LR {lr_now:.2e}")

        save_checkpoint(last_ckpt, epoch, model, optimizer, scheduler,
                        best_psnr, patience_counter)
        logger.log(epoch + 1, loss_m.avg, val_psnr, val_ssim)

        import gc
        torch.cuda.empty_cache()
        gc.collect()

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} — "
                  f"no PSNR gain for {patience} val rounds")
            break

    # summary
    med_t = statistics.median(batch_times) if batch_times else 0.0
    tag = 'ITS (Indoor)' if args.domain == 'its' else 'OTS (Outdoor)'
    summary = (f"\n{'='*55}\n"
               f"  Dataset : {tag}\n"
               f"  Images  : {len(train_ds)}\n"
               f"  PSNR    : {best_psnr:.3f} dB\n"
               f"  SSIM    : {best_ssim:.4f}\n"
               f"  Median t: {med_t:.4f} s/batch\n"
               f"  Output  : {args.output_dir}\n"
               f"{'='*55}")
    print(summary)

    with open(os.path.join(args.output_dir, 'training_summary.txt'), 'w') as f:
        f.write(summary.strip() + '\n')

    return best_psnr, best_ssim


def main():
    p = argparse.ArgumentParser(description="Train SFD-Net")
    p.add_argument('--domain',        required=True, choices=['its', 'ots'])
    p.add_argument('--data-root',     required=True, help="Dataset root with hazy/ and clear/")
    p.add_argument('--saliency-root', required=True, help="Pre-generated saliency maps directory")
    p.add_argument('--output-dir',    required=True, help="Checkpoints + metrics output")

    p.add_argument('--epochs',        type=int, default=0,            help="0 = use domain default (ITS:200, OTS:100)")
    p.add_argument('--warmup',        type=int, default=WARMUP)
    p.add_argument('--lr',            type=float, default=LR)
    p.add_argument('--weight-decay',  type=float, default=WEIGHT_DECAY)
    p.add_argument('--batch-size',    type=int, default=BATCH_SIZE)
    p.add_argument('--val-every',     type=int, default=VAL_EVERY)
    p.add_argument('--num-workers',   type=int, default=NUM_WORKERS)
    p.add_argument('--patience',      type=int, default=PATIENCE)
    p.add_argument('--device',        default='cuda')

    train(p.parse_args())


if __name__ == '__main__':
    main()
