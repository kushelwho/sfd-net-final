"""
Generate saliency maps from ground-truth clear images using U²-Net.
Resume-safe: skips files that already exist in the output directory.

Usage:
    python generate_saliency.py --data-root data/ITS --out-dir saliency/its --domain ITS
    python generate_saliency.py --data-root data/OTS --out-dir saliency/ots --domain OTS
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

U2NET_REPO    = os.path.join(os.path.dirname(__file__), 'U-2-Net')
U2NET_WEIGHTS = os.path.join(U2NET_REPO, 'saved_models', 'u2net', 'u2net.pth')


def load_u2net(device):
    sys.path.insert(0, U2NET_REPO)
    from model.u2net import U2NET  # type: ignore

    net = U2NET(3, 1)
    net.load_state_dict(torch.load(U2NET_WEIGHTS, map_location='cpu', weights_only=True))
    net.eval().to(device)
    return net


_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(net, img_path, device):
    """Returns uint8 grayscale saliency map at original resolution."""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    with torch.no_grad():
        d1 = net(_transform(img).unsqueeze(0).to(device))[0]

    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    pred = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)


def run(data_root, out_dir, domain_label, device):
    clear_dir = os.path.join(data_root, 'clear')
    if not os.path.isdir(clear_dir):
        raise FileNotFoundError(f"Clear directory not found: {clear_dir}")

    os.makedirs(out_dir, exist_ok=True)

    exts = {'.png', '.jpg', '.jpeg'}
    images = sorted(f for f in os.listdir(clear_dir)
                    if os.path.splitext(f)[1].lower() in exts)
    print(f"[{domain_label}] {len(images)} images in {clear_dir}")
    print(f"[{domain_label}] Output -> {out_dir}")

    net = load_u2net(device)
    skipped = failed = 0

    for fname in tqdm(images, desc=f"[{domain_label}]"):
        stem = os.path.splitext(fname)[0]
        dst  = os.path.join(out_dir, stem + '.png')

        if os.path.exists(dst):
            skipped += 1
            continue

        try:
            cv2.imwrite(dst, predict(net, os.path.join(clear_dir, fname), device))
        except Exception as e:
            tqdm.write(f"  WARN: {fname} — {e}")
            failed += 1

    saved = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    print(f"\n[{domain_label}] done — {saved} maps, {skipped} skipped, {failed} failed")

    del net
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description="Generate saliency maps via U²-Net")
    p.add_argument('--data-root', required=True, help="Dataset root (must contain clear/ subdir)")
    p.add_argument('--out-dir',   required=True, help="Where to save saliency PNGs")
    p.add_argument('--domain',    default='',    help="Label for progress bar (e.g. ITS, OTS)")
    p.add_argument('--device',    default='cuda', help="'cuda' or 'cpu'")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    run(args.data_root, args.out_dir, args.domain or os.path.basename(args.data_root), device)


if __name__ == '__main__':
    main()
