"""
matting/train.py — Train the U-Net matting model on AISegment dataset.

Dataset structure expected:
  aisegment/
    clip_img/
      <session_id>/
        <sub_id>/
          *.jpg      ← RGB portrait images
    matting/
      <session_id>/
        <sub_id>/
          *.png      ← RGBA matting files (alpha = 4th channel)

Usage:
  python matting/train.py

CPU-feasible settings (default):
  - 2500 train / 300 val / 300 test pairs (subset of 34k)
  - Image size: 256×256
  - 20 epochs, batch 8
  - ~2-4 hours on modern laptop CPU
"""

import os
import sys
import random
import time
import csv
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from model import MattingUNet

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent.parent / 'config_task2.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

SEED        = cfg['matting']['seed']
IMG_SIZE    = cfg['matting']['img_size']       # 256
BATCH       = cfg['matting']['batch_size']     # 8
MAX_EPOCHS  = cfg['matting']['max_epochs']     # 20
PATIENCE    = cfg['matting']['patience']       # 5
LR          = cfg['matting']['lr']
N_TRAIN     = cfg['matting']['n_train']        # 2500
N_VAL       = cfg['matting']['n_val']          # 300
N_TEST      = cfg['matting']['n_test']         # 300

ROOT = Path(__file__).parent.parent
DATASET_ROOT = (ROOT / cfg['paths']['aisegment_root']).resolve()
WEIGHTS_OUT  = (ROOT / cfg['paths']['matting_weights']).resolve()
PLOTS_OUT    = (ROOT / cfg['paths']['matting_plots']).resolve()
OUTPUT_DIR   = (ROOT / cfg['paths']['task2_outputs']).resolve()
WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)
PLOTS_OUT.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────
def find_pairs(root, max_pairs=None):
    """Walk AISegment directory structure and collect (img_path, matte_path) pairs."""
    clip_root    = root / 'clip_img'
    matting_root = root / 'matting'
    pairs = []

    for session in sorted(clip_root.iterdir()):
        if not session.is_dir():
            continue
        mat_session = matting_root / session.name
        if not mat_session.exists():
            continue

        for clip_sub in sorted(session.iterdir()):
            if not clip_sub.is_dir():
                continue
            for img_file in sorted(clip_sub.glob('*.jpg')):
                alpha_name = img_file.stem + '.png'

                # Try exact matching subfolder name first.
                matte_path = mat_session / clip_sub.name / alpha_name

                # Fallback if matting folder uses a mirrored prefix like matting_000000xx.
                if not matte_path.exists() and clip_sub.name.startswith('clip_'):
                    alt_sub = 'matting_' + clip_sub.name[len('clip_'):]
                    matte_path = mat_session / alt_sub / alpha_name

                # Last resort: scan all subfolders in the session for the alpha file.
                if not matte_path.exists():
                    for mat_sub in sorted(mat_session.iterdir()):
                        if not mat_sub.is_dir():
                            continue
                        candidate = mat_sub / alpha_name
                        if candidate.exists():
                            matte_path = candidate
                            break

                if matte_path.exists():
                    pairs.append((img_file, matte_path))

            if max_pairs and len(pairs) >= max_pairs:
                break
        if max_pairs and len(pairs) >= max_pairs:
            break

    random.shuffle(pairs)
    return pairs


class AISegmentDataset(Dataset):
    def __init__(self, pairs, img_size=256, augment=False):
        self.pairs    = pairs
        self.img_size = img_size
        self.augment  = augment
        self.normalize = T.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, matte_path = self.pairs[idx]

        # Load image (BGR→RGB)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load matte — extract alpha channel
        matte_rgba = cv2.imread(str(matte_path), cv2.IMREAD_UNCHANGED)
        if matte_rgba is not None and matte_rgba.shape[2] == 4:
            alpha = matte_rgba[:, :, 3].astype(np.float32) / 255.0
        else:
            # Fallback: use luminance of the matting image
            if matte_rgba is None:
                alpha = np.ones((img.shape[0], img.shape[1]), np.float32)
            else:
                gray  = cv2.cvtColor(matte_rgba[:, :, :3], cv2.COLOR_BGR2GRAY)
                alpha = gray.astype(np.float32) / 255.0

        # Resize
        img   = cv2.resize(img, (self.img_size, self.img_size))
        alpha = cv2.resize(alpha, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                img   = np.fliplr(img).copy()
                alpha = np.fliplr(alpha).copy()
            # Color jitter
            bf = random.uniform(0.8, 1.2)
            img = np.clip(img.astype(np.float32) * bf, 0, 255).astype(np.uint8)

        img_t   = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t   = self.normalize(img_t)
        alpha_t = torch.from_numpy(alpha).unsqueeze(0)  # (1, H, W)

        return img_t, alpha_t


# ── Loss ──────────────────────────────────────────────────────────────────────
def matting_loss(pred_alpha, true_alpha, l1_weight=0.5, bce_weight=0.5):
    """Combined L1 + BCE loss as specified in assignment."""
    l1  = nn.L1Loss()(pred_alpha, true_alpha)
    bce = nn.BCELoss()(pred_alpha.clamp(1e-6, 1-1e-6), true_alpha)
    return l1_weight * l1 + bce_weight * bce


def compute_iou(pred, target, threshold=0.5):
    pred_bin   = (pred > threshold).float()
    target_bin = (target > threshold).float()
    inter = (pred_bin * target_bin).sum()
    union = (pred_bin + target_bin).clamp(0, 1).sum()
    return (inter / (union + 1e-6)).item()


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    print("Scanning AISegment dataset...")
    all_pairs = find_pairs(DATASET_ROOT, max_pairs=N_TRAIN + N_VAL + N_TEST + 100)

    if len(all_pairs) < N_TRAIN + N_VAL + N_TEST:
        print(f"WARNING: Only {len(all_pairs)} pairs found. "
              f"Need {N_TRAIN+N_VAL+N_TEST}. Using all available.")

    n = len(all_pairs)
    n_train = min(N_TRAIN, int(0.7 * n))
    n_val   = min(N_VAL,   int(0.15 * n))

    train_pairs = all_pairs[:n_train]
    val_pairs   = all_pairs[n_train:n_train + n_val]
    test_pairs  = all_pairs[n_train + n_val:n_train + n_val + N_TEST]

    print(f"Pairs — train: {len(train_pairs)}, val: {len(val_pairs)}, "
          f"test: {len(test_pairs)}")

    train_loader = DataLoader(
        AISegmentDataset(train_pairs, IMG_SIZE, augment=True),
        batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(
        AISegmentDataset(val_pairs,   IMG_SIZE, augment=False),
        batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(
        AISegmentDataset(test_pairs,  IMG_SIZE, augment=False),
        batch_size=BATCH, shuffle=False, num_workers=0)

    model     = MattingUNet(base_ch=32).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                      factor=0.5, patience=3)
    print(f"Model parameters: {model.count_params():,}")

    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    best_iou = 0.0
    patience_counter = 0
    best_path = WEIGHTS_OUT / 'matting_best.pt'

    log_path = WEIGHTS_OUT / 'matting_log.csv'
    with open(log_path, 'w', newline='') as lf:
        writer = csv.writer(lf)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou'])

        for epoch in range(1, MAX_EPOCHS + 1):
            t0 = time.time()

            # Train
            model.train()
            tr_loss = 0.0
            for imgs, alphas in train_loader:
                imgs, alphas = imgs.to(DEVICE), alphas.to(DEVICE)
                preds = model(imgs)
                loss  = matting_loss(preds, alphas)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * imgs.size(0)
            tr_loss /= len(train_loader.dataset)

            # Validate
            model.eval()
            va_loss, va_iou = 0.0, 0.0
            with torch.no_grad():
                for imgs, alphas in val_loader:
                    imgs, alphas = imgs.to(DEVICE), alphas.to(DEVICE)
                    preds   = model(imgs)
                    loss    = matting_loss(preds, alphas)
                    va_loss += loss.item() * imgs.size(0)
                    va_iou  += compute_iou(preds, alphas) * imgs.size(0)
            va_loss /= len(val_loader.dataset)
            va_iou  /= len(val_loader.dataset)

            scheduler.step(va_loss)

            history['train_loss'].append(tr_loss)
            history['val_loss'].append(va_loss)
            history['val_iou'].append(va_iou)
            writer.writerow([epoch, f'{tr_loss:.4f}', f'{va_loss:.4f}', f'{va_iou:.4f}'])
            lf.flush()

            print(f"Epoch {epoch:3d} | tr_loss {tr_loss:.4f} | "
                  f"va_loss {va_loss:.4f} | va_iou {va_iou:.4f} | "
                  f"{time.time()-t0:.1f}s")

            if va_iou > best_iou:
                best_iou = va_iou
                patience_counter = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}.")
                    break

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()
    test_iou = 0.0
    with torch.no_grad():
        for imgs, alphas in test_loader:
            imgs, alphas = imgs.to(DEVICE), alphas.to(DEVICE)
            preds = model(imgs)
            test_iou += compute_iou(preds, alphas) * imgs.size(0)
    test_iou /= max(len(test_loader.dataset), 1)
    print(f"\nTest IoU: {test_iou:.4f}  (target: ≥ 0.85)")

    # ── Save plots ────────────────────────────────────────────────────────────
    ep = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ep, history['train_loss'], label='train')
    ax1.plot(ep, history['val_loss'],   label='val')
    ax1.set_title('Matting Loss'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(True)
    ax2.plot(ep, history['val_iou'], color='green')
    ax2.axhline(0.85, color='red', linestyle='--', label='Target IoU=0.85')
    ax2.set_title('Validation IoU'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_OUT / 'matting_curves.png', dpi=150)
    plt.close()

    # ── Matting visualization (5 samples) ─────────────────────────────────────
    sample_imgs, sample_alphas = next(iter(test_loader))
    sample_imgs = sample_imgs[:5].to(DEVICE)
    with torch.no_grad():
        pred_alphas = model(sample_imgs).cpu()

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i in range(5):
        img_np  = sample_imgs[i].cpu().permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = np.clip(img_np * std + mean, 0, 1)

        axes[0, i].imshow(img_np); axes[0, i].set_title('Input')
        axes[0, i].axis('off')
        axes[1, i].imshow(pred_alphas[i, 0].numpy(), cmap='gray')
        axes[1, i].set_title('Pred Alpha'); axes[1, i].axis('off')
        # Cutout (subject isolated)
        cutout = img_np * pred_alphas[i, 0].numpy()[:, :, None]
        axes[2, i].imshow(cutout); axes[2, i].set_title('Cutout')
        axes[2, i].axis('off')

    plt.suptitle(f'Matting Visualization — Test IoU: {test_iou:.4f}')
    plt.tight_layout()
    overlay_out = OUTPUT_DIR / 'matting_overlay.png'
    plt.savefig(overlay_out, dpi=150)
    plt.close()
    print(f"Saved matting_overlay.png → {overlay_out}")

    return test_iou


if __name__ == '__main__':
    train()
