"""
data.py — Dataset loading and augmentation for Task 1 CNN seed counting.

The seed task has 141 images with counts 1–144. We treat this as a
REGRESSION problem (predict the count directly) because:
  - 144 unique count values → too many classes for a 141-image dataset
  - Regression naturally produces MAE/RMSE metrics matching Assignments 2 & 3
  - The assignment's Softmax/cross-entropy spec is interpreted as a classification
    over 15 count bins (1–10, 11–20, … 131–144) for the architecture comparison,
    with final count predicted as the bin midpoint for fair metric comparison.

We implement BOTH heads so you can ablate them:
  - regression head: single output neuron, MSE loss
  - classification head: 15-class softmax, cross-entropy loss
    (predicted count = bin midpoint, compatible with calculate_metrics)
"""

import os
import json
import csv
import random
import numpy as np
import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ── Bin definitions (classification mode) ─────────────────────────────────────
NUM_BINS = 15
BIN_SIZE = 10   # counts 1-10 → bin 0, 11-20 → bin 1, …
BIN_MIDPOINTS = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 142]

def count_to_bin(count):
    """Map raw count to class index (0-14)."""
    return min((count - 1) // BIN_SIZE, NUM_BINS - 1)

def bin_to_count(bin_idx):
    """Map class index to predicted count (bin midpoint)."""
    return BIN_MIDPOINTS[bin_idx]


# ── Dataset ───────────────────────────────────────────────────────────────────
class SeedDataset(Dataset):
    """
    Loads preprocessed filtered seed images and ground-truth counts.

    Args:
        image_dir   : path to intermediate_outputs/preprocessed_images/filtered/
        gt_csv      : path to data/ground_truth/counts.csv
        img_size    : resize target (H, W) — default 224×224
        augment     : apply training augmentation
        mode        : 'regression' | 'classification'
        split       : 'train' | 'val' | 'test'
        seed        : random seed for deterministic split
    """
    def __init__(self, image_dir, gt_csv, img_size=224,
                 augment=False, mode='regression',
                 split='train', seed=42):
        self.image_dir = Path(image_dir)
        self.img_size  = img_size
        self.augment   = augment
        self.mode      = mode

        # Load ground truth CSV
        gt = {}
        with open(gt_csv, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename'].strip()
                count = int(row['actual_count'].strip())
                gt[filename] = count

        # Collect images present on disk and match to ground truth
        all_files = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])
        
        samples = []
        for fp in all_files:
            # Try direct match first (e.g., "1.jpg" matches "1.jpg")
            if fp.name in gt:
                samples.append((fp, gt[fp.name]))
            # Try without extension variation
            elif fp.stem in gt or f"{fp.stem}.jpg" in gt:
                key = f"{fp.stem}.jpg" if f"{fp.stem}.jpg" in gt else fp.stem
                if key in gt:
                    samples.append((fp, gt[key]))

        if not samples:
            raise ValueError(
                f"No samples found matching ground truth. "
                f"Checked {len(all_files)} image files against {len(gt)} GT entries.")
        
        print(f"Loaded {len(samples)} samples from {self.image_dir}")

        # Deterministic 70/15/15 split
        rng = random.Random(seed)
        rng.shuffle(samples)
        n = len(samples)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        if split == 'train':
            self.samples = samples[:n_train]
        elif split == 'val':
            self.samples = samples[n_train:n_train + n_val]
        else:
            self.samples = samples[n_train + n_val:]

        # Normalisation (ImageNet stats — seeds are natural images)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.samples)

    def _augment(self, img):
        """Training augmentation matching assignment spec."""
        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
        # Rotation ±30°
        angle = random.uniform(-30, 30)
        img = TF.rotate(img, angle)
        # Brightness ±20%
        bf = random.uniform(0.8, 1.2)
        img = TF.adjust_brightness(img, bf)
        # Zoom ±10% (random resized crop)
        scale = random.uniform(0.9, 1.1)
        h = w = self.img_size
        ch = int(h / scale)
        cw = int(w / scale)
        img_pil = T.ToPILImage()(img)
        img_pil = T.RandomCrop((min(ch, img_pil.height), min(cw, img_pil.width)))(img_pil)
        img_pil = T.Resize((h, w))(img_pil)
        img = T.ToTensor()(img_pil)
        return img

    def __getitem__(self, idx):
        fp, count = self.samples[idx]

        # Load as BGR, convert to RGB, resize
        bgr = cv2.imread(str(fp))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read {fp}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size))

        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        if self.augment:
            img = self._augment(img)

        img = self.normalize(img)

        if self.mode == 'regression':
            label = torch.tensor(float(count), dtype=torch.float32)
        else:
            label = torch.tensor(count_to_bin(count), dtype=torch.long)

        return img, label, fp.name


def get_dataloaders(image_dir, gt_csv, img_size=224, batch_size=16,
                    mode='regression', seed=42, num_workers=0):
    """Return train / val / test DataLoaders."""
    train_ds = SeedDataset(image_dir, gt_csv, img_size, augment=True,
                           mode=mode, split='train', seed=seed)
    val_ds   = SeedDataset(image_dir, gt_csv, img_size, augment=False,
                           mode=mode, split='val',   seed=seed)
    test_ds  = SeedDataset(image_dir, gt_csv, img_size, augment=False,
                           mode=mode, split='test',  seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    print(f"Dataset splits — train: {len(train_ds)}, "
          f"val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
