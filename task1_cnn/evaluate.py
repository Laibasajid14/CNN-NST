"""
evaluate.py — Loads saved CNN weights and evaluates on the test set.
Produces the unified comparison table (Clustering → Edge → CNN A → CNN B).
Run after train.py has completed.
Usage:  python evaluate.py
"""

import sys
import json
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

from data   import get_dataloaders, BIN_MIDPOINTS
from models import build_model

CONFIG_PATH = Path(__file__).parent / 'config.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

SEED      = cfg['training']['seed']
IMG_SIZE  = cfg['training']['img_size']
BATCH     = cfg['training']['batch_size']
MODE      = cfg['training']['mode']
NUM_CLS   = cfg['training']['num_classes']

# Convert relative paths to absolute (relative to Task_1 directory)
TASK1_DIR = Path(__file__).parent
def resolve_path(rel_path):
    p = Path(rel_path)
    if not p.is_absolute():
        p = TASK1_DIR / p
    return p.resolve()

IMAGE_DIR = resolve_path(cfg['paths']['filtered_images'])
GT_CSV    = resolve_path(cfg['paths']['ground_truth'])
OUT_DIR   = resolve_path(cfg['paths']['cnn_outputs'])
PLOTS_DIR = OUT_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preds_to_counts(outputs, mode):
    if mode == 'regression':
        return outputs.squeeze(1).detach().cpu().numpy()
    else:
        bin_preds = outputs.argmax(dim=1).detach().cpu().numpy()
        return np.array([BIN_MIDPOINTS[b] for b in bin_preds])


def evaluate(model, loader, mode):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = preds_to_counts(outputs, mode)
            all_preds.extend([int(round(p)) for p in preds])
            if mode == 'regression':
                all_targets.extend([int(l) for l in labels.numpy()])
            else:
                all_targets.extend([BIN_MIDPOINTS[int(l)] for l in labels.numpy()])

    preds   = np.array(all_preds)
    targets = np.array(all_targets)
    errors  = np.abs(preds - targets)
    mae     = float(np.mean(errors))
    rmse    = float(np.sqrt(np.mean(errors**2)))
    pct_err = errors / np.where(targets > 0, targets, 1) * 100
    acc     = float(np.mean(pct_err <= 10) * 100)
    return mae, rmse, acc, preds, targets


def load_failure_cases():
    failure_path = resolve_path(cfg['paths']['failure_cases'])
    if failure_path.exists():
        with open(failure_path) as f:
            cases = json.load(f)
        return {c['filename']: c for c in cases}
    print(f"WARNING: failure_cases.json not found at {failure_path}")
    return {}


def count_fixed(preds, targets, filenames, failures):
    """Count failure cases from A2 that CNN now handles within 10%."""
    fixed = 0
    for fname, pred, actual in zip(filenames, preds, targets):
        key = fname.replace('_filtered', '').replace('.png', '.jpg')
        if key in failures:
            pct = abs(pred - actual) / max(actual, 1) * 100
            if pct <= 10:
                fixed += 1
    return fixed, len(failures)


if __name__ == '__main__':
    _, _, test_loader = get_dataloaders(
        IMAGE_DIR, GT_CSV, img_size=IMG_SIZE,
        batch_size=BATCH, mode=MODE, seed=SEED)

    failures = load_failure_cases()
    weights_dir = OUT_DIR / 'weights'

    rows = [
        {'Method': 'Clustering (A2)',      'MAE': 25.05, 'RMSE': 42.00, 'Acc%': 56.0,  'Failure Fixed': 'reference'},
        {'Method': 'Edge Detection (A3)',  'MAE': 43.20, 'RMSE': 64.23, 'Acc%': 12.8,  'Failure Fixed': '1 / 62 (1.9%)'},
    ]

    configs = [
        ('model_a', 'ModelA_Adam_best.pt', 'CNN Model A (Adam)'),
        ('model_a', 'ModelA_SGD_best.pt',  'CNN Model A (SGD)'),
        ('model_b', 'ModelB_*.pt',         'CNN Model B'),
    ]

    for model_name, weight_pattern, label in configs:
        # Find weights file
        if weight_pattern.endswith('*.pt'):
            # Use glob to find Model B variant
            candidates = list(weights_dir.glob(weight_pattern))
            wp = candidates[0] if candidates else None
        else:
            wp = weights_dir / weight_pattern

        if wp is None or not wp.exists():
            print(f"Weights not found for {label} at {wp} — skipping")
            continue

        model = build_model(model_name, mode=MODE, num_classes=NUM_CLS).to(DEVICE)
        model.load_state_dict(torch.load(wp, map_location=DEVICE))

        mae, rmse, acc, preds, targets = evaluate(model, test_loader, MODE)

        # Get filenames for failure case analysis
        fnames = []
        for _, _, fns in test_loader:
            fnames.extend(fns)
        n_fixed, n_total = count_fixed(preds, targets, fnames, failures)

        rows.append({
            'Method':        label,
            'MAE':           round(mae, 2),
            'RMSE':          round(rmse, 2),
            'Acc%':          round(acc, 1),
            'Failure Fixed': f"{n_fixed} / {n_total} ({100*n_fixed/max(n_total,1):.1f}%)",
        })
        print(f"{label}: MAE={mae:.2f}  RMSE={rmse:.2f}  Acc={acc:.1f}%  "
              f"Fixed={n_fixed}/{n_total}")

    # Save table
    table_path = OUT_DIR / 'comparison_table.csv'
    with open(table_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved comparison table → {table_path}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    methods = [r['Method'] for r in rows]
    maes    = [r['MAE'] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
    bars    = ax.bar(methods, maes, color=colors[:len(methods)], edgecolor='black')
    ax.set_ylabel('MAE (seeds)')
    ax.set_title('Unified Method Comparison — MAE on Seed Counting Task')
    ax.bar_label(bars, fmt='%.2f', padding=3)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'unified_comparison.png', dpi=150)
    plt.close()
    print(f"Saved comparison chart → {PLOTS_DIR / 'unified_comparison.png'}")
