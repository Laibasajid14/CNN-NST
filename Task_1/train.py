"""
train.py — Training script for Task 1 CNN seed counting.

Runs the following experiments in sequence:
  1. Model A + Adam  (classification, 30+ epochs, early stopping)
  2. Model A + SGD   (classification, same spec) — optimizer comparison
  3. Model B + winner optimizer (classification)

All results are logged to CSV and plots are saved automatically.
Run:  python train.py
"""

import os
import sys
import json
import csv
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# ── Add paths for imports ─────────────────────────────────────────────────────
TASK1_DIR = Path(__file__).parent
PROJECT_ROOT = TASK1_DIR.parent
sys.path.insert(0, str(TASK1_DIR))

from data   import get_dataloaders, bin_to_count, count_to_bin, BIN_MIDPOINTS
from models import build_model

# ── Load config ───────────────────────────────────────────────────────────────
CONFIG_PATH = TASK1_DIR / 'config.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

SEED         = cfg['training']['seed']
IMG_SIZE     = cfg['training']['img_size']
BATCH_SIZE   = cfg['training']['batch_size']
MAX_EPOCHS   = cfg['training']['max_epochs']
PATIENCE     = cfg['training']['patience']
LR_ADAM      = cfg['training']['lr_adam']
LR_SGD       = cfg['training']['lr_sgd']
MOMENTUM     = cfg['training']['momentum']
WEIGHT_DECAY = cfg['training']['weight_decay']
MODE         = cfg['training']['mode']          # 'regression' or 'classification'
NUM_CLASSES  = cfg['training']['num_classes']

# Convert relative paths to absolute (relative to Task_1 directory)
def resolve_path(rel_path):
    p = Path(rel_path)
    if not p.is_absolute():
        p = TASK1_DIR / p
    return p.resolve()

IMAGE_DIR = resolve_path(cfg['paths']['filtered_images'])
GT_CSV    = resolve_path(cfg['paths']['ground_truth'])
OUT_DIR   = resolve_path(cfg['paths']['cnn_outputs'])
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR = OUT_DIR / 'weights'
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR = OUT_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR = OUT_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

print(f"Config paths:")
print(f"  IMAGE_DIR:  {IMAGE_DIR}")
print(f"  GT_CSV:     {GT_CSV}")
print(f"  OUT_DIR:    {OUT_DIR}")

# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ── Loss helpers ──────────────────────────────────────────────────────────────
def get_loss_fn():
    if MODE == 'regression':
        return nn.MSELoss()
    else:
        return nn.NLLLoss() # nn.CrossEntropyLoss() = nn.LogSoftmax() + nn.NLLLoss()

def preds_to_counts(outputs, mode):
    """Convert raw model output to predicted seed count."""
    if mode == 'regression':
        return outputs.squeeze(1).detach().cpu().numpy()
    else:
        bin_preds = outputs.argmax(dim=1).detach().cpu().numpy()
        return np.array([BIN_MIDPOINTS[b] for b in bin_preds])


# ── Training / validation pass ────────────────────────────────────────────────
def run_epoch(model, loader, loss_fn, optimizer=None, mode='regression'):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.set_grad_enabled(is_train):
        for imgs, labels, _ in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)

            if mode == 'regression':
                loss = loss_fn(outputs.squeeze(1), labels.float())
            else:
                loss = loss_fn(outputs, labels.long())

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(preds_to_counts(outputs, mode).tolist())

            if mode == 'regression':
                all_targets.extend(labels.cpu().numpy().tolist())
            else:
                all_targets.extend(
                    [BIN_MIDPOINTS[b] for b in labels.cpu().numpy()])

    n        = len(loader.dataset)
    avg_loss = total_loss / n
    preds    = np.array(all_preds)
    targets  = np.array(all_targets)
    mae      = np.mean(np.abs(preds - targets))
    return avg_loss, mae, preds, targets


# ── Training run ──────────────────────────────────────────────────────────────
def train_model(model_name, optimizer_name='adam', label=None):
    label = label or f"{model_name}_{optimizer_name}"
    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"{'='*60}")

    set_seed(SEED)

    train_loader, val_loader, test_loader = get_dataloaders(
        IMAGE_DIR, GT_CSV, img_size=IMG_SIZE,
        batch_size=BATCH_SIZE, mode=MODE, seed=SEED)

    model    = build_model(model_name, mode=MODE, num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn  = get_loss_fn()

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=LR_ADAM,
                               weight_decay=WEIGHT_DECAY)
    else:  # sgd
        optimizer = optim.SGD(model.parameters(), lr=LR_SGD,
                              momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights_path = WEIGHTS_DIR / f"{label}_best.pt"

    log_path = OUT_DIR / f"{label}_log.csv"
    with open(log_path, 'w', newline='') as lf:
        writer = csv.writer(lf)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_mae', 'val_mae', 'lr'])

        for epoch in range(1, MAX_EPOCHS + 1):
            t0 = time.time()
            tr_loss, tr_mae, _, _ = run_epoch(
                model, train_loader, loss_fn, optimizer, MODE)
            va_loss, va_mae, _, _ = run_epoch(
                model, val_loader, loss_fn, None, MODE)

            scheduler.step(va_loss)
            lr = optimizer.param_groups[0]['lr']

            history['train_loss'].append(tr_loss)
            history['val_loss'].append(va_loss)
            history['train_mae'].append(tr_mae)
            history['val_mae'].append(va_mae)

            writer.writerow([epoch, f"{tr_loss:.4f}", f"{va_loss:.4f}",
                             f"{tr_mae:.2f}", f"{va_mae:.2f}", f"{lr:.2e}"])
            lf.flush()

            print(f"Epoch {epoch:3d} | "
                  f"tr_loss {tr_loss:.4f} tr_mae {tr_mae:.2f} | "
                  f"va_loss {va_loss:.4f} va_mae {va_mae:.2f} | "
                  f"lr {lr:.2e} | {time.time()-t0:.1f}s")

            # Early stopping
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_weights_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}.")
                    break

    # If no improvement was made (e.g., due to NaN losses), save the final model state
    if best_val_loss == float('inf'):
        torch.save(model.state_dict(), best_weights_path)
        best_val_loss = va_loss if not np.isnan(va_loss) else float('inf')
        print("No improvement during training, saving final model state.")

    # ── Evaluate best model on test set ──────────────────────────────────────
    model.load_state_dict(torch.load(best_weights_path, map_location=DEVICE, weights_only=True))
    _, test_mae, test_preds, test_targets = run_epoch(
        model, test_loader, loss_fn, None, MODE)

    test_preds   = np.round(test_preds).astype(int)
    test_targets = test_targets.astype(int)
    errors       = np.abs(test_preds - test_targets)
    rmse         = float(np.sqrt(np.mean(errors**2)))
    acc          = float(np.mean(errors / np.where(test_targets > 0, test_targets, 1) * 100 <= 10) * 100)

    print(f"\nTest results — MAE: {test_mae:.2f}  RMSE: {rmse:.2f}  Acc: {acc:.1f}%")

    # ── Save plots ────────────────────────────────────────────────────────────
    _plot_curves(history, label)

    return {
        'label':    label,
        'test_mae': round(float(test_mae), 2),
        'test_rmse': round(rmse, 2),
        'accuracy': round(acc, 1),
        'epochs_run': len(history['train_loss']),
        'best_val_loss': round(best_val_loss, 4),
        'weights': str(best_weights_path),
    }, model, test_loader


def _plot_curves(history, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='train')
    ax1.plot(epochs, history['val_loss'],   label='val')
    ax1.set_title(f'{label} — Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, history['train_mae'], label='train')
    ax2.plot(epochs, history['val_mae'],   label='val')
    ax2.set_title(f'{label} — MAE')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MAE (seeds)')
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{label}_curves.png', dpi=150)
    plt.close()
    print(f"Saved curves → {PLOTS_DIR / f'{label}_curves.png'}")


# ── Failure-case analysis ──────────────────────────────────────────────────────
def analyze_failure_cases(model, test_loader, label, mode):
    """Check which Assignment 2 failure cases the CNN now handles correctly."""
    failure_path = resolve_path(cfg['paths']['failure_cases'])
    
    if not failure_path.exists():
        print(f"failure_cases.json not found at {failure_path} — skipping failure analysis")
        return {}

    with open(failure_path) as f:
        failures = {item['filename']: item for item in json.load(f)}

    model.eval()
    loss_fn = get_loss_fn()
    results = {}

    with torch.no_grad():
        for imgs, labels, fnames in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = preds_to_counts(outputs, mode)
            for fname, pred, actual in zip(fnames, preds, labels.numpy()):
                fname_key = fname.replace('_filtered', '').replace('.png', '.jpg')
                if fname_key in failures:
                    pred_int = int(round(float(pred)))
                    actual_int = int(actual if mode == 'regression'
                                     else BIN_MIDPOINTS[int(actual)])
                    pct_err = abs(pred_int - actual_int) / max(actual_int, 1) * 100
                    results[fname_key] = {
                        'actual':         failures[fname_key]['actual_count'],
                        'baseline_pred':  failures[fname_key]['predicted_count'],
                        'cnn_pred':       pred_int,
                        'cnn_fixed':      pct_err <= 10,
                        'baseline_error': failures[fname_key]['error'],
                        'cnn_error':      abs(pred_int - actual_int),
                    }

    n_fixed = sum(1 for v in results.values() if v['cnn_fixed'])
    n_total = len(failures)
    print(f"\nFailure cases fixed by {label}: {n_fixed} / {n_total} "
          f"({100*n_fixed/max(n_total,1):.1f}%)")

    with open(LOGS_DIR / f'{label}_failure_analysis.json', 'w') as f:
        json.dump({'fixed': n_fixed, 'total': n_total,
                   'pct': round(100*n_fixed/max(n_total,1), 1),
                   'details': results}, f, indent=2)

    return results


# ── Confusion matrix ──────────────────────────────────────────────────────────
def plot_predictions(model, test_loader, label, mode):
    """Scatter plot: actual vs predicted counts."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = preds_to_counts(outputs, mode)
            all_preds.extend([int(round(p)) for p in preds])
            if mode == 'regression':
                all_targets.extend(labels.numpy().tolist())
            else:
                all_targets.extend(
                    [BIN_MIDPOINTS[b] for b in labels.numpy()])

    preds   = np.array(all_preds)
    targets = np.array(all_targets)
    mae     = np.mean(np.abs(preds - targets))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, alpha=0.7, edgecolors='k', linewidths=0.5)
    lim = max(targets.max(), preds.max()) + 5
    ax.plot([0, lim], [0, lim], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual count')
    ax.set_ylabel('Predicted count')
    ax.set_title(f'{label} — Actual vs Predicted (MAE={mae:.1f})')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{label}_scatter.png', dpi=150)
    plt.close()
    print(f"Saved scatter → {PLOTS_DIR / f'{label}_scatter.png'}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    all_results = []

    # Experiment 1: Model A + Adam
    res_a_adam, model_a_adam, tl_a = train_model('model_a', 'adam', 'ModelA_Adam')
    all_results.append(res_a_adam)
    analyze_failure_cases(model_a_adam, tl_a, 'ModelA_Adam', MODE)
    plot_predictions(model_a_adam, tl_a, 'ModelA_Adam', MODE)

    # Experiment 2: Model A + SGD
    res_a_sgd, model_a_sgd, tl_a2 = train_model('model_a', 'sgd', 'ModelA_SGD')
    all_results.append(res_a_sgd)
    if not math.isnan(res_a_sgd['test_mae']):
        plot_predictions(model_a_sgd, tl_a2, 'ModelA_SGD', MODE)

    # Pick best optimizer
    best_opt = 'adam' if res_a_adam['best_val_loss'] <= res_a_sgd['best_val_loss'] else 'sgd'
    print(f"\nBest optimizer for Model B: {best_opt}")

    # Experiment 3: Model B + best optimizer
    res_b, model_b, tl_b = train_model('model_b', best_opt, f'ModelB_{best_opt.upper()}')
    all_results.append(res_b)
    analyze_failure_cases(model_b, tl_b, f'ModelB_{best_opt.upper()}', MODE)
    plot_predictions(model_b, tl_b, f'ModelB_{best_opt.upper()}', MODE)

    # ── Save comparison table ─────────────────────────────────────────────────
    # Add Assignment 2 baseline numbers
    baseline_row = {
        'label':     'Clustering_A2',
        'test_mae':  25.05,
        'test_rmse': 42.00,
        'accuracy':  56.0,
        'epochs_run': 'N/A',
        'best_val_loss': 'N/A',
        'weights':   'N/A',
    }
    # Add Assignment 3 edge detection numbers
    edge_row = {
        'label':     'EdgeDetection_A3',
        'test_mae':  43.20,
        'test_rmse': 64.23,
        'accuracy':  12.8,
        'epochs_run': 'N/A',
        'best_val_loss': 'N/A',
        'weights':   'N/A',
    }

    comparison = [baseline_row, edge_row] + all_results

    csv_path = OUT_DIR / 'comparison_table.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)

    print(f"\n{'='*60}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'MAE':>8} {'RMSE':>8} {'Acc%':>8}")
    print('-' * 55)
    for row in comparison:
        print(f"{row['label']:<25} {row['test_mae']:>8} "
              f"{row['test_rmse']:>8} {row['accuracy']:>8}")
    print(f"\nSaved → {csv_path}")
