"""
nst.py — Neural Style Transfer (Gatys et al., 2015) using pretrained VGG19.

Features:
  - VGG19 frozen backbone, eval mode
  - Content layer: relu4_2
  - Style layers: relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
  - Gram matrix style representation
  - L-BFGS or Adam pixel optimization
  - Beta/alpha sweep (style weight ratios)
  - Layer ablation (shallow vs deep style layers)
  - Temporal consistency: initialize frame t from stylized frame t-1
  - CPU-feasible: 150 iterations, 256x256 default

Usage:
  # Single image NST
  python nst.py --content content/frame_001.jpg --style style/vangogh.jpg

  # Full ablation grid
  python nst.py --ablation

  # Feature map visualization
  python nst.py --feature_maps --content content/frame_001.jpg
"""

import argparse
import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / 'config_task2.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

IMG_SIZE   = cfg['nst']['img_size']           # 256
N_STEPS    = cfg['nst']['n_steps']            # 150
STYLE_W    = cfg['nst']['style_weight']       # default 1e5
CONTENT_W  = cfg['nst']['content_weight']     # 1.0
ROOT = Path(__file__).parent
OUTPUT_DIR = (ROOT / cfg['paths']['task2_outputs']).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"NST Device: {DEVICE}")

# ── VGG19 layer names ────────────────────────────────────────────────────────
CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS  = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

# VGG19 feature layer name mapping
VGG19_LAYER_MAP = {
    '0':  'conv1_1', '1':  'relu1_1', '2':  'conv1_2', '3':  'relu1_2', '4': 'pool1',
    '5':  'conv2_1', '6':  'relu2_1', '7':  'conv2_2', '8':  'relu2_2', '9': 'pool2',
    '10': 'conv3_1','11': 'relu3_1', '12': 'conv3_2','13': 'relu3_2',
    '14': 'conv3_3','15': 'relu3_3', '16': 'conv3_4','17': 'relu3_4', '18': 'pool3',
    '19': 'conv4_1','20': 'relu4_1', '21': 'conv4_2','22': 'relu4_2',
    '23': 'relu4_3','24': 'conv4_4','25': 'relu4_4', '26': 'pool4',
    '27': 'conv5_1','28': 'relu5_1', '29': 'conv5_2','30': 'relu5_2',
    '31': 'conv5_3','32': 'relu5_3', '33': 'conv5_4','34': 'relu5_4', '35': 'pool5',
}


# ── Image loading/saving ──────────────────────────────────────────────────────
def load_image(path, size=IMG_SIZE):
    img = Image.open(path).convert('RGB')
    if isinstance(size, int):
        img = img.resize((size, size), Image.LANCZOS)
    else:
        img = img.resize(size, Image.LANCZOS)
    t = transforms.ToTensor()(img).unsqueeze(0)
    return t.to(DEVICE)


def tensor_to_pil(t):
    t = t.squeeze(0).cpu().clamp(0, 1)
    return transforms.ToPILImage()(t)


def save_image(t, path):
    tensor_to_pil(t).save(path)


# ── VGG19 feature extractor ───────────────────────────────────────────────────
class VGGFeatures(nn.Module):
    """Extract features from named VGG19 layers."""
    def __init__(self, layers):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)

        self.layers    = set(layers)
        self.vgg_slice = nn.ModuleList()
        self.layer_names = []

        # Build sequential slices up to the last required layer
        last_needed = max(
            int(k) for k, v in VGG19_LAYER_MAP.items() if v in layers)
        for i, layer in enumerate(vgg.children()):
            self.vgg_slice.append(layer)
            self.layer_names.append(VGG19_LAYER_MAP.get(str(i), f'layer{i}'))
            if i >= last_needed:
                break

    def forward(self, x):
        # VGG19 ImageNet normalisation (mean/std in pixel space)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x    = (x - mean) / std

        features = {}
        for layer, name in zip(self.vgg_slice, self.layer_names):
            x = layer(x)
            if name in self.layers:
                features[name] = x
        return features


# ── Gram matrix ───────────────────────────────────────────────────────────────
def gram_matrix(feat):
    B, C, H, W = feat.shape
    f = feat.view(B, C, H * W)
    G = torch.bmm(f, f.transpose(1, 2))
    return G / (C * H * W)  # Normalized


# ── NST core ──────────────────────────────────────────────────────────────────
def run_nst(content_img, style_img, style_weight=STYLE_W,
            content_weight=CONTENT_W, n_steps=N_STEPS,
            init_img=None, style_layers=None):
    """
    Run NST optimization.
    Args:
        content_img  : (1,3,H,W) tensor
        style_img    : (1,3,H,W) tensor
        style_weight : β (style loss weight)
        content_weight: α (content loss weight)
        n_steps      : optimization steps
        init_img     : initialization tensor (for temporal consistency)
        style_layers : list of layer names to use (default = STYLE_LAYERS)
    Returns:
        stylized (1,3,H,W) tensor
    """
    if style_layers is None:
        style_layers = STYLE_LAYERS

    all_layers = style_layers + [CONTENT_LAYER]
    extractor  = VGGFeatures(all_layers).to(DEVICE)

    # Get target features
    with torch.no_grad():
        content_feats = extractor(content_img)
        style_feats   = extractor(style_img)
        style_grams   = {l: gram_matrix(style_feats[l]) for l in style_layers}
        target_content = content_feats[CONTENT_LAYER].detach()

    # Initialize generated image
    if init_img is not None:
        gen = init_img.clone().requires_grad_(True)
    else:
        gen = content_img.clone().requires_grad_(True)

    optimizer = optim.Adam([gen], lr=0.02)

    step = [0]
    losses = []

    # Run optimization with Adam
    for opt_step in range(n_steps):
        optimizer.zero_grad()

        gen.data.clamp_(0, 1)
        gen_feats = extractor(gen)

        # Content loss
        content_loss = F.mse_loss(gen_feats[CONTENT_LAYER], target_content)

        # Style loss (sum over layers)
        style_loss = 0.0
        for l in style_layers:
            gen_gram  = gram_matrix(gen_feats[l])
            style_loss += F.mse_loss(gen_gram, style_grams[l])
        style_loss /= len(style_layers)

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        optimizer.step()

        step[0] += 1
        if step[0] % 50 == 0:
            losses.append(total_loss.item())
            print(f"  Step {step[0]:4d} | loss {total_loss.item():.2f} | "
                  f"content {content_loss.item():.4f} | "
                  f"style {style_loss.item():.4f}")

    gen.data.clamp_(0, 1)
    return gen.detach()


# ── Grid: 5 content × 3 style ─────────────────────────────────────────────────
def run_grid(content_paths, style_paths, out_path=None):
    """Generate 5×3 NST grid for sanity check."""
    if out_path is None:
        out_path = OUTPUT_DIR / 'grid.png'

    n_c, n_s = len(content_paths), len(style_paths)
    fig, axes = plt.subplots(n_c, n_s, figsize=(n_s * 4, n_c * 4))
    if n_c == 1: axes = [axes]
    if n_s == 1: axes = [[ax] for ax in axes]

    for i, cp in enumerate(content_paths):
        content = load_image(cp)
        for j, sp in enumerate(style_paths):
            style = load_image(sp)
            print(f"Grid [{i},{j}]: {Path(cp).name} × {Path(sp).name}")
            result = run_nst(content, style, n_steps=N_STEPS)
            axes[i][j].imshow(tensor_to_pil(result))
            axes[i][j].set_title(f'{Path(cp).stem} × {Path(sp).stem}', fontsize=8)
            axes[i][j].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Saved grid → {out_path}")


# ── β/α ablation ──────────────────────────────────────────────────────────────
def run_beta_alpha_ablation(content_path, style_path):
    """Render same pair at 3 style weights side by side."""
    content = load_image(content_path)
    style   = load_image(style_path)
    ratios  = [1e3, 1e5, 1e7]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(tensor_to_pil(content)); axes[0].set_title('Content'); axes[0].axis('off')

    for i, sw in enumerate(ratios):
        print(f"β/α ablation: style_weight={sw:.0e}")
        result = run_nst(content, style, style_weight=sw, n_steps=N_STEPS)
        axes[i+1].imshow(tensor_to_pil(result))
        axes[i+1].set_title(f'β/α = {sw:.0e}')
        axes[i+1].axis('off')

    plt.tight_layout()
    out = OUTPUT_DIR / 'beta_alpha_ablation.png'
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Saved β/α ablation → {out}")


# ── Layer ablation ────────────────────────────────────────────────────────────
def run_layer_ablation(content_path, style_path):
    """Render using shallow-only vs deep-only style layers."""
    content = load_image(content_path)
    style   = load_image(style_path)

    shallow = ['relu1_1', 'relu2_1']
    deep    = ['relu4_1', 'relu5_1']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(tensor_to_pil(content)); axes[0].set_title('Content'); axes[0].axis('off')

    for i, (layers, label) in enumerate([(shallow, 'Shallow (relu1-2)'),
                                          (deep,    'Deep (relu4-5)')]):
        print(f"Layer ablation: {label}")
        result = run_nst(content, style, style_layers=layers, n_steps=N_STEPS)
        axes[i+1].imshow(tensor_to_pil(result))
        axes[i+1].set_title(label); axes[i+1].axis('off')

    plt.tight_layout()
    out = OUTPUT_DIR / 'layer_ablation.png'
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Saved layer ablation → {out}")


# ── Feature map visualization ─────────────────────────────────────────────────
def visualize_feature_maps(content_path, seed_image_path=None):
    """Plot 8 channels each from one shallow and one deep VGG19 layer."""
    content = load_image(content_path)
    extractor = VGGFeatures(['relu1_1', 'relu4_1']).to(DEVICE)

    with torch.no_grad():
        feats_video = extractor(content)

    images_to_plot = [('Video frame', content, feats_video)]
    if seed_image_path and Path(seed_image_path).exists():
        seed = load_image(seed_image_path)
        with torch.no_grad():
            feats_seed = extractor(seed)
        images_to_plot.append(('Seed image', seed, feats_seed))

    n_imgs = len(images_to_plot)
    fig, axes = plt.subplots(n_imgs * 2, 9, figsize=(36, 8 * n_imgs))
    if n_imgs == 1:
        axes = axes.reshape(2, 9)

    for row_base, (title, img_t, feats) in enumerate(images_to_plot):
        # Input image
        img_np = img_t.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[row_base * 2, 0].imshow(img_np)
        axes[row_base * 2, 0].set_title(f'{title}\n(input)', fontsize=7)
        axes[row_base * 2, 0].axis('off')

        for ch in range(8):
            # Shallow: relu1_1
            fm = feats['relu1_1'][0, ch].cpu().numpy()
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-6)
            axes[row_base * 2, ch + 1].imshow(fm, cmap='viridis')
            axes[row_base * 2, ch + 1].set_title(f'relu1_1 ch{ch}', fontsize=6)
            axes[row_base * 2, ch + 1].axis('off')

        axes[row_base * 2 + 1, 0].imshow(img_np)
        axes[row_base * 2 + 1, 0].axis('off')
        for ch in range(8):
            # Deep: relu4_1
            fm = feats['relu4_1'][0, ch].cpu().numpy()
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-6)
            axes[row_base * 2 + 1, ch + 1].imshow(fm, cmap='magma')
            axes[row_base * 2 + 1, ch + 1].set_title(f'relu4_1 ch{ch}', fontsize=6)
            axes[row_base * 2 + 1, ch + 1].axis('off')

    plt.suptitle('VGG19 Feature Maps — Shallow (relu1_1) vs Deep (relu4_1)')
    plt.tight_layout()
    out = OUTPUT_DIR / 'feature_maps.png'
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"Saved feature maps → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content',      type=str, default=None)
    parser.add_argument('--style',        type=str, default=None)
    parser.add_argument('--out',          type=str, default=None)
    parser.add_argument('--style_weight', type=float, default=STYLE_W)
    parser.add_argument('--n_steps',      type=int,   default=N_STEPS)
    parser.add_argument('--ablation',     action='store_true')
    parser.add_argument('--layer_ablation', action='store_true')
    parser.add_argument('--feature_maps',   action='store_true')
    parser.add_argument('--seed_img',     type=str, default=None)
    args = parser.parse_args()

    if args.ablation:
        assert args.content and args.style, "Need --content and --style for ablation"
        run_beta_alpha_ablation(args.content, args.style)

    elif args.layer_ablation:
        assert args.content and args.style
        run_layer_ablation(args.content, args.style)

    elif args.feature_maps:
        assert args.content
        visualize_feature_maps(args.content, args.seed_img)

    elif args.content and args.style:
        content = load_image(args.content)
        style   = load_image(args.style)
        result  = run_nst(content, style,
                          style_weight=args.style_weight,
                          n_steps=args.n_steps)
        out_path = args.out or str(OUTPUT_DIR / 'stylized_single.png')
        save_image(result, out_path)
        print(f"Saved → {out_path}")

    else:
        print("No action specified. Use --help for options.")
