"""
models.py — CNN architectures for Task 1.

Model A — Baseline CNN
  3 conv blocks (Conv2D → BN → ReLU → MaxPool)
  Filter progression: 32 → 64 → 128
  Global Average Pooling → Dense → output
  ≤ 1.5M parameters
  Two output heads switchable: regression (1 neuron) or classification (15 classes)

Model B — Deeper / Regularized CNN
  4 conv blocks with Dropout(0.3), L2 weight decay (via optimizer), optional residual
  Filter progression: 32 → 64 → 128 → 256
  Same dual-head design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared building block ─────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Conv2D → BatchNorm → ReLU → (optional Dropout) → MaxPool."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.drop(F.relu(self.bn(self.conv(x)))))


# ── Residual block (used in Model B) ─────────────────────────────────────────
class ResBlock(nn.Module):
    """Two conv layers with a skip connection (identity or 1×1 projection)."""
    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.drop  = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop(out)
        return F.relu(out + residual)


# ── Model A — Baseline CNN ────────────────────────────────────────────────────
class ModelA(nn.Module):
    """
    3 conv blocks (Conv → BN → ReLU → MaxPool), filters 32→64→128.
    Global Average Pooling → FC(128) → output head.
    Parameters ≤ 1.5M.
    """
    def __init__(self, mode='regression', num_classes=15, img_size=224):
        super().__init__()
        self.mode = mode

        self.block1 = ConvBlock(3,  32)     # 224→112
        self.block2 = ConvBlock(32, 64)     # 112→56
        self.block3 = ConvBlock(64, 128)    # 56→28

        self.gap = nn.AdaptiveAvgPool2d(1)  # (B, 128, 1, 1)
        self.fc1 = nn.Linear(128, 128)

        if mode == 'regression':
            self.head = nn.Linear(128, 1)
        else:
            self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        x = F.relu(self.fc1(x))
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Model B — Deeper / Regularized CNN ───────────────────────────────────────
class ModelB(nn.Module):
    """
    4 conv blocks, filters 32→64→128→256, dropout 0.3, residual connection
    at the 256-channel level.
    L2 weight decay applied via optimizer weight_decay parameter.
    """
    def __init__(self, mode='regression', num_classes=15, dropout=0.3):
        super().__init__()
        self.mode = mode

        self.block1 = ConvBlock(3,   32,  dropout=0.0)   # 224→112
        self.block2 = ConvBlock(32,  64,  dropout=0.2)   # 112→56
        self.block3 = ConvBlock(64,  128, dropout=0.3)   # 56→28
        self.block4 = ConvBlock(128, 256, dropout=0.0)   # 28→14

        # Residual block at 256 channels (spatial 14×14)
        self.res = ResBlock(256, dropout=dropout)

        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(256, 256)
        self.fc2  = nn.Linear(256, 128)

        if mode == 'regression':
            self.head = nn.Linear(128, 1)
        else:
            self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.res(x)
        x = self.gap(x).flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(model_name, mode='regression', num_classes=15):
    """Factory: 'model_a' or 'model_b'."""
    if model_name == 'model_a':
        model = ModelA(mode=mode, num_classes=num_classes)
    elif model_name == 'model_b':
        model = ModelB(mode=mode, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    print(f"{model_name} parameters: {model.count_params():,}")
    return model


if __name__ == '__main__':
    # Quick sanity check
    for name in ['model_a', 'model_b']:
        for mode in ['regression', 'classification']:
            m = build_model(name, mode=mode)
            x = torch.randn(2, 3, 224, 224)
            out = m(x)
            print(f"{name} | {mode} | output shape: {out.shape}")
