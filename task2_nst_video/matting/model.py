"""
matting/model.py — Lightweight U-Net for human alpha matting.

Architecture:
  Encoder: 4 downsampling stages (Conv→BN→ReLUx2 + MaxPool)
  Bottleneck: Conv block
  Decoder: 4 upsampling stages with skip connections from encoder
  Output: 1xHxW sigmoid → alpha matte ∈ [0, 1]

Input:  RGB frame resized to 256x256
Output: alpha matte, 256x256, float32

Training target: IoU ≥ 0.85 on AISegment validation split.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """MaxPool then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Bilinear upsample then DoubleConv with skip connection."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial sizes mismatch (edge case for non-power-of-2 inputs)
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        x  = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class MattingUNet(nn.Module):
    """
    Lightweight U-Net for binary alpha matting.
    Encoder channels: 3→32→64→128→256
    Bottleneck:       256→512
    Decoder:          mirrors encoder with skip connections
    Output head:      Conv(64→1) + Sigmoid
    """
    def __init__(self, base_ch=32):
        super().__init__()
        b = base_ch  # 32

        # Encoder
        self.enc1 = DoubleConv(3, b)       # 256→256
        self.enc2 = Down(b,    b*2)        # 256→128
        self.enc3 = Down(b*2,  b*4)        # 128→64
        self.enc4 = Down(b*4,  b*8)        # 64→32

        # Bottleneck
        self.bottle = Down(b*8, b*16)      # 32→16, ch=512

        # Decoder
        self.dec4 = Up(b*16, b*8,  b*8)   # 16→32
        self.dec3 = Up(b*8,  b*4,  b*4)   # 32→64
        self.dec2 = Up(b*4,  b*2,  b*2)   # 64→128
        self.dec1 = Up(b*2,  b,    b)      # 128→256

        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bt = self.bottle(e4)
        d4 = self.dec4(bt, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return torch.sigmoid(self.out(d1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = MattingUNet()
    x     = torch.randn(2, 3, 256, 256)
    out   = model(x)
    print(f"Output shape: {out.shape}")   # (2, 1, 256, 256)
    print(f"Parameters:   {model.count_params():,}")
