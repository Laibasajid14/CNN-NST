"""
video_pipeline.py — End-to-end stylized video pipeline.

Steps:
  1. Decode input video into frames (OpenCV)
  2. For each frame: run matting model → alpha_t
  3. Run NST → stylized frame S_t
  4. Composite:
       background-stylized: O_t = alpha_t * F_t + (1-alpha_t) * S_t
       subject-stylized:    O_t = alpha_t * S_t + (1-alpha_t) * F_t
       full-frame:          O_t = S_t
  5. Re-encode to MP4

Usage:
  python video_pipeline.py --video input_video.mp4 --style style/vangogh.jpg
  python video_pipeline.py --video input_video.mp4 --style style/vangogh.jpg --variant all
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent / 'matting'))
from model import MattingUNet
from nst   import run_nst, load_image, tensor_to_pil, VGGFeatures, STYLE_LAYERS, CONTENT_LAYER

CONFIG_PATH = Path(__file__).parent / 'config_task2.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

ROOT = Path(__file__).parent
TARGET_FPS = cfg['video']['target_fps']        # 5-8 FPS for CPU feasibility
FRAME_SIZE = cfg['video']['frame_size']        # 256
NST_STEPS  = cfg['nst']['n_steps']            # 150
STYLE_W    = cfg['nst']['style_weight']
MATTING_WEIGHTS = (ROOT / cfg['paths']['matting_weights'] / 'matting_best.pt').resolve()
OUTPUT_DIR      = (ROOT / cfg['paths']['task2_outputs']).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NORMALIZE   = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
DENORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DENORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def frame_to_tensor(frame_bgr, size=FRAME_SIZE):
    """BGR uint8 numpy → normalized (1,3,H,W) tensor on DEVICE."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size))
    t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return NORMALIZE(t).unsqueeze(0).to(DEVICE)


def tensor_to_bgr(t):
    """(1,3,H,W) image tensor in [0,1] → BGR uint8 numpy."""
    t = t.squeeze(0).cpu()
    t = t.clamp(0, 1)
    rgb = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def unnorm_tensor(t):
    """(1,3,H,W) normalized → (1,3,H,W) in [0,1]."""
    t = t.clone().cpu()
    t = t * DENORM_STD.unsqueeze(0) + DENORM_MEAN.unsqueeze(0)
    return t.clamp(0, 1)


# ── Load matting model ────────────────────────────────────────────────────────
def load_matting_model():
    model = MattingUNet(base_ch=32).to(DEVICE)
    if not MATTING_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Matting weights not found at {MATTING_WEIGHTS}. "
            "Run matting/train.py first.")
    model.load_state_dict(torch.load(MATTING_WEIGHTS, map_location=DEVICE))
    model.eval()
    print(f"Loaded matting model from {MATTING_WEIGHTS}")
    return model


def predict_alpha(model, frame_tensor):
    """Run matting model on a single frame tensor. Returns (H,W) numpy alpha."""
    with torch.no_grad():
        alpha = model(frame_tensor)  # (1,1,H,W)
    return alpha.squeeze().cpu().numpy()  # (H,W) in [0,1]


# ── Composite ────────────────────────────────────────────────────────────────
def composite(original_frame, stylized_frame, alpha, variant='background'):
    """
    Composite original and stylized frames using alpha matte.
    All inputs are (H,W,3) uint8 numpy BGR.
    variant: 'background' | 'subject'
    """
    orig = original_frame.astype(np.float32) / 255.0
    styl = stylized_frame.astype(np.float32) / 255.0
    a    = alpha[:, :, None]  # (H,W,1)

    if variant == 'background':
        # Keep subject natural, stylize background
        out = a * orig + (1 - a) * styl
    else:
        # Stylize subject, keep background natural
        out = a * styl + (1 - a) * orig

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# ── Extract frames ────────────────────────────────────────────────────────────
def extract_frames(video_path, target_fps=TARGET_FPS, max_frames=300):
    """Extract frames at reduced FPS. Returns list of BGR numpy arrays."""
    cap      = cv2.VideoCapture(str(video_path))
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(src_fps / target_fps)))
    frames, idx = [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        idx += 1

    cap.release()
    print(f"Extracted {len(frames)} frames (src {src_fps:.1f}fps → target {target_fps}fps)")
    return frames, target_fps


# ── Encode video ──────────────────────────────────────────────────────────────
def encode_video(frames, out_path, fps=TARGET_FPS):
    if not frames:
        print("No frames to encode.")
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"Encoded {len(frames)} frames → {out_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(video_path, style_path, variant='all'):
    """
    Run the full pipeline.
    variant: 'background' | 'subject' | 'full' | 'all'
    """
    print(f"\n{'='*60}")
    print(f"Video Pipeline: {video_path}")
    print(f"Style:          {style_path}")
    print(f"Variant(s):     {variant}")
    print(f"{'='*60}")

    # 1. Load matting model
    matting_model = load_matting_model()

    # 2. Load style image
    style_img = load_image(str(style_path), size=FRAME_SIZE)

    # 3. Extract frames
    frames, fps = extract_frames(video_path)
    if not frames:
        raise ValueError("No frames extracted from video.")

    n_frames  = len(frames)
    style_key = Path(style_path).stem

    # Output frame lists
    bg_frames   = []
    subj_frames = []
    full_frames = []

    prev_stylized = None  # for temporal consistency

    for i, frame_bgr in enumerate(frames):
        t0 = time.time()
        print(f"\nFrame {i+1}/{n_frames}")

        # 4a. Convert frame to tensor
        frame_t = frame_to_tensor(frame_bgr)

        # 4b. Get alpha matte
        alpha = predict_alpha(matting_model, frame_t)  # (H,W) [0,1]

        # 4c. NST — initialize from previous stylized frame for temporal consistency
        frame_unnorm = unnorm_tensor(frame_t)
        result_t     = run_nst(
            frame_unnorm.to(DEVICE), style_img,
            style_weight=STYLE_W, n_steps=NST_STEPS,
            init_img=prev_stylized)
        prev_stylized = result_t.clone()

        # Convert stylized tensor to BGR
        stylized_bgr = tensor_to_bgr(result_t)
        stylized_bgr = cv2.resize(stylized_bgr, (FRAME_SIZE, FRAME_SIZE))

        # 4d. Composite
        if variant in ('background', 'all'):
            bg_frames.append(composite(frame_bgr, stylized_bgr, alpha, 'background'))

        if variant in ('subject', 'all'):
            subj_frames.append(composite(frame_bgr, stylized_bgr, alpha, 'subject'))

        if variant in ('full', 'all'):
            full_frames.append(stylized_bgr)

        print(f"  Done in {time.time()-t0:.1f}s")

    # 5. Encode videos
    if bg_frames:
        encode_video(bg_frames,
                     OUTPUT_DIR / f'stylized_background_{style_key}.mp4', fps)

    if subj_frames:
        encode_video(subj_frames,
                     OUTPUT_DIR / f'stylized_subject_{style_key}.mp4', fps)

    if full_frames:
        encode_video(full_frames,
                     OUTPUT_DIR / f'stylized_full_{style_key}.mp4', fps)

    print("\nPipeline complete.")

    # 6. Generate branded poster (best frame from background-stylized)
    if bg_frames:
        best_frame = bg_frames[len(bg_frames) // 2]
        poster     = cv2.resize(best_frame, (1024, 1024))
        cv2.imwrite(str(OUTPUT_DIR / 'branded_poster.png'), poster)
        print(f"Saved branded poster → {OUTPUT_DIR / 'branded_poster.png'}")


# ── Extract content frames for NST grid ──────────────────────────────────────
def extract_content_frames(video_path, n=5, out_dir=None):
    """Save N evenly-spaced frames to content/ folder."""
    if out_dir is None:
        out_dir = Path(__file__).parent / 'content'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap    = cv2.VideoCapture(str(video_path))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs   = [int(total * i / (n - 1)) for i in range(n)]
    saved  = []

    for j, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            path  = out_dir / f'frame_{j+1:03d}.jpg'
            cv2.imwrite(str(path), frame)
            saved.append(str(path))
            print(f"Saved content frame → {path}")

    cap.release()
    return saved


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',   required=True,  help='Path to input_video.mp4')
    parser.add_argument('--style',   required=True,  help='Path to style image')
    parser.add_argument('--variant', default='all',
                        choices=['background', 'subject', 'full', 'all'])
    parser.add_argument('--extract_frames', action='store_true',
                        help='Only extract 5 content frames and exit')
    args = parser.parse_args()

    if args.extract_frames:
        extract_content_frames(args.video)
    else:
        run_pipeline(args.video, args.style, args.variant)
