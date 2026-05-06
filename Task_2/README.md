# Assignment 5 — Deep Learning for Computer Vision
## Neural Style Transfer Video

---

## Overview

This assignment has two independent tasks:

| Task | What it does |
| **Task 2** | Neural Style Transfer video pipeline — trains a U-Net matting model on AISegment portraits, then applies Gatys-style NST to a video with selective compositing |

---

## What Video Do You Need for Task 2?

**Record or use any short video (10–30 seconds) where a person is the main subject**, filmed against any background. Ideal characteristics:

- A person walking slowly, standing, or doing a simple activity (reading, writing, etc.)
- Filmed from a fixed camera position (reduces background complexity)
- Indoor or outdoor — both work
- 1080p or 720p (the pipeline downsamples to 256×256 anyway)
- MP4 format preferred

**What the video is used for:**
The matting model segments the person from the background. Then NST applies an artwork style either to just the background (person stays natural), just the person (background stays natural), or the whole frame. The final output is a stylized video.

**Example:** Film yourself at your desk for 15 seconds. That is enough.

---

## What Is the Public Domain Artwork (Style Image)?

You need **1–3 style images** — these are the artworks whose visual style is transferred onto your video frames. They must be **public domain** so there are no copyright issues.

**Recommended sources:**

| Artwork | Artist | Where to get it |
|---|---|---|
| The Starry Night | Van Gogh | [WikiArt](https://www.wikiart.org/en/vincent-van-gogh/the-starry-night-1889) |
| The Great Wave | Hokusai | [Wikipedia Commons](https://commons.wikimedia.org/wiki/File:The_Great_Wave_off_Kanagawa.jpg) |
| Composition VIII | Kandinsky | [WikiArt](https://www.wikiart.org/en/wassily-kandinsky/composition-viii-1923) |
| The Persistence of Memory | Dalí | Public domain in many countries |
| Mosaic / Byzantine patterns | Various | [Wikipedia Commons](https://commons.wikimedia.org/wiki/Category:Byzantine_mosaics) |

**Download any one painting as a JPG and save it to `task2_nst_video/style/`.**

The assignment requires:
- At minimum 1 style applied to the video
- β/α weight ablation (3 different style strengths on a single frame)
- Layer ablation (shallow vs deep VGG layers)
- Feature map visualization comparing video frames vs seed images

---

## Project Directory Structure

```
Task_2/
│
│
├── data/
│   └── aisegment/                     # AISegment dataset (Task 2)
│       ├── clip_img/
│       │   └── <session>/<sub>/*.jpg
│       └── matting/
│           └── <session>/<sub>/*.png
│
├── task2_nst_video/
│   ├── config_task2.yaml              # Matting + NST + video settings
│   ├── run_task2.py                   # Master script — runs everything
│   ├── nst.py                         # NST core (VGG19, Gram matrix, ablations)
│   ├── video_pipeline.py             # Frame extraction + compositing + encoding
│   ├── matting/
│   │   ├── model.py                   # U-Net architecture
│   │   └── train.py                   # Training on AISegment
│   ├── content/                       # ← Put extracted video frames here
│   │   └── frame_001.jpg  ...
│   └── style/                         # ← Put your artwork images here
│       └── vangogh_starry_night.jpg
│
│
└── task2_outputs/                     # Created automatically by Task 2
    ├── matting_weights/
    │   └── matting_best.pt
    ├── matting_plots/
    │   └── matting_curves.png
    ├── content/                       # Auto-extracted video frames
    ├── beta_alpha_ablation.png
    ├── layer_ablation.png
    ├── feature_maps.png
    ├── grid.png
    ├── stylized_background_vangogh.mp4
    ├── stylized_subject_vangogh.mp4
    ├── stylized_full_vangogh.mp4
    ├── branded_poster.png
    └── matting_overlay.png
```

---

## Setup

### 1. Install dependencies


---


## Running Task 2 — NST Video Pipeline

### Step-by-step (recommended)

**Step 1: Train matting model**
```bash
cd task2_nst_video
python matting/train.py
```
Expected time: 2–4 hours on CPU (2500 training pairs, 20 epochs).
Target IoU ≥ 0.85.

**Step 2: Extract content frames + run ablations + process video**
```bash
python run_task2.py --video content/my_video.mp4 --style style/vangogh.jpg
```

Or if matting is already trained:
```bash
python run_task2.py --video content/my_video.mp4 --style style/vangogh.jpg --skip_train
```

### Run components individually

```bash
# NST on a single image
python nst.py --content content/frame_001.jpg --style style/vangogh.jpg

# β/α weight ablation
python nst.py --ablation --content content/frame_001.jpg --style style/vangogh.jpg

# Layer ablation (shallow vs deep)
python nst.py --layer_ablation --content content/frame_001.jpg --style style/vangogh.jpg

# Feature map visualization
python nst.py --feature_maps --content content/frame_001.jpg

# Extract frames only
python video_pipeline.py --video content/my_video.mp4 --style style/vangogh.jpg --extract_frames

# Full pipeline, background-stylized variant only
python video_pipeline.py --video content/my_video.mp4 --style style/vangogh.jpg --variant background
```

---

## Deliverables Summary


### Task 2

| Deliverable | File |
|---|---|
| U-Net matting architecture | `task2_nst_video/matting/model.py` |
| Matting training code | `task2_nst_video/matting/train.py` |
| Matting training curves + IoU | `task2_outputs/matting_plots/matting_curves.png` |
| Matting overlay visualization | `task2_outputs/matting_overlay.png` |
| NST implementation | `task2_nst_video/nst.py` |
| β/α weight ablation image | `task2_outputs/beta_alpha_ablation.png` |
| Layer ablation image | `task2_outputs/layer_ablation.png` |
| VGG19 feature map visualization | `task2_outputs/feature_maps.png` |
| Stylized video (background) | `task2_outputs/stylized_background_*.mp4` |
| Stylized video (subject) | `task2_outputs/stylized_subject_*.mp4` |
| Stylized video (full frame) | `task2_outputs/stylized_full_*.mp4` |
| Branded poster | `task2_outputs/branded_poster.png` |

---

## Key Design Decisions


### Task 2

**Temporal consistency:**  
Each frame's NST is initialized from the previous stylized frame (`init_img=prev_stylized`). This reduces flickering and halves convergence time because the frame-to-frame content change is small.

**CPU feasibility:**  
- Video sampled at 8 FPS → ~80 frames for a 10s clip
- NST: 150 L-BFGS steps at 256×256 — ~90s per frame on a modern CPU
- Matting: U-Net inference is fast (under 1s per frame)
- Matting training: 2500 samples × 20 epochs ≈ 2–4 hours

---

## Troubleshooting

**CUDA / MPS not available:**  
All code falls back to CPU automatically. Training will be slower but functionally identical.

**AISegment dataset structure:**  
The dataset from Kaggle has sessions like `1803151818/` containing subdirectories. The `find_pairs()` function in `matting/train.py` walks this structure automatically. Just point `aisegment_root` to the folder containing `clip_img/` and `matting/`.

**VGG19 download:**  
`torchvision` downloads VGG19 weights (~550MB) automatically on first run. Requires internet access.

**Video codec errors:**  
The pipeline uses `mp4v` codec. If your player cannot open the output, convert with:
```bash
ffmpeg -i stylized_background.mp4 -c:v libx264 stylized_background_x264.mp4
```
