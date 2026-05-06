# Assignment 5 — Task 2: Neural Style Transfer Video

## Overview

This Task 2 project trains a U-Net human matting model on the AISegment dataset and uses Gatys-style Neural Style Transfer (NST) to produce stylized video variants:

- **background stylized** (subject natural)
- **subject stylized** (background natural)
- **full frame stylized** (baseline)

The pipeline is designed to run on CPU, with reduced frame rate and 256×256 processing.

---

## Contents

- `config_task2.yaml` — dataset, matting, NST, and video settings
- `matting/model.py` — lightweight U-Net alpha matting architecture
- `matting/train.py` — AISegment training pipeline
- `nst.py` — Neural Style Transfer implementation with Gram matrices
- `video_pipeline.py` — frame extraction, matting, compositing, encoding
- `content/` — your input video and auto-extracted frames
- `style/` — public domain artwork images
- `task2_outputs/` — generated outputs, weights, and plots

---

## A — Setup (Conda / CPU)

### Create the environment

```bash
conda create -n assignment5 python=3.11 -y
conda activate assignment5
```

### Install required Python packages

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python pyyaml matplotlib pillow
```

> If the above PyTorch command fails, use the latest CPU package line from https://pytorch.org/.

---

## B — Dataset Setup

### Download AISegment dataset

Download the AISegment human matting subset from Kaggle or your source.

### Place the dataset in the repository

The code expects:

- `Task_2/data/clip_img/`
- `Task_2/data/matting/`

Each folder should contain the AISegment session directories, e.g.:

- `Task_2/data/clip_img/1803151818/.../*.jpg`
- `Task_2/data/matting/1803151818/.../*.png`

If your dataset archive contains `clip_img/` and `matting/`, place them directly under `Task_2/data/`.

---

## C — Folder Preparation

### Video input

Put your recorded video anywhere, for example:

- `Task_2/content/my_video.mp4`

This video is used for NST and compositing only. It is **not** used for matting training.

### Style images

Place one or more public-domain style images in:

- `Task_2/style/`

Example:

- `Task_2/style/vangogh.jpg`

---

## D — Run Instructions

### 1. Train the matting model

```bash
cd e:\CV\Assignment5\Task_2
python matting/train.py
```

This trains the U-Net on the AISegment subset and saves the best weights to:

- `Task_2/task2_outputs/matting_weights/matting_best.pt`

It also writes:

- `Task_2/task2_outputs/matting_plots/matting_curves.png`
- `Task_2/task2_outputs/matting_overlay.png`

### 2. Run the full pipeline

```bash
cd e:\CV\Assignment5\Task_2
python run_task2.py --video content/my_video.mp4 --style style/vangogh.jpg
```

If you have multiple style images in `Task_2/style/`, pass the folder instead:

```bash
python run_task2.py --video content/my_video.mp4 --style style/
```

If matting is already trained, skip training with:

```bash
python run_task2.py --video content/my_video.mp4 --style style/vangogh.jpg --skip_train
```

### 3. Run individual components

```bash
cd e:\CV\Assignment5\Task_2
python nst.py --content content/frame_001.jpg --style style/vangogh.jpg
python nst.py --ablation --content content/frame_001.jpg --style style/vangogh.jpg
python nst.py --layer_ablation --content content/frame_001.jpg --style style/vangogh.jpg
python nst.py --feature_maps --content content/frame_001.jpg --seed_img content/frame_002.jpg
python video_pipeline.py --video content/my_video.mp4 --style style/vangogh.jpg --extract_frames
python video_pipeline.py --video content/my_video.mp4 --style style/vangogh.jpg --variant background
```

---

## E — Expected Outputs

The pipeline writes outputs into `Task_2/task2_outputs/`.

Generated files:

- `matting_weights/matting_best.pt`
- `matting_plots/matting_curves.png`
- `matting_overlay.png`
- `content/frame_001.jpg`, ..., `content/frame_005.jpg`
- `beta_alpha_ablation.png`
- `layer_ablation.png`
- `feature_maps.png`
- `grid.png`
- `stylized_background_<style>.mp4`
- `stylized_subject_<style>.mp4`
- `stylized_full_<style>.mp4`
- `branded_poster.png`

---

## Notes

- The video is used only for testing/compositing, not for training.
- The AISegment dataset is used only by `matting/train.py`.
- All code falls back to CPU automatically when CUDA is unavailable.
- The model saves `matting_best.pt` for later reuse.

## Troubleshooting

- If the video is not found, verify the `--video` path is correct relative to `Task_2/`.
- If style image is not found, verify the `--style` path is correct.
- If video codec issues occur, convert the output MP4 with `ffmpeg`.


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
