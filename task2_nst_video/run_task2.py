"""
run_task2.py — Master script for Task 2.

Runs in order:
  Step 1: Train matting model on AISegment subset
  Step 2: Extract content frames from your video
  Step 3: Run NST ablations (β/α + layer) on sample frames
  Step 4: Generate VGG19 feature map visualizations
  Step 5: Run full video pipeline (all 3 compositing variants)

Usage:
  python run_task2.py --video content/my_video.mp4 --style style/vangogh.jpg
  python run_task2.py --video content/my_video.mp4 --style style/vangogh.jpg --skip_train
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from matting.train   import train  as train_matting
from video_pipeline  import run_pipeline, extract_content_frames
from nst             import (run_grid, run_beta_alpha_ablation,
                              run_layer_ablation, visualize_feature_maps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',       required=True, help='Path to your input video')
    parser.add_argument('--style',       required=True,
                        help='Path to a style image or style directory')
    parser.add_argument('--skip_train',  action='store_true',
                        help='Skip matting training (use if weights already exist)')
    parser.add_argument('--variant',     default='all',
                        choices=['background', 'subject', 'full', 'all'])
    args = parser.parse_args()

    video_path = Path(args.video)
    style_path = Path(args.style)
    assert video_path.exists(), f"Video not found: {video_path}"
    assert style_path.exists(), f"Style path not found: {style_path}"

    if style_path.is_dir():
        style_paths = sorted([p for p in style_path.iterdir()
                              if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        assert style_paths, f"No style images found in directory: {style_path}"
        style_for_video = style_paths[0]
    else:
        style_paths = [style_path]
        style_for_video = style_path

    print("\n" + "="*60)
    print("TASK 2 PIPELINE — Assignment 5")
    print("="*60)

    # ── Step 1: Train matting model ───────────────────────────────────────────
    if not args.skip_train:
        print("\n[Step 1] Training matting model...")
        test_iou = train_matting()
        print(f"Matting training complete. Test IoU: {test_iou:.4f}")
    else:
        print("\n[Step 1] Skipping matting training (--skip_train set).")

    # ── Step 2: Extract content frames ────────────────────────────────────────
    print("\n[Step 2] Extracting content frames from video...")
    content_frames = extract_content_frames(str(video_path), n=5)
    print(f"Extracted {len(content_frames)} content frames.")

    # ── Step 3: NST ablations ─────────────────────────────────────────────────
    if content_frames:
        sample_content = content_frames[len(content_frames) // 2]

        if len(style_paths) >= 3:
            print("\n[Step 3] Generating NST style grid (5×3)...")
            run_grid(content_frames, style_paths[:3])
        else:
            print("\n[Step 3] Skipping 5×3 style grid because fewer than 3 style images were provided.")

        print("\n[Step 4] Running β/α ablation...")
        run_beta_alpha_ablation(sample_content, str(style_for_video))

        print("\n[Step 5] Running layer ablation...")
        run_layer_ablation(sample_content, str(style_for_video))

    # ── Step 4: Feature map visualization ────────────────────────────────────
    if content_frames:
        print("\n[Step 4] Generating VGG19 feature map visualization...")
        visualize_feature_maps(content_frames[0])

    # ── Step 5: Full video pipeline ───────────────────────────────────────────
    print(f"\n[Step 5] Running full video pipeline (variant={args.variant})...")
    run_pipeline(str(video_path), str(style_for_video), variant=args.variant)

    print("\n" + "="*60)
    print("Task 2 complete. Outputs in task2_outputs/")
    print("="*60)


if __name__ == '__main__':
    main()
