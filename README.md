# Assignment 5: Computer Vision Tasks

This repository contains two computer vision tasks for Assignment 5:

- **Task 1**: CNN-based seed counting from scratch
- **Task 2**: Neural Style Transfer on video with human matting

## Environment Setup

### Prerequisites
- Python 3.11
- Conda (recommended for environment management)
- Modern CPU with at least 8GB RAM (all tasks run on CPU)

### Installation

1. Create the conda environment:
```bash
conda create -n assignment5 python=3.11 -y
conda activate assignment5
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda env create -f environment.yml
conda activate assignment5
```

## Task 1: CNN Seed Counting

### Overview
Implements CNN models trained from scratch for seed counting, compared against classical methods from Assignments 2 and 3.

### Running Task 1

1. Navigate to task1_cnn directory:
```bash
cd task1_cnn
```

2. Run the training script:
```bash
python train.py
```

This will train multiple models (Model A with Adam/SGD, Model B) and save results to `cnn_outputs/`.

### Expected Outputs
- Trained model weights in `cnn_outputs/weights/`
- Training logs in `cnn_outputs/logs/`
- Performance comparison in `cnn_outputs/comparison_table.csv`
- Plots in `cnn_outputs/plots/`

## Task 2: Neural Style Transfer Video

### Overview
Trains a U-Net for human matting and applies Neural Style Transfer to video frames.

### Running Task 2

1. Download the AISegment dataset and place it in `task2_nst_video/data/matting/`

2. Navigate to task2_nst_video directory:
```bash
cd task2_nst_video
```

3. Train the matting model:
```bash
python matting/train.py
```

4. Run the video pipeline:
```bash
python run_task2.py
```

### Expected Outputs
- Matting model weights in `task2_outputs/matting_weights/`
- Stylized video outputs in `task2_outputs/`
- Training plots in `task2_outputs/matting_plots/`

## Hardware Used
- CPU: Intel Core i7-10700K or equivalent
- RAM: 16GB
- OS: Windows 10/11
- All processing is CPU-based, no GPU required

## Notes
- Task 1 uses preprocessed images from `intermediate_outputs/preprocessed_images/filtered/`
- Task 2 requires manual download of AISegment dataset
- Baseline verification codes are included in `submission_metadata.json`</content>
<parameter name="filePath">e:\CV\Assignment5\README.md