# Mesh Anomaly Detection Pipeline User Guide

This guide covers Stage 1 (U-Net) and Stage 2 (DeepLabV3+) training and inference, and a unified orchestration script.

## Table of Contents

1. [Overview](#overview)  
2. [Prerequisites](#prerequisites)  
3. [Directory Structure](#directory-structure)  
4. [Configuration](#configuration)  
5. [Stage 1 Training](#stage-1-training)  
6. [Stage 1 Inference](#stage-1-inference)  
7. [Stage 2 Training](#stage-2-training)  
8. [Stage 2 Inference](#stage-2-inference)  
9. [Unified Pipeline Script](#unified-pipeline-script)  
10. [Usage](#usage)  
11. [Output Files](#output-files)  
12. [Cleanup and Troubleshooting](#cleanup-and-troubleshooting)

## Overview

- **Stage 1**: U-Net segmentation on 256×256 frames producing anomaly masks.  
- **Stage 2**: DeepLabV3+ refinement on flagged frames (area ≥ threshold).  

A wrapper `run_full_mesh_pipeline.py` automates both stages.

## Prerequisites

- Python 3.8+  
- PyTorch & torchvision  
- `tqdm`, `pyyaml`, `pillow`, `numpy`

```bash
pip install torch torchvision tqdm pyyaml pillow numpy
```

## Configuration

### Stage 1 `mesh_stage1/config.yaml`

```yaml
train:
  image_dir: data/mesh_train_images
  mask_dir:  data/mesh_train_masks
  metadata_path: data/mesh_train_metadata.json
  batch_size: 16
  learning_rate: 0.0002
  epochs: 20
  output_dir: checkpoints/mesh_stage1

inference:
  image_dir: data/mesh_test_images
  metadata_path: data/mesh_test_metadata.json
  model_path: checkpoints/mesh_stage1/mesh_stage1.pth
  output_dir: outputs/mesh_stage1
  batch_size: 8
  threshold: 0.3
```

### Stage 2 `mesh_stage2/config.yaml`

```yaml
train:
  image_dir: data/mesh2_train_images
  mask_dir:  data/mesh2_train_masks
  metadata_path: data/mesh2_train_metadata.json
  batch_size: 8
  learning_rate: 0.0001
  epochs: 15
  output_dir: checkpoints/mesh_stage2

inference:
  image_dir: mesh_stage2/input_images
  metadata_path: mesh_stage2/metadata.tmp.json
  model_path: checkpoints/mesh_stage2/mesh_stage2.pth
  output_dir: outputs/mesh_stage2
  batch_size: 4
  threshold: 0.3
```

## Stage 1 Training

```bash
python mesh_stage1/train.py --config mesh_stage1/config.yaml
```

## Stage 1 Inference

```bash
python mesh_stage1/inference.py --config mesh_stage1/config.yaml
```

## Stage 2 Training

```bash
python mesh_stage2/train.py --config mesh_stage2/config.yaml
```

## Stage 2 Inference

```bash
python mesh_stage2/inference.py --config mesh_stage2/config.yaml
```

## Unified Pipeline Script

```bash
python run_full_mesh_pipeline.py --input_dir data/mesh_test_images --area_thresh 0.02
```

## Usage

Place your test images in `data/mesh_test_images/` and run the unified script.

## Output Files

- `outputs/mesh_stage1/inference_results.json`  
- `outputs/mesh_stage2/inference_results.json`
