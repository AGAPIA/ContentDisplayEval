# Texture Anomaly Detection Pipeline User Guide

This guide explains how to set up and run the two-stage texture anomaly detection pipeline, including Stage 1 and Stage 2 training and inference, as a unified process.

## Table of Contents

1. [Overview](#overview)  
2. [Prerequisites](#prerequisites)  
3. [Directory Structure](#directory-structure)  
4. [Configuration](#configuration)  
   - [Stage 1 `config.yaml`](#stage-1-configyaml)  
   - [Stage 2 `config.yaml`](#stage-2-configyaml)  
5. [Stage 1 Training](#stage-1-training)  
6. [Stage 1 Inference](#stage-1-inference)  
7. [Stage 2 Training](#stage-2-training)  
8. [Stage 2 Inference](#stage-2-inference)  
9. [Pipeline Script](#pipeline-script)  
10. [Usage](#usage)  
11. [Output Files](#output-files)  
12. [Cleanup and Troubleshooting](#cleanup-and-troubleshooting)

## Overview

The pipeline comprises two sequential stages:

- **Stage 1**: Lightweight classifier (ShuffleNetV2) producing an anomaly score in [0,1].  
- **Stage 2**: Deep feature extractor (DenseNet121) on flagged images, producing final predictions.

The orchestration script `run_full_inference.py` automates both inference stages and metadata management.

## Prerequisites

- Python 3.8+  
- PyTorch & torchvision  
- `tqdm`  
- `pyyaml`

Install dependencies:

```bash
pip install torch torchvision tqdm pyyaml
```

## Directory Structure

## Configuration

### Stage 1 `texture_stage1/config.yaml`

```yaml
train:
  image_dir: data/train_images
  metadata_path: data/train_labels.json
  batch_size: 32
  learning_rate: 0.0005
  epochs: 10
  output_dir: checkpoints/stage1

inference:
  image_dir: data/test_images
  metadata_path: data/test_metadata.json
  model_path: checkpoints/stage1/texture_stage1.pth
  output_dir: outputs/stage1
  batch_size: 32
  threshold: 0.5
```

### Stage 2 `texture_stage2/config.yaml`

```yaml
train:
  image_dir: data/stage2_train_images
  metadata_path: data/stage2_train_labels.json
  batch_size: 16
  learning_rate: 0.0003
  epochs: 8
  output_dir: checkpoints/stage2

inference:
  image_dir: data/stage2_test_images
  metadata_path: data/stage2_test_metadata.json
  model_path: checkpoints/stage2/texture_stage2.pth
  output_dir: outputs/stage2
  batch_size: 32
  threshold: 0.5
```

## Stage 1 Training

```bash
python texture_stage1/train.py --config texture_stage1/config.yaml
```
- **Input**: `data/train_images/`, `data/train_labels.json`  
- **Output**: Checkpoint at `checkpoints/stage1/texture_stage1.pth`

## Stage 1 Inference

```bash
python texture_stage1/inference.py --config texture_stage1/config.yaml
```
- **Output**: Masks and `inference_results.json` in `outputs/stage1/`

## Stage 2 Training

```bash
python texture_stage2/train.py --config texture_stage2/config.yaml
```
- **Output**: Checkpoint at `checkpoints/stage2/texture_stage2.pth`

## Stage 2 Inference

```bash
python texture_stage2/inference.py --config texture_stage2/config.yaml
```
- **Output**: Features/masks and `inference_results.json` in `outputs/stage2/`

## Pipeline Script

```bash
python run_full_inference.py --input_dir data/test_images
```

Runs Stage 1 → filters flagged images → Stage 2.

## Usage

Place your test images under `data/test_images/` and run the pipeline script.

## Output Files

- `outputs/stage1/inference_results.json`  
- `outputs/stage2/inference_results.json`

## Cleanup and Troubleshooting

- Temporary metadata files (`metadata.tmp.json`) are removed automatically.  