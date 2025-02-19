# Edge Detection Model

A PyTorch-based implementation of an edge detection model using Mask R-CNN with ResNet-50 backbone for detecting and extracting edges from images, with export capabilities to Rhino 3DM format.

## Overview

This model uses instance segmentation to detect edges in images and can:
- Train on custom datasets in COCO format
- Evaluate model performance
- Run inference on test images
- Export detected edges to Rhino 3DM format
- Visualize results with bounding boxes and labels

## Requirements

```bash
pip install torch torchvision opencv-python pillow numpy rhino3dm
```

## Dataset Structure

```
project_root/
├── train/
│   └── _annotations.coco.json
├── valid/
│   └── _annotations.coco.json
├── test/
│   └── _annotations.coco.json
├── model_checkpoints/
└── output/
```

## Usage

### Training

Train the model for a specified number of epochs:
```bash
python edge_detector.py --train 10
```

### Evaluation

Evaluate model performance on the test set:
```bash
python edge_detector.py --evaluate
```

### Inference

Run inference on a test image:
```bash
python edge_detector.py --inference
```

Run inference on a random test image:
```bash
python edge_detector.py --inference --random
```

### Combined Operations

You can combine operations (they will execute in this order):
```bash
python edge_detector.py --train 10 --evaluate --inference
```

## Model Architecture

- Base: Mask R-CNN
- Backbone: ResNet-50
- FPN: Feature Pyramid Network
- Customized for edge detection task

## Outputs

The model generates:
- Model checkpoints (`model_checkpoints/`)
- Evaluation metrics (`output/evaluation_results.json`)
- Visualizations with bounding boxes (`output/*_with_labels.jpg`)
- Rhino 3DM files with extracted edges (`output/detected_edges_*.3dm`)

## Metrics

Evaluation produces the following metrics:
- Mean IoU (Intersection over Union)
- Precision
- Recall
- F1-Score

## Features

- Automatic checkpoint saving and loading
- CUDA support for GPU acceleration
- Custom data loading with COCO format support
- Edge extraction with Douglas-Peucker simplification
- Rhino 3DM export functionality