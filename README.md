# Edge Detection Model for Building Facades

A PyTorch-based implementation of an edge detection model using Mask R-CNN with ResNet-50 backbone for detecting and extracting edges from building facades, with export capabilities to Rhino 3DM format.

## Overview

This model uses instance segmentation to detect building elements and can:
- Train on custom datasets in COCO format
- Evaluate model performance
- Run inference on test images
- Export detected edges to Rhino 3DM format
- Visualize results with both bounding boxes and segmentation masks

## Get Started
We recommend creating a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```
Then install requirements:
```bash
pip install -r requirements.txt
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

## Categories
The model detects the following building elements:
- Facade
- Balcony-fence
- Car
- Fence
- Non-building-infrastructure
- Shop
- Street
- Traffic-infrastructure
- Vegetation
- Window

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
- Customized for facade element detection

## Outputs

The model generates:
- Model checkpoints (`model_checkpoints/model_checkpoint_epoch_*.pth`)
- Evaluation metrics (`output/evaluation_results.json`)
- Bounding box visualizations (`output/*_with_boxes.jpg`)
- Segmentation mask visualizations (`output/*_with_masks.jpg`)
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
- Dual visualization outputs (boxes and masks)
- Category-based color coding
- Rhino 3DM export functionality

## Hardware Acceleration

The model supports the following hardware acceleration:
- NVIDIA CUDA GPUs
- Apple Metal (MPS) for M1/M2 Macs
- CPU (fallback)

### Known Issues with Apple Metal

If you encounter MPS-related errors, you can:
1. Force CPU usage:
```bash
python edge_detector.py --train 10 --force-cpu
```

2. Try reducing batch size:
```python
train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
```

3. Check your PyTorch version supports MPS:
```bash
pip install --upgrade torch
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- Pillow
- NumPy
- rhino3dm