# Edge Detection Model for Building Facades

A PyTorch-based implementation of an edge detection model using instance segmentation. By default, it uses Mask R-CNN with a ResNet-50 backbone for detecting and extracting edges from building facades and now includes performance graphing capabilities as well as Rhino 3DM export for segmentation masks.

## Overview

This project supports:
- **Training:** Train the model on custom datasets in COCO format.
- **Evaluation:** Evaluate the model using bounding box metrics.
- **Inference:** Run inference on test images with visualizations of predictions (bounding boxes and, if available, segmentation masks). If masks are present, edges are extracted and exported as Rhino 3DM files.
- **Graphing:** Generate graphs of model performance (IoU, Precision, Recall, F1-Score) across selected epochs using new command-line options.

The following outputs are generated:
- **Checkpoints:** Saved in a per-model subfolder (e.g., `maskrcnn_resnet50_fpn_checkpoints`).
- **Evaluation Metrics:** Saved as JSON files in the `output/` folder.
- **Bounding Box Visualizations:** Output images in the `output/` folder.
- **Segmentation Mask Visualizations:** If available, saved in the `output/` folder.
- **Performance Graphs:** Generated and saved as PNG images into `output/`.
- **Rhino 3DM Exports:** Created from segmentation outputs if masks are available.

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
├── output/
└── <model_name>_checkpoints/    # Checkpoints are saved per-model (see below)
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

Train the model for a specified number of epochs using the default model (Mask R-CNN with ResNet-50):
```bash
python edge_detector.py --train 10
```

You can also specify a different pretrained model with the `--model` parameter. Valid options are:
- maskrcnn_resnet50_fpn
- fasterrcnn_resnet50_fpn
- fasterrcnn_resnet50_fpn_v2
- fasterrcnn_mobilenet_v3_large_fpn
- retinanet_resnet50_fpn
- fcos_resnet50_fpn
- keypointrcnn_resnet50_fpn

For example, to use Faster R-CNN with ResNet-50:
```bash
python edge_detector.py --train 10 --model fasterrcnn_resnet50_fpn
```

The checkpoints for each model will be saved in a dedicated subfolder named `<model_name>_checkpoints` (e.g., `maskrcnn_resnet50_fpn_checkpoints`).

### Evaluation

Evaluate model performance on the test set:
```bash
python edge_detector.py --evaluate
```

### Inference

Run inference on a given test image:
```bash
python edge_detector.py --inference --file path/to/test_image.jpg
```

Or run inference on a random test image:
```bash
python edge_detector.py --inference --random
```
During inference, bounding box predictions are visualized, and if segmentation masks are available, they are overlaid on the image and processed to export a Rhino 3DM file with extracted edge curves.

### Graphing Performance

Generate performance graphs (IoU, Precision, Recall, F1-Score) over selected epochs:
```bash
python edge_detector.py --graph-epochs 1 2 3 4 5
```
or using a range:
```bash
python edge_detector.py --graph-range 1-10
```
The performance graphs will be saved in the `output/` folder.

### Combined Operations

You can combine operations (they will execute in this order):
```bash
python edge_detector.py --train 10 --evaluate --inference --graph-range 1-10
```

## Model Architecture

- **Default Model:** Mask R-CNN  
- **Backbone:** ResNet-50 (when using Mask R-CNN)  
- **Alternate Models:** You can choose any of the following:
  - Faster R-CNN with ResNet-50
  - Faster R-CNN with ResNet-50 (v2)
  - Faster R-CNN with MobileNet V3 Large FPN
  - Retinanet with ResNet-50 FPN
  - FCOS with ResNet-50 FPN
  - Keypoint R-CNN with ResNet-50 FPN

Your chosen model will be loaded using a command line argument, and its checkpoints will automatically save to a dedicated folder.

## Outputs

The model generates:
- Model checkpoints (`<model_name>_checkpoints/model_checkpoint_epoch_*.pth`)
- Evaluation metrics (`output/evaluation_results.json`)
- Bounding box and mask visualization images (`output/*_with_boxes.jpg` and `output/*_with_masks.jpg`)
- Performance graphs (`output/*_graph.png`)
- Rhino 3DM files with extracted edges (`output/detected_edges_*.3dm`)

## Metrics

Evaluation produces the following metrics:
- Mean IoU (Intersection over Union)
- Precision
- Recall
- F1-Score

## Features

- Automatic checkpoint saving and loading into per-model subfolders
- CUDA support for GPU acceleration
- Custom data loading with COCO format support
- Edge extraction with Douglas-Peucker simplification
- Dual visualization outputs (boxes and masks)
- Category-based color coding
- Rhino 3DM export functionality
- Performance graphing across epochs via CLI arguments

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