# Edge Detection Model for Building Facades

A PyTorch-based implementation of an edge detection model using instance segmentation. By default, it uses Mask R-CNN with a ResNet-50 backbone for detecting and extracting edges from building facades. New features include class-specific evaluation/graphing and Rhino 3DM export for segmentation masks.

## Overview

This project supports:
- **Training:** Train the model on custom datasets in COCO format.
- **Evaluation:** Evaluate the model using either bounding box or mask-based metrics.
- **Class-specific Evaluation/Graphing:** Limit evaluation and performance graphs to a specific class or see per-class curves.
- **Inference:** Run inference on test images with visualizations of predictions (bounding boxes and segmentation masks). If masks are present, edges are extracted and exported as Rhino 3DM files.
- **Graphing:** Generate graphs of model performance (IoU, Precision, Recall, F1-Score) across selected epochs. **Note: When using graph commands, `--evaluate` (with either "bbox" or "mask") is required.**

The outputs include:
- **Checkpoints** saved in a per-model folder (e.g., `maskrcnn_resnet50_fpn_checkpoints`)
- **Evaluation Metrics** saved as JSON files in the `output/` folder
- **Visualization Images** (bounding boxes and, when available, masks)
- **Performance Graphs** as PNG images in `output/`
- **Rhino 3DM Files** with extracted edge curves from segmentation masks

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
└── <model_name>_checkpoints/    # Checkpoints saved per-model
```

## Categories

The model detects the following building elements (use lowercase names when specifying classes via the CLI):
- facade
- balcony-fence
- car
- fence
- non-building-infrastructure
- shop
- street
- traffic-infrastructure
- vegetation
- window

## Usage

### Training

Train the model for a given number of epochs using the default model (Mask R-CNN with ResNet-50):

```bash
python main.py --train 10
```

Specify a different pretrained model with the `--model` parameter. For example, to use Faster R-CNN with ResNet-50:

```bash
python main.py --train 10 --model fasterrcnn_resnet50_fpn
```

Checkpoints are saved in a dedicated subfolder (e.g., `maskrcnn_resnet50_fpn_checkpoints`).

### Evaluation

To evaluate the model on your test set, use the `--evaluate` argument with a value:
- `--evaluate bbox` for bounding box based evaluation
- `--evaluate mask` for mask based evaluation

For example:
```bash
python main.py --evaluate mask
```

You can additionally filter evaluation by class using the `--class` parameter (e.g., `--class window`). When not provided, aggregated evaluation is used.

### Inference

Run inference on a specified test image:
```bash
python main.py --inference --file path/to/test_image.jpg
```

Or run inference on a random test image:
```bash
python main.py --inference --random
```

During inference, predicted bounding boxes and, if available, segmentation masks are visualized. If masks are present, extracted edge curves are exported as Rhino 3DM files.

### Graphing Performance

Generate performance graphs over selected epochs with the `--graph-epochs` or `--graph-range` arguments. **Note: When using graph commands, you must specify `--evaluate` with a value ("bbox" or "mask").**

For example, to graph a range of epochs:
```bash
python main.py --graph-range 1-10 --evaluate bbox
```

You can also add the `--class` parameter to limit graphs to a specific class (e.g., `--class window`), or use `--class all` to generate separate graphs per metric with lines for each class.

The performance graphs and raw metrics JSON files are saved in the `output/` folder.

### Combined Operations

You can combine commands; for example:
```bash
python main.py --train 10 --evaluate mask --class window --graph-range 1-10
```

This sequence will train the model, then perform a mask-based evaluation for the class "window" and generate corresponding performance graphs.

## Model Architecture

- **Default Model:** Mask R-CNN  
- **Backbone:** ResNet-50  
- **Alternate Models:** You may choose among:
  - fasterrcnn_resnet50_fpn
  - fasterrcnn_resnet50_fpn_v2
  - fasterrcnn_mobilenet_v3_large_fpn
  - retinanet_resnet50_fpn
  - fcos_resnet50_fpn
  - keypointrcnn_resnet50_fpn

The chosen model is loaded using a CLI argument, and its checkpoints are automatically saved to a dedicated folder.

## Outputs

The model generates:
- **Model Checkpoints:** (e.g., `maskrcnn_resnet50_fpn_checkpoints/model_checkpoint_epoch_*.pth`)
- **Evaluation Metrics:** JSON files saved to `output/`
- **Visualization Images:** Bounding box and mask images saved to `output/`
- **Performance Graphs:** PNG files saved to `output/`
- **Rhino 3DM Files:** Files with extracted edge curves saved to `output/`

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

The model supports:
- NVIDIA CUDA GPUs
- Apple Metal (MPS for M1/M2 Macs)
- CPU (fallback)

### Troubleshooting for Apple Metal

If you experience errors, force CPU usage:
```bash
python main.py --train 10 --force-cpu
```

And consider reducing the batch size or upgrading PyTorch.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- Pillow
- NumPy
- rhino3dm

## Contact and License

For issues or contributions, please refer to the repository guidelines.