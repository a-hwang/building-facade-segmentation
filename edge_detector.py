import cv2
import torch
import torchvision
import rhino3dm
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
import json
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import re
import glob
import random

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            if target is not None:
                image, target = t(image, target)
            else:
                image = t(image)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = torchvision.transforms.functional.normalize(image, self.mean, self.std)
        return image, target

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, coco_annotations_path, transforms=None):
        self.transforms = transforms
        self.root_dir = os.path.dirname(coco_annotations_path)
        with open(coco_annotations_path, "r") as f:
            self.coco = json.load(f)
        self.images = self.coco.get("images", [])
        self.annotations = self.coco.get("annotations", [])

    def create_mask(self, height, width, polygons):
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            # Convert polygon to numpy array
            pts = np.array(polygon).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return torch.as_tensor(mask, dtype=torch.uint8)

    def __getitem__(self, idx):
        try:
            image_info = self.images[idx]
            image_path = os.path.join(self.root_dir, image_info["file_name"])
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            
            # Collect annotations for this image
            target_annotations = [ann for ann in self.annotations if ann["image_id"] == image_info["id"]]
            
            boxes = []
            labels = []
            masks = []
            
            for ann in target_annotations:
                if "bbox" in ann and "segmentation" in ann:
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    # Use the actual category id from the annotation
                    labels.append(ann["category_id"])
                    
                    # Create binary mask from segmentation
                    mask = self.create_mask(height, width, ann["segmentation"])
                    masks.append(mask)
            
            # Convert to tensor format required by Mask R-CNN
            boxes = torch.FloatTensor(boxes) if boxes else torch.zeros((0, 4))
            labels = torch.LongTensor(labels) if labels else torch.zeros(0, dtype=torch.int64)
            masks = torch.stack(masks) if masks else torch.zeros((0, height, width), dtype=torch.uint8)
            
            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": torch.tensor([image_info["id"]])
            }
            
            if self.transforms is not None:
                image, target = self.transforms(image, target)
                
            return image, target
            
        except Exception as e:
            print(f"Error loading image {idx}: {str(e)}")
            # Return a dummy sample with masks
            return torch.zeros((3, 224, 224)), {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks": torch.zeros((0, 224, 224), dtype=torch.uint8),
                "image_id": torch.tensor([0])
            }

    def __len__(self):
        return len(self.images)

def build_model(model_name, num_classes):
    # Get the constructor from torchvision.models.detection using the model name.
    model_fn = getattr(torchvision.models.detection, model_name)
    model = model_fn(weights="DEFAULT")
    # For models with ROI heads (which have a box_predictor), modify for our number of classes.
    if model_name in {
        "maskrcnn_resnet50_fpn", 
        "fasterrcnn_resnet50_fpn", 
        "fasterrcnn_resnet50_fpn_v2", 
        "fasterrcnn_mobilenet_v3_large_fpn"
    }:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def extract_edges_from_masks(masks):
    edges = []
    for mask in masks:
        # Squeeze to ensure it's 2D (H x W)
        mask = np.squeeze(mask)
        # Convert float mask to binary uint8
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Find contours using cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to polylines
        for contour in contours:
            # Simplify contour using Douglas-Peucker algorithm
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert to list of points
            points = [(point[0][0], point[0][1]) for point in approx]
            edges.append(points)
    
    return edges

def export_to_rhino(masks, base_name):
    edges = extract_edges_from_masks(masks)
    model3dm = rhino3dm.File3dm()
    
    for edge_points in edges:
        points3d = [rhino3dm.Point3d(x, y, 0) for x, y in edge_points]
        if len(points3d) < 2:  # Skip if not enough points to form a curve.
            continue
        curve = rhino3dm.Curve.CreateControlPointCurve(points3d, 1)
        if curve is not None and curve.IsValid:
            model3dm.Objects.AddCurve(curve)
    
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    out_path = os.path.join(output_folder, f"detected_edges_{base_name}.3dm")
    out_path = get_unique_filepath(out_path)
    model3dm.Write(out_path, 6)
    print("Exported Rhino file to", out_path)

def collate_fn(batch):
    """
    Custom collate function to handle variable-sized tensors
    """
    return tuple(zip(*batch))

def evaluate_model(model, test_loader, device):
    model.eval()
    total_iou = 0
    total_precision = 0
    total_recall = 0
    num_samples = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                # Get predicted masks above threshold
                pred_masks = output['masks'][output['scores'] > 0.5]
                if len(pred_masks) == 0:
                    continue
                
                # Get ground truth masks
                gt_masks = targets[i]['masks']
                if len(gt_masks) == 0:
                    continue
                
                # Calculate metrics
                pred_masks = pred_masks.squeeze(1)
                pred_binary = (pred_masks > 0.5).float()
                
                # Ensure gt_masks shape matches pred_binary; if not, resize gt_masks
                if pred_binary.shape != gt_masks.shape:
                    import torch.nn.functional as F
                    gt_masks = gt_masks.unsqueeze(1).float()  # add channel dimension
                    gt_masks = F.interpolate(gt_masks, size=pred_binary.shape[-2:], mode='nearest')
                    gt_masks = gt_masks.squeeze(1)
                
                # If the number of instances still don't match, trim both tensors:
                if pred_binary.shape[0] != gt_masks.shape[0]:
                    min_instances = min(pred_binary.shape[0], gt_masks.shape[0])
                    pred_binary = pred_binary[:min_instances]
                    gt_masks = gt_masks[:min_instances]

                # Now calculate IoU
                intersection = torch.sum(pred_binary * gt_masks, dim=(1,2))
                union = torch.sum(pred_binary + gt_masks > 0.5, dim=(1,2))
                iou = (intersection / (union + 1e-6)).mean()
                
                # Calculate Precision and Recall
                true_positives = torch.sum(pred_binary * gt_masks, dim=(1,2))
                pred_positives = torch.sum(pred_binary, dim=(1,2))
                gt_positives = torch.sum(gt_masks, dim=(1,2))
                
                precision = (true_positives / (pred_positives + 1e-6)).mean()
                recall = (true_positives / (gt_positives + 1e-6)).mean()
                
                total_iou += iou.item()
                total_precision += precision.item()
                total_recall += recall.item()
                num_samples += 1
    
    # Calculate final metrics
    mean_iou = total_iou / num_samples if num_samples > 0 else 0
    mean_precision = total_precision / num_samples if num_samples > 0 else 0
    mean_recall = total_recall / num_samples if num_samples > 0 else 0
    f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall + 1e-6)
    
    metrics = {
        'IoU': mean_iou,
        'Precision': mean_precision,
        'Recall': mean_recall,
        'F1-Score': f1_score
    }
    
    return metrics

def visualize_predictions(image, prediction, categories, output_folder, base_name):
    """Helper function to visualize both bounding boxes and masks"""
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    preds = prediction[0]
    boxes = preds['boxes'].cpu().numpy()
    labels = preds['labels'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()
    masks = preds['masks'].cpu().numpy()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    bbox_image = image_cv.copy()
    for box, label, score in zip(boxes, labels, scores):
        if score < 0.5:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        category_name = categories.get(int(label), "unknown")
        cv2.putText(
            bbox_image,
            f"{category_name} {score:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )
    bbox_path = os.path.join(output_folder, f"{base_name}_with_boxes.jpg")
    bbox_path = get_unique_filepath(bbox_path)
    cv2.imwrite(bbox_path, bbox_image)

    mask_image = image_cv.copy()
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(categories), 3), dtype=np.uint8)
    
    for mask, label, score in zip(masks, labels, scores):
        if score < 0.5:
            continue
        
        color = colors[label].tolist()
        binary_mask = (mask[0] > 0.5).astype(np.uint8)
        colored_mask = np.zeros_like(image_cv)
        colored_mask[binary_mask > 0] = color
        cv2.addWeighted(colored_mask, 0.5, mask_image, 1, 0, mask_image)
        
        moments = cv2.moments(binary_mask)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            category_name = categories.get(int(label), "unknown")
            cv2.putText(
                mask_image,
                f"{category_name} {score:.2f}",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
    mask_path = os.path.join(output_folder, f"{base_name}_with_masks.jpg")
    mask_path = get_unique_filepath(mask_path)
    cv2.imwrite(mask_path, mask_image)
    print(f"Saved visualizations to {output_folder}")
    
    return masks

def get_device():
    """Helper function to get the best available device with fallback options"""
    if torch.cuda.is_available():
        return torch.device('cuda'), "CUDA GPU: " + torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            # Test MPS with a small tensor operation
            test_tensor = torch.ones(1).to('mps')
            _ = test_tensor + test_tensor
            return torch.device('mps'), "Apple Metal GPU"
        except Exception as e:
            print(f"Warning: MPS (Metal) initialization failed, falling back to CPU. Error: {str(e)}")
            return torch.device('cpu'), "CPU (MPS fallback)"
    return torch.device('cpu'), "CPU"

def to_device(data, device):
    """Helper function to safely move data to device"""
    try:
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        elif isinstance(data, dict):
            return {k: to_device(v, device) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        return data
    except Exception as e:
        print(f"Warning: Failed to move data to device {device}, using CPU. Error: {str(e)}")
        if isinstance(data, torch.Tensor):
            return data.cpu()
        return data

def get_unique_filepath(filepath):
    if not os.path.exists(filepath):
        return filepath
    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = f"{base}_{counter}{ext}"
    while os.path.exists(new_filepath):
        counter += 1
        new_filepath = f"{base}_{counter}{ext}"
    return new_filepath

def main():
    parser = argparse.ArgumentParser(description="Edge Detector")
    parser.add_argument('--train', type=int, metavar='N', help="Train model for N epochs")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate model performance on test set")
    parser.add_argument('--inference', action='store_true', help="Run inference only on test image")
    parser.add_argument('--random', action='store_true', help="Select a random test image from the test folder")
    parser.add_argument('--file', type=str, help="Specify a test image file for inference")
    parser.add_argument('--checkpoint', type=int, help="Manually specify checkpoint epoch number to load")
    parser.add_argument('--model', type=str, default="maskrcnn_resnet50_fpn",
                        choices=[
                            "maskrcnn_resnet50_fpn",
                            "fasterrcnn_resnet50_fpn",
                            "fasterrcnn_resnet50_fpn_v2",
                            "fasterrcnn_mobilenet_v3_large_fpn",
                            "retinanet_resnet50_fpn",
                            "fcos_resnet50_fpn",
                            "keypointrcnn_resnet50_fpn"
                        ],
                        help="Select which pretrained model to use. Defaults to maskrcnn_resnet50_fpn")
    parser.add_argument('--force-cpu', action='store_true', help="Force CPU usage even if GPU is available")
    args = parser.parse_args()
    
    if args.force_cpu:
        device = torch.device('cpu')
        device_name = "CPU (forced)"
    else:
        device, device_name = get_device()
    
    if not any([args.train is not None, args.evaluate, args.inference]):
        parser.error("At least one of --train, --evaluate, or --inference must be specified")
    
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = COCODataset("train/_annotations.coco.json", transforms=transform)
    val_dataset = COCODataset("valid/_annotations.coco.json", transforms=transform)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)
    
    with open("train/_annotations.coco.json", "r") as f:
        train_coco = json.load(f)
    num_categories = len(train_coco.get("categories", []))
    num_classes = num_categories + 1
    
    print(f"Using {device_name} for training")
    model = build_model(args.model, num_classes).to(device)
    
    # Use a subfolder for checkpoints based on the chosen model.
    checkpoints_folder = f"{args.model}_checkpoints"
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)
    
    starting_epoch = 0
    if args.checkpoint:
        ckpt_path = os.path.join(checkpoints_folder, f"model_checkpoint_epoch_{args.checkpoint}.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint)
            starting_epoch = args.checkpoint + 1
            print(f"Loaded checkpoint from {ckpt_path}")
        else:
            print(f"Checkpoint {ckpt_path} not found. Starting from scratch.")
    else:
        checkpoint_files = glob.glob(os.path.join(checkpoints_folder, "model_checkpoint_epoch_*.pth"))
        if checkpoint_files:
            latest_ckpt = max(checkpoint_files, key=lambda fname: int(re.findall(r'\d+', fname)[0]))
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint)
            starting_epoch = int(re.findall(r'\d+', latest_ckpt)[0]) + 1
            print(f"Loaded latest checkpoint from {latest_ckpt}")

    # Training phase
    if args.train is not None:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        num_epochs = args.train  # Use specified number of epochs

        try:
            for epoch in range(starting_epoch, starting_epoch + num_epochs):
                model.train()
                total_loss = 0
                for images, targets in train_loader:
                    try:
                        images = to_device(images, device)
                        targets = to_device(targets, device)

                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())

                        optimizer.zero_grad()
                        losses.backward()
                        optimizer.step()

                        total_loss += losses.item()
                    except RuntimeError as e:
                        print(f"Warning: Training batch failed, skipping. Error: {str(e)}")
                        continue

                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
                
                # Save checkpoint on CPU
                model.to('cpu')
                ckpt_path = os.path.join(checkpoints_folder, f"model_checkpoint_epoch_{epoch}.pth")
                torch.save(model.state_dict(), ckpt_path)
                model.to(device)
                print(f"Saved checkpoint to {ckpt_path}")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            print("Falling back to CPU...")
            device = torch.device('cpu')
            model = model.cpu()
            # Continue training on CPU

    # Evaluation phase
    if args.evaluate:
        print("\nEvaluating model on test set...")
        test_dataset = COCODataset("test/_annotations.coco.json", transforms=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
        
        metrics = evaluate_model(model, test_loader, device)
        
        print("\nEvaluation Results:")
        print(f"Mean IoU: {metrics['IoU']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1-Score: {metrics['F1-Score']:.4f}")
        
        results_path = os.path.join("output", "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\nResults saved to {results_path}")

    # Inference phase
    if args.inference or args.random:
        model.eval()
        test_folder = "test"
        
        if args.file:
            test_image_path = args.file
            if not os.path.exists(test_image_path):
                print(f"Specified file {test_image_path} does not exist.")
                return
            print("Using specified test image:", test_image_path)
        elif args.random:
            test_files = glob.glob(os.path.join(test_folder, "*.jpg"))
            if not test_files:
                print("No jpg files found in the test folder.")
                return
            test_image_path = random.choice(test_files)
            print("Selected random test image:", test_image_path)
        else:
            test_image_path = os.path.join(test_folder, "20230329_200455_067_R_scaled_1_png_jpg.rf.ca555ec50f4cc78a87475458b99004dc.jpg")
        
        # Load and process image
        test_image = Image.open(test_image_path).convert("RGB")
        test_transform = ToTensor()
        image, _ = test_transform(test_image, None)
        
        # Get predictions
        with torch.no_grad():
            prediction = model([image.to(device)])
        
        # Get category mapping
        categories = {cat["id"]: cat["name"] for cat in train_coco["categories"]}
        
        # Generate visualizations and get masks
        base_name = os.path.splitext(os.path.basename(test_image_path))[0]
        masks = visualize_predictions(test_image, prediction, categories, "output", base_name)
        
        # Export to Rhino
        export_to_rhino(masks, base_name)

if __name__ == "__main__":
    main()
