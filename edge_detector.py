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

def build_model(num_classes):
    # Load pre-trained model
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Modify the model for multi-class detection
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
        curve = rhino3dm.Curve.CreateControlPointCurve(points3d, 1)
        if curve.IsValid:
            model3dm.Objects.AddCurve(curve)
    
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    out_path = os.path.join(output_folder, f"detected_edges_{base_name}.3dm")
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
                
                # Calculate IoU
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

def main():
    parser = argparse.ArgumentParser(description="Edge Detector")
    parser.add_argument('--inference', action='store_true', help="Run inference only without training")
    parser.add_argument('--random', action='store_true', help="Select a random test image from the test folder")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate model performance on test set")
    args = parser.parse_args()

    # Create dataset and data loaders
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = COCODataset("train/_annotations.coco.json", transforms=transform)
    val_dataset = COCODataset("valid/_annotations.coco.json", transforms=transform)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

    # Determine number of classes from annotations
    with open("train/_annotations.coco.json", "r") as f:
        train_coco = json.load(f)
    num_categories = len(train_coco.get("categories", []))
    num_classes = num_categories + 1  # +1 for background

    # Build model and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        print("Using CUDA GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU for training")
    model = build_model(num_classes).to(device)
    
    # Define the checkpoints folder
    checkpoints_folder = "model_checkpoints"
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    # Get a list of existing checkpoint files (e.g., model_checkpoint_epoch_9.pth)
    checkpoint_files = glob.glob(os.path.join(checkpoints_folder, "model_checkpoint_epoch_*.pth"))
    starting_epoch = 0
    if checkpoint_files:
        # Get the checkpoint with the largest epoch number
        latest_ckpt = max(checkpoint_files, key=lambda fname: int(re.findall(r'\d+', fname)[0]))
        # Load the checkpoint
        model.load_state_dict(torch.load(latest_ckpt))
        # Parse the epoch number from filename and set training to resume after that epoch
        starting_epoch = int(re.findall(r'\d+', latest_ckpt)[0]) + 1
        print(f"Loaded model checkpoint from {latest_ckpt}")
    else:
        # You can still load a checkpoint passed by args if required (optional)
        pass

    if args.evaluate:
        print("Evaluating model on test set...")
        # Create test dataset and loader
        test_dataset = COCODataset("test/_annotations.coco.json", transforms=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
        
        # Run evaluation
        metrics = evaluate_model(model, test_loader, device)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Mean IoU: {metrics['IoU']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1-Score: {metrics['F1-Score']:.4f}")
        
        # Save metrics to file
        results_path = os.path.join("output", "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\nResults saved to {results_path}")
        
        if not args.inference:  # If not only running inference, continue with training
            print("\nProceeding with training...")

    # Run training loop only if not in inference mode
    if not args.inference:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        num_epochs = 10  # Total epochs to run from the resume point

        for epoch in range(starting_epoch, starting_epoch + num_epochs):
            model.train()
            total_loss = 0
            for images, targets in train_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                total_loss += losses.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
            # Save checkpoint with the epoch number appended
            ckpt_path = os.path.join(checkpoints_folder, f"model_checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Run inference and overlay labels on test image
    model.eval()
    test_folder = "test"
    if args.random:
        test_files = glob.glob(os.path.join(test_folder, "*.jpg"))
        if not test_files:
            print("No jpg files found in the test folder.")
            return
        test_image_path = random.choice(test_files)
        print("Selected random test image:", test_image_path)
    else:
        test_image_path = os.path.join(test_folder, "20230329_200455_067_R_scaled_1_png_jpg.rf.ca555ec50f4cc78a87475458b99004dc.jpg")
    
    test_image = Image.open(test_image_path).convert("RGB")
    test_transform = ToTensor()
    image, _ = test_transform(test_image, None)
    with torch.no_grad():
        prediction = model([image.to(device)])
        masks = prediction[0]['masks'].cpu().numpy()

    test_image_cv = cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2BGR)
    preds = prediction[0]
    boxes = preds['boxes'].cpu().numpy()
    labels = preds['labels'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()

    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Use the base name from the test image for output files
    base_name = os.path.splitext(os.path.basename(test_image_path))[0]
    for box, label, score in zip(boxes, labels, scores):
        if score < 0.5:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(test_image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            test_image_cv,
            f"ID:{label} {score:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    overlay_path = os.path.join(output_folder, f"{base_name}_with_labels.jpg")
    cv2.imwrite(overlay_path, test_image_cv)
    print("Overlayed image saved to", overlay_path)

    # Pass along base_name to export_to_rhino so output filename contains it
    export_to_rhino(masks, base_name)

if __name__ == "__main__":
    main()
