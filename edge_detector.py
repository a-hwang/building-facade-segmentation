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
                    labels.append(1)  # 1 for building class
                    
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

def build_model():
    # Load pre-trained model
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Modify the model for binary segmentation (building vs background)
    num_classes = 2  # background and building
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def extract_edges_from_masks(masks):
    edges = []
    for mask in masks:
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

def export_to_rhino(edges):
    model = rhino3dm.File3dm()
    
    for edge_points in edges:
        # Convert points to 3D points
        points3d = [rhino3dm.Point3d(x, y, 0) for x, y in edge_points]
        
        # Create polyline curve
        curve = rhino3dm.Curve.CreateControlPointCurve(points3d, 1)  # degree 1 for polyline
        
        # Add curve to model
        if curve.IsValid:
            model.Objects.AddCurve(curve)
    
    model.Write("detected_edges.3dm")

def collate_fn(batch):
    """
    Custom collate function to handle variable-sized tensors
    """
    return tuple(zip(*batch))

def main():
    # Create dataset and data loaders
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = COCODataset("train/_annotations.coco.json", transforms=transform)
    val_dataset = COCODataset("valid/_annotations.coco.json", transforms=transform)
    
    # Add collate_fn to DataLoader
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

    # Build and train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        print("Using CUDA GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU for training")
    model = build_model().to(device)
    
    # Training parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
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

    # Run inference
    model.eval()
    test_image = Image.open("test/image_example.jpg").convert("RGB")
    test_transform = ToTensor()
    test_image = Image.open("test/image_example.jpg").convert("RGB")
    with torch.no_grad():
        image, _ = test_transform(test_image, None)
        prediction = model([image.to(device)])
        masks = prediction[0]['masks'].cpu().numpy()
    
    edges = extract_edges_from_masks(masks)
    export_to_rhino(edges)

if __name__ == "__main__":
    main()
