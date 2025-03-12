import argparse
import json
import os
import re
import random
import glob
import torch
from torch.utils.data import DataLoader

from dataset import COCODataset, Compose, ToTensor, Normalize, collate_fn
from model_utils import build_model
from evaluation import evaluate_model_bbox, evaluate_model_masks  # update imports
from visualization import visualize_predictions, export_to_rhino
from utils import get_device, to_device, get_unique_filepath

# Add allowed class names in lowercase.
ALLOWED_CLASSES = ["facade", "balcony-fence", "car", "fence", "non-building-infrastructure", "shop", "street", "traffic-infrastructure", "vegetation", "window", "all"]

def main():
    parser = argparse.ArgumentParser(description="Edge Detector")
    parser.add_argument('--train', type=int, metavar='N', help="Train model for N epochs")
    # Change --evaluate to accept parameters "bbox" or "mask"
    parser.add_argument('--evaluate', type=str, choices=["bbox", "mask"],
                        help="Run evaluation on the test set. Specify evaluation type (bbox or mask).")
    parser.add_argument('--class', dest="target_class", type=str, choices=ALLOWED_CLASSES,
                        help="Evaluate only a specific class (e.g. 'window') or 'all' for per-class graphs")
    parser.add_argument('--inference', action='store_true', help="Run inference only on a test image")
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
    parser.add_argument('--force-cpu', action='store_true', help="Force CPU usage even if a GPU is available")
    parser.add_argument('--graph-epochs', type=int, nargs='+', help="Specify epochs to graph")
    parser.add_argument('--graph-range', type=str, help="Specify range of epochs to graph (e.g., '1-10')")
    args = parser.parse_args()

    # Require at least one of the operations to be specified.
    if not any([args.train is not None, args.evaluate, args.inference, args.graph_epochs, args.graph_range]):
        parser.error("At least one of --train, --evaluate, --inference, --graph-epochs or --graph-range must be specified")
    
    # When graph arguments are provided, require --evaluate
    if (args.graph_epochs or args.graph_range) and args.evaluate is None:
        parser.error("When using graph commands, --evaluate must be specified with a value (bbox or mask)")
    
    # Set device.
    if args.force_cpu:
        device = torch.device('cpu')
        device_name = "CPU (forced)"
    else:
        device, device_name = get_device()

    # Setup transform and datasets.
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

    # Load checkpoint if available.
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

    # Training phase.
    if args.train is not None:
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                    lr=0.001, momentum=0.9, weight_decay=0.0005)
        num_epochs = args.train
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
                
                # Save checkpoint on CPU for consistency.
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

    # Evaluation phase.
    if args.evaluate is not None and not (args.graph_epochs or args.graph_range):
        print(f"\nEvaluating model on test set using {args.evaluate} evaluation...")
        test_dataset = COCODataset("test/_annotations.coco.json", transforms=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
        checkpoint_info = args.checkpoint if args.checkpoint is not None else "latest"
        if args.target_class:
            # Call new class-filtered evaluation functions.
            if args.evaluate == "bbox":
                from evaluation import evaluate_model_bbox_by_class
                metrics = evaluate_model_bbox_by_class(model, test_loader, device, args.target_class, train_coco= train_coco)
            else:
                from evaluation import evaluate_model_masks_by_class
                metrics = evaluate_model_masks_by_class(model, test_loader, device, args.target_class, train_coco= train_coco)
        else:
            # Default aggregated evaluation.
            if args.evaluate == "bbox":
                metrics = evaluate_model_bbox(model, test_loader, device)
            else:
                metrics = evaluate_model_masks(model, test_loader, device)
        print("\nEvaluation Results:")
        print(f"Mean IoU: {metrics['IoU']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1-Score: {metrics['F1-Score']:.4f}")
        results_filename = f"{args.model}_epoch_{checkpoint_info}_{args.evaluate}_{args.target_class or 'all'}_evaluation_results.json"
        results_path = os.path.join("output", results_filename)
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\nResults saved to {results_path}")

    # Inference phase.
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
            test_image_path = os.path.join(test_folder, "default_test_image.jpg")
        
        from PIL import Image
        test_image = Image.open(test_image_path).convert("RGB")
        test_transform = ToTensor()
        image, _ = test_transform(test_image, None)
        with torch.no_grad():
            prediction = model([image.to(device)])
        categories = {cat["id"]: cat["name"] for cat in train_coco["categories"]}
        checkpoint_info = args.checkpoint if args.checkpoint is not None else "latest"
        original_base = os.path.splitext(os.path.basename(test_image_path))[0]
        new_base_name = f"{args.model}_epoch_{checkpoint_info}_{original_base}"
        
        # Visualize predictions and export Rhino file if masks are available.
        masks = visualize_predictions(test_image, prediction, categories, "output", new_base_name)
        if masks is not None:
            export_to_rhino(masks, new_base_name)
        else:
            print("Skipping Rhino export as no masks are available.")

    # Graphing phase.
    if args.graph_epochs or args.graph_range:
        test_dataset = COCODataset("test/_annotations.coco.json", transforms=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
        epochs_to_graph = set()
        if args.graph_epochs:
            epochs_to_graph.update(args.graph_epochs)
        if args.graph_range:
            try:
                start, end = map(int, args.graph_range.split('-'))
                epochs_to_graph.update(range(start, end+1))
            except Exception as e:
                print(f"Could not parse --graph-range: {args.graph_range}. Error: {str(e)}")
        checkpoints_folder = f"{args.model}_checkpoints"
        from visualization import graph_model_performance
        # Pass the evaluate type, target class, and train_coco.
        graph_model_performance(list(epochs_to_graph), model, test_loader, device,
                                checkpoints_folder, args.model, args.evaluate, args.target_class, train_coco)

if __name__ == "__main__":
    main()