import torch
import torchvision

def evaluate_model_masks(model, test_loader, device):
    """
    Evaluate model performance using segmentation masks by aggregating instance masks 
    into single binary masks for pixel-level evaluation.
    Computes overall IoU, precision, recall, and F1-Score over the test set.
    """
    model.eval()
    total_tp = total_fp = total_fn = 0
    print("Starting mask evaluation...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            for i, output in enumerate(outputs):
                if 'masks' not in output:
                    continue
                # Select masks with score > 0.5.
                mask_preds = output['masks'][output['scores'] > 0.5]
                if len(mask_preds) == 0:
                    continue
                # Remove channel dimension: shape -> [N, H, W]
                mask_preds = mask_preds.squeeze(1)
                # Binarize predictions and aggregate via union (logical OR)
                binary_preds = (mask_preds > 0.5).float()
                agg_pred = (binary_preds.sum(dim=0) > 0.5).float()
                
                gt_masks = targets[i]['masks']
                if gt_masks.numel() == 0:
                    continue
                # Check and resize ground truth masks if shape differs.
                if agg_pred.shape != gt_masks.shape[1:]:
                    import torch.nn.functional as F
                    gt_masks = gt_masks.unsqueeze(1).float()
                    gt_masks = F.interpolate(gt_masks, size=agg_pred.shape, mode='nearest')
                    gt_masks = gt_masks.squeeze(1)
                # Aggregate ground truth masks.
                agg_gt = (gt_masks.sum(dim=0) > 0.5).float()
                
                # Compute pixel-wise true positives, false positives, false negatives.
                tp = ((agg_pred == 1) & (agg_gt == 1)).sum()
                fp = ((agg_pred == 1) & (agg_gt == 0)).sum()
                fn = ((agg_pred == 0) & (agg_gt == 1)).sum()
                
                total_tp += tp.item()
                total_fp += fp.item()
                total_fn += fn.item()
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
    metrics = {
        'IoU': iou,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    }
    return metrics

def evaluate_model_bbox(model, test_loader, device):
    """
    Evaluate model performance using bounding boxes.
    Computes average IoU, precision, recall, and F1-Score on the test set.
    """
    model.eval()
    total_iou = 0
    total_precision = 0
    total_recall = 0
    num_samples = 0

    print("Starting bounding box evaluation...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for i, output in enumerate(outputs):
                # Use boxes with score > 0.5.
                pred_boxes = output['boxes'][output['scores'] > 0.5]
                gt_boxes = targets[i]['boxes']
                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    continue

                # Compute IoU matrix using torchvision.ops.box_iou.
                iou_matrix = torchvision.ops.box_iou(pred_boxes, gt_boxes)
                max_iou, _ = iou_matrix.max(dim=1)
                iou = max_iou.mean()

                # Precision: fraction of predicted boxes that have sufficient overlap.
                tp = (max_iou > 0.5).float().sum()
                precision = tp / (pred_boxes.shape[0] + 1e-6)
                # Recall: check overlap for each ground truth box.
                max_iou_gt, _ = iou_matrix.max(dim=0)
                recall = (max_iou_gt > 0.5).float().sum() / (gt_boxes.shape[0] + 1e-6)

                total_iou += iou.item()
                total_precision += precision.item()
                total_recall += recall.item()
                num_samples += 1

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

def evaluate_model_masks_by_class(model, test_loader, device, target_class, train_coco):
    """
    Evaluate mask-based performance for a specific class.
    :param target_class: a lowercase string (e.g. "window")
    :param train_coco: dictionary loaded from train annotations, used to get class id mappings.
    """
    # Build a mapping: name (lowercase) -> id.
    class_map = {cat["name"].lower(): cat["id"] for cat in train_coco.get("categories", [])}
    print("DEBUG: Class map (masks):", class_map)  # Debug print to inspect mapping
    if target_class not in class_map:
        print(f"Class '{target_class}' not found in training categories.")
        return {}
    target_id = class_map[target_class]
    
    model.eval()
    total_tp = total_fp = total_fn = 0
    print(f"Starting mask evaluation for class '{target_class}'...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            for i, output in enumerate(outputs):
                if 'masks' not in output or 'labels' not in output:
                    continue
                # Filter predictions by score > 0.5 and target class.
                labels = output['labels']
                scores = output['scores']
                mask_preds = output['masks'][(scores > 0.5) & (labels == target_id)]
                if len(mask_preds) == 0:
                    continue
                mask_preds = mask_preds.squeeze(1)
                binary_preds = (mask_preds > 0.5).float()
                agg_pred = (binary_preds.sum(dim=0) > 0.5).float()
                
                # Filter ground truth masks using target labels.
                gt = targets[i]
                if 'masks' not in gt or 'labels' not in gt:
                    continue
                gt_labels = gt['labels']
                if target_id not in gt_labels:
                    continue
                # Select only those masks corresponding to target_id.
                gt_masks = gt['masks'][gt_labels == target_id]
                if gt_masks.numel() == 0:
                    continue
                if agg_pred.shape != gt_masks.shape[1:]:
                    import torch.nn.functional as F
                    gt_masks = gt_masks.unsqueeze(1).float()
                    gt_masks = F.interpolate(gt_masks, size=agg_pred.shape, mode='nearest')
                    gt_masks = gt_masks.squeeze(1)
                agg_gt = (gt_masks.sum(dim=0) > 0.5).float()
                
                tp = ((agg_pred == 1) & (agg_gt == 1)).sum()
                fp = ((agg_pred == 1) & (agg_gt == 0)).sum()
                fn = ((agg_pred == 0) & (agg_gt == 1)).sum()
                total_tp += tp.item()
                total_fp += fp.item()
                total_fn += fn.item()
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
    return {'IoU': iou, 'Precision': precision, 'Recall': recall, 'F1-Score': f1_score}


def evaluate_model_bbox_by_class(model, test_loader, device, target_class, train_coco):
    """
    Evaluate bbox-based performance for a specific class.
    """
    class_map = {cat["name"].lower(): cat["id"] for cat in train_coco.get("categories", [])}
    print("DEBUG: Class map (bbox):", class_map)  # Debug print to inspect mapping
    if target_class not in class_map:
        print(f"Class '{target_class}' not found in training categories.")
        return {}
    target_id = class_map[target_class]
    
    model.eval()
    total_iou = 0
    total_precision = 0
    total_recall = 0
    num_samples = 0

    print(f"Starting bounding box evaluation for class '{target_class}'...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            for i, output in enumerate(outputs):
                if 'boxes' not in output or 'labels' not in output:
                    continue
                # Filter predicted boxes by score > 0.5 and class.
                labels = output['labels']
                scores = output['scores']
                pred_boxes = output['boxes'][(scores > 0.5) & (labels == target_id)]
                # Filter ground truth boxes by class.
                gt = targets[i]
                if 'boxes' not in gt or 'labels' not in gt:
                    continue
                gt_boxes = gt['boxes'][gt['labels'] == target_id]
                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    continue

                iou_matrix = torchvision.ops.box_iou(pred_boxes, gt_boxes)
                max_iou, _ = iou_matrix.max(dim=1)
                iou = max_iou.mean()
                tp = (max_iou > 0.5).float().sum()
                precision = tp / (pred_boxes.shape[0] + 1e-6)
                max_iou_gt, _ = iou_matrix.max(dim=0)
                recall = (max_iou_gt > 0.5).float().sum() / (gt_boxes.shape[0] + 1e-6)
                total_iou += iou.item()
                total_precision += precision.item()
                total_recall += recall.item()
                num_samples += 1

    mean_iou = total_iou / num_samples if num_samples > 0 else 0
    mean_precision = total_precision / num_samples if num_samples > 0 else 0
    mean_recall = total_recall / num_samples if num_samples > 0 else 0
    f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall + 1e-6)
    return {'IoU': mean_iou, 'Precision': mean_precision, 'Recall': mean_recall, 'F1-Score': f1_score}