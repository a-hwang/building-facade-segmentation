import torch
import torchvision

def evaluate_model(model, test_loader, device):
    """
    Evaluate model performance using segmentation masks.
    Computes average IoU, precision, recall, and F1-Score over the test set.
    """
    model.eval()
    total_iou = 0
    total_precision = 0
    total_recall = 0
    num_samples = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for i, output in enumerate(outputs):
                # Evaluate only if the model provides masks.
                if 'masks' not in output:
                    continue

                pred_masks = output['masks'][output['scores'] > 0.5]
                if len(pred_masks) == 0:
                    continue

                gt_masks = targets[i]['masks']
                if len(gt_masks) == 0:
                    continue

                # Prepare prediction and ground truth tensors.
                pred_masks = pred_masks.squeeze(1)
                pred_binary = (pred_masks > 0.5).float()

                # Resize ground truth masks if necessary.
                if pred_binary.shape != gt_masks.shape:
                    import torch.nn.functional as F
                    gt_masks = gt_masks.unsqueeze(1).float()
                    gt_masks = F.interpolate(gt_masks, size=pred_binary.shape[-2:], mode='nearest')
                    gt_masks = gt_masks.squeeze(1)

                if pred_binary.shape[0] != gt_masks.shape[0]:
                    min_instances = min(pred_binary.shape[0], gt_masks.shape[0])
                    pred_binary = pred_binary[:min_instances]
                    gt_masks = gt_masks[:min_instances]

                intersection = torch.sum(pred_binary * gt_masks, dim=(1, 2))
                union = torch.sum((pred_binary + gt_masks) > 0.5, dim=(1, 2))
                iou = (intersection / (union + 1e-6)).mean()

                true_positives = torch.sum(pred_binary * gt_masks, dim=(1, 2))
                pred_positives = torch.sum(pred_binary, dim=(1, 2))
                gt_positives = torch.sum(gt_masks, dim=(1, 2))
                precision = (true_positives / (pred_positives + 1e-6)).mean()
                recall = (true_positives / (gt_positives + 1e-6)).mean()

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