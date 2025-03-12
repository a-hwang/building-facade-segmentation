import os
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import numpy as np
import rhino3dm
from utils import get_unique_filepath  # reuse helper from utils.py
from evaluation import evaluate_model_bbox  # reuse helper from evaluation.py

def extract_edges_from_masks(masks):
    """
    Extracts simplified edge contours from each mask using Douglas-Peucker algorithm.
    """
    edges = []
    for mask in masks:
        mask = np.squeeze(mask)
        mask = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = [(point[0][0], point[0][1]) for point in approx]
            edges.append(points)
    return edges

def export_to_rhino(masks, base_name):
    """
    Exports detected edges as curves in a Rhino 3DM file.
    """
    edges = extract_edges_from_masks(masks)
    model3dm = rhino3dm.File3dm()
    for edge_points in edges:
        points3d = [rhino3dm.Point3d(x, y, 0) for x, y in edge_points]
        if len(points3d) < 2:
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

def visualize_predictions(image, prediction, categories, output_folder, base_name):
    """
    Visualizes predictions by drawing bounding boxes and masks (if available) on the image.
    """
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    preds = prediction[0]
    boxes = preds['boxes'].cpu().numpy()
    labels = preds['labels'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()
    masks = preds['masks'].cpu().numpy() if 'masks' in preds else None
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    bbox_image = image_cv.copy()
    for box, label, score in zip(boxes, labels, scores):
        if score < 0.5:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        category_name = categories.get(int(label), "unknown")
        cv2.putText(bbox_image, f"{category_name} {score:.2f}", (x1, max(y1-10,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    bbox_path = os.path.join(output_folder, f"{base_name}_with_boxes.jpg")
    bbox_path = get_unique_filepath(bbox_path)
    cv2.imwrite(bbox_path, bbox_image)
    if masks is not None:
        mask_image = image_cv.copy()
        np.random.seed(42)
        colors = np.random.randint(0,255, size=(len(categories), 3), dtype=np.uint8)
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
                cx = int(moments["m10"]/moments["m00"])
                cy = int(moments["m01"]/moments["m00"])
                category_name = categories.get(int(label), "unknown")
                cv2.putText(mask_image, f"{category_name} {score:.2f}", (cx,cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
        mask_path = os.path.join(output_folder, f"{base_name}_with_masks.jpg")
        mask_path = get_unique_filepath(mask_path)
        cv2.imwrite(mask_path, mask_image)
        print(f"Saved visualizations to {output_folder}")
    else:
        print("No masks predicted; skipping mask visualization.")
    return masks

def graph_model_performance(epochs, model, test_loader, device, checkpoints_folder, model_name):
    """
    For each epoch in 'epochs', if the checkpoint exists, load it,
    evaluate the model using bounding boxes, and then graph IoU,
    Precision, Recall, and F1-Score.
    The graph title will be <model>_epoch_<largest epoch>_graph.
    """
    epochs_list = []
    iou_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    for epoch in sorted(epochs):
        ckpt_path = os.path.join(checkpoints_folder, f"model_checkpoint_epoch_{epoch}.pth")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint for epoch {epoch} is missing. Skipping.")
            continue
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint)
        metrics = evaluate_model_bbox(model, test_loader, device)
        print(f"Epoch {epoch} - IoU: {metrics['IoU']:.4f}, Precision: {metrics['Precision']:.4f}, "
              f"Recall: {metrics['Recall']:.4f}, F1: {metrics['F1-Score']:.4f}")
        epochs_list.append(epoch)
        iou_list.append(metrics['IoU'])
        precision_list.append(metrics['Precision'])
        recall_list.append(metrics['Recall'])
        f1_list.append(metrics['F1-Score'])
    
    if not epochs_list:
        print("No valid checkpoints found. Exiting graph generation.")
        return
    
    largest_epoch = max(epochs_list)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, iou_list, label='IoU', marker='o')
    plt.plot(epochs_list, precision_list, label='Precision', marker='o')
    plt.plot(epochs_list, recall_list, label='Recall', marker='o')
    plt.plot(epochs_list, f1_list, label='F1-Score', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title(f"{model_name}_epoch_{largest_epoch}_graph")
    plt.legend()
    plt.grid(True)
    graph_path = os.path.join("output", f"{model_name}_epoch_{largest_epoch}_graph.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"Performance graph saved to {graph_path}")