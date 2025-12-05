import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import numpy as np
from collections import defaultdict
import torchvision.transforms as T

# --- Configuration ---
base_path = "/courses/CS5330.202610/students/patel.harshil3/FinalProject"
dataset_path = os.path.join(base_path, "dataset")
model_path = os.path.join(base_path, "models/best_model.pth")

NUM_CLASSES = 4
IMAGE_SIZE = 960
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

CATEGORIES = {
    1: 'human',
    2: 'trafficsign',
    3: 'vehicle'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# --- Dataset ---
class CocoDataset(Dataset):
    def __init__(self, img_dir, annotation_file, image_size=960):
        self.img_dir = img_dir
        self.image_size = image_size
        
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            img = Image.new('RGB', (self.image_size, self.image_size))
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'image_id': torch.tensor([img_info['id']])
            }
            img_tensor = T.ToTensor()(img)
            return img_tensor, target
        
        img = Image.open(img_path).convert("RGB")
        orig_width, orig_height = img.size
        
        img_id = img_info['id']
        anns = self.img_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        img_resized = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        scale_x = self.image_size / orig_width
        scale_y = self.image_size / orig_height
        
        scaled_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            scaled_box = [
                x_min * scale_x,
                y_min * scale_y,
                x_max * scale_x,
                y_max * scale_y
            ]
            scaled_boxes.append(scaled_box)
        
        boxes = torch.as_tensor(scaled_boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }
        
        img_tensor = T.ToTensor()(img_resized)
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- IoU Calculation ---
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# --- Calculate AP for one class ---
def calculate_ap(precisions, recalls):
    """Calculate Average Precision using 11-point interpolation"""
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap

# --- Evaluation ---
def evaluate_model(model, data_loader, device, conf_threshold=0.5, iou_threshold=0.5):
    model.eval()
    
    # Store all predictions and ground truths per class
    class_predictions = defaultdict(list)  # {class_id: [(score, is_correct), ...]}
    class_ground_truths = defaultdict(int)  # {class_id: num_gt_boxes}
    
    total_images = 0
    total_predictions = 0
    total_ground_truth = 0
    
    # For overall metrics
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    
    print("Evaluating model on validation set...\n")
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)
            
            for pred, target in zip(predictions, targets):
                total_images += 1
                
                # Filter predictions by confidence
                keep = pred['scores'] > conf_threshold
                pred_boxes = pred['boxes'][keep].cpu().numpy()
                pred_labels = pred['labels'][keep].cpu().numpy()
                pred_scores = pred['scores'][keep].cpu().numpy()
                
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                total_predictions += len(pred_boxes)
                total_ground_truth += len(gt_boxes)
                
                # Count ground truths per class
                for label in gt_labels:
                    class_ground_truths[label] += 1
                
                # Track which ground truths have been matched
                matched_gt = set()
                
                # Sort predictions by score (descending)
                sorted_indices = np.argsort(-pred_scores)
                
                # Match predictions to ground truths
                for idx in sorted_indices:
                    pred_box = pred_boxes[idx]
                    pred_label = pred_labels[idx]
                    pred_score = pred_scores[idx]
                    
                    best_iou = 0
                    best_gt_idx = -1
                    
                    # Find best matching ground truth
                    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if gt_idx in matched_gt:
                            continue
                        if pred_label != gt_label:
                            continue
                        
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    # Record prediction result
                    is_correct = (best_iou >= iou_threshold)
                    class_predictions[pred_label].append((pred_score, is_correct))
                    
                    if is_correct:
                        matched_gt.add(best_gt_idx)
                        overall_tp += 1
                    else:
                        overall_fp += 1
                
                # Count false negatives
                overall_fn += (len(gt_boxes) - len(matched_gt))
    
    return class_predictions, class_ground_truths, total_images, total_predictions, total_ground_truth, overall_tp, overall_fp, overall_fn

# --- Main ---
def main():
    print("="*70)
    print("MODEL EVALUATION - VALIDATION SET WITH mAP")
    print("="*70)
    
    # Load validation dataset
    val_dataset = CocoDataset(
        img_dir=os.path.join(dataset_path, "val"),
        annotation_file=os.path.join(dataset_path, "val_annotations.json"),
        image_size=IMAGE_SIZE
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Load model
    print("Loading model...")
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("✓ Model loaded\n")
    
    # Evaluate
    class_preds, class_gts, total_imgs, total_preds, total_gt, overall_tp, overall_fp, overall_fn = evaluate_model(
        model, val_loader, device, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
    )
    
    # Calculate per-class metrics and AP
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Total images: {total_imgs}")
    print(f"Total predictions: {total_preds}")
    print(f"Total ground truth: {total_gt}")
    print()
    
    print("Per-Class Metrics:")
    print("-" * 70)
    
    aps = []
    
    for class_id in sorted(class_gts.keys()):
        class_name = CATEGORIES.get(class_id, f'class_{class_id}')
        num_gt = class_gts[class_id]
        
        if class_id not in class_preds or len(class_preds[class_id]) == 0:
            print(f"\n{class_name.upper()}:")
            print(f"  Ground Truth: {num_gt}")
            print(f"  Predictions: 0")
            print(f"  AP: 0.000")
            aps.append(0.0)
            continue
        
        # Sort predictions by score
        predictions = sorted(class_preds[class_id], key=lambda x: x[0], reverse=True)
        
        # Calculate precision and recall at each prediction
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []
        
        for score, is_correct in predictions:
            if is_correct:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / num_gt if num_gt > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        
        # Calculate AP
        ap = calculate_ap(precisions, recalls)
        aps.append(ap)
        
        # Final precision and recall
        final_precision = precisions[-1] if len(precisions) > 0 else 0
        final_recall = recalls[-1] if len(recalls) > 0 else 0
        final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0
        
        print(f"\n{class_name.upper()}:")
        print(f"  Ground Truth: {num_gt}")
        print(f"  Predictions: {len(predictions)}")
        print(f"  Precision: {final_precision:.3f}")
        print(f"  Recall: {final_recall:.3f}")
        print(f"  F1-Score: {final_f1:.3f}")
        print(f"  AP (Average Precision): {ap:.3f}")
    
    # Calculate mAP
    mAP = np.mean(aps) if len(aps) > 0 else 0.0
    
    # Overall metrics
    print("\n" + "="*70)
    print("OVERALL METRICS")
    print("="*70)
    
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"Overall Precision: {overall_precision:.3f}")
    print(f"Overall Recall: {overall_recall:.3f}")
    print(f"Overall F1-Score: {overall_f1:.3f}")
    print(f"\nmAP@0.5 (mean Average Precision): {mAP:.3f}")
    print("="*70)
    
if __name__ == "__main__":
    main()