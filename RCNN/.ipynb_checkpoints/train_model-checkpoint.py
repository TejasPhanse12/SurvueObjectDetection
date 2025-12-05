import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import csv
import time
import numpy as np
from collections import defaultdict

# --- Configuration ---
base_path = "/courses/CS5330.202610/students/patel.harshil3/FinalProject"
dataset_path = os.path.join(base_path, "dataset")
model_save_path = os.path.join(base_path, "models")
os.makedirs(model_save_path, exist_ok=True)

csv_log_path = os.path.join(base_path, "training_metrics.csv")

# Training parameters
NUM_CLASSES = 4  
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.005
NUM_WORKERS = 4
IMAGE_SIZE = 960
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Dataset Class ---
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
        
        # Resize image
        img_resized = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Scale boxes
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
    model = fasterrcnn_resnet50_fpn(pretrained=True)
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

# --- Validation Metrics ---
def evaluate_epoch(model, data_loader, device, conf_threshold=0.5, iou_threshold=0.5):
    """Quick evaluation for precision, recall during training"""
    model.eval()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)
            
            for pred, target in zip(predictions, targets):
                keep = pred['scores'] > conf_threshold
                pred_boxes = pred['boxes'][keep].cpu().numpy()
                pred_labels = pred['labels'][keep].cpu().numpy()
                
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                matched_gt = set()
                
                for pred_box, pred_label in zip(pred_boxes, pred_labels):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if gt_idx in matched_gt or pred_label != gt_label:
                            continue
                        
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= iou_threshold:
                        total_tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        total_fp += 1
                
                total_fn += (len(gt_boxes) - len(matched_gt))
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    model.train()
    return precision, recall, f1

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_objectness = 0
    loss_rpn_box = 0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Track individual losses
        for key, value in loss_dict.items():
            if 'loss_classifier' in key:
                loss_classifier += value.item()
            elif 'loss_box_reg' in key:
                loss_box_reg += value.item()
            elif 'loss_objectness' in key:
                loss_objectness += value.item()
            elif 'loss_rpn_box_reg' in key:
                loss_rpn_box += value.item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        progress_bar.set_postfix({'loss': losses.item()})
    
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    avg_loss_classifier = loss_classifier / num_batches
    avg_loss_box_reg = loss_box_reg / num_batches
    avg_loss_objectness = loss_objectness / num_batches
    avg_loss_rpn_box = loss_rpn_box / num_batches
    
    return avg_loss, avg_loss_classifier, avg_loss_box_reg, avg_loss_objectness, avg_loss_rpn_box

def main():
    print("="*60)
    print("FASTER R-CNN TRAINING WITH METRIC LOGGING")
    print("="*60)
    
    print("\nLoading datasets...")
    train_dataset = CocoDataset(
        img_dir=os.path.join(dataset_path, "train"),
        annotation_file=os.path.join(dataset_path, "train_annotations.json"),
        image_size=IMAGE_SIZE
    )
    
    val_dataset = CocoDataset(
        img_dir=os.path.join(dataset_path, "val"),
        annotation_file=os.path.join(dataset_path, "val_annotations.json"),
        image_size=IMAGE_SIZE
    )
    
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    print(f"\nCreating model...")
    model = get_model(NUM_CLASSES)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    
    # Initialize CSV file
    with open(csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Epoch', 'Time(s)', 'Total_Loss', 'Classifier_Loss', 
            'Box_Reg_Loss', 'Objectness_Loss', 'RPN_Box_Loss',
            'Val_Precision', 'Val_Recall', 'Val_F1', 'Val_mAP50'
        ])
    
    print(f"✓ CSV log initialized: {csv_log_path}")
    print("\nStarting training...\n")
    
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Training
        avg_loss, loss_cls, loss_box, loss_obj, loss_rpn = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )
        
        # Validation metrics
        print(f"\nEvaluating epoch {epoch+1}...")
        val_precision, val_recall, val_f1 = evaluate_epoch(
            model, val_loader, device, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
        )
        
        # Approximate mAP as F1 score (quick estimate)
        # For true mAP, run full evaluation after training
        val_map50 = val_f1  # Approximation
        
        epoch_time = time.time() - epoch_start_time
        
        # Print metrics
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Time: {epoch_time:.1f}s")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Losses - Cls: {loss_cls:.4f}, Box: {loss_box:.4f}, "
              f"Obj: {loss_obj:.4f}, RPN: {loss_rpn:.4f}")
        print(f"  Val Metrics - Precision: {val_precision:.3f}, "
              f"Recall: {val_recall:.3f}, F1: {val_f1:.3f}")
        
        # Log to CSV
        with open(csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{epoch_time:.2f}",
                f"{avg_loss:.4f}",
                f"{loss_cls:.4f}",
                f"{loss_box:.4f}",
                f"{loss_obj:.4f}",
                f"{loss_rpn:.4f}",
                f"{val_precision:.4f}",
                f"{val_recall:.4f}",
                f"{val_f1:.4f}",
                f"{val_map50:.4f}"
            ])
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(model_save_path, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved best model with loss: {best_loss:.4f}")
        
        print()
    
    # Save final model
    final_path = os.path.join(model_save_path, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best model: {save_path}")
    print(f"Final model: {final_path}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Training log: {csv_log_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
    
# def train_one_epoch(model, optimizer, data_loader, device, epoch):
#     model.train()
#     total_loss = 0
    
#     progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
#     for images, targets in progress_bar:
#         images = [img.to(device) for img in images]
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
        
#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()
        
#         total_loss += losses.item()
#         progress_bar.set_postfix({'loss': losses.item()})
    
#     avg_loss = total_loss / len(data_loader)
#     print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
#     return avg_loss

# def main():
#     print("="*60)
#     print("FASTER R-CNN TRAINING")
#     print("="*60)
    
#     print("\nLoading datasets...")
#     train_dataset = CocoDataset(
#         img_dir=os.path.join(dataset_path, "train"),
#         annotation_file=os.path.join(dataset_path, "train_annotations.json"),
#         image_size=IMAGE_SIZE
#     )
    
#     val_dataset = CocoDataset(
#         img_dir=os.path.join(dataset_path, "val"),
#         annotation_file=os.path.join(dataset_path, "val_annotations.json"),
#         image_size=IMAGE_SIZE
#     )
    
#     print(f"Train dataset: {len(train_dataset)} images")
#     print(f"Val dataset: {len(val_dataset)} images")
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=NUM_WORKERS,
#         collate_fn=collate_fn
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         collate_fn=collate_fn
#     )
    
#     print(f"\nCreating model...")
#     model = get_model(NUM_CLASSES)
#     model.to(device)
    
#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    
#     print("\nStarting training...\n")
#     best_loss = float('inf')
    
#     for epoch in range(NUM_EPOCHS):
#         train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
#         if train_loss < best_loss:
#             best_loss = train_loss
#             save_path = os.path.join(model_save_path, "best_model.pth")
#             torch.save(model.state_dict(), save_path)
#             print(f"✓ Saved best model with loss: {best_loss:.4f}")
        
#         print()
    
#     final_path = os.path.join(model_save_path, "final_model.pth")
#     torch.save(model.state_dict(), final_path)
    
#     print(f"\n{'='*60}")
#     print(f"Training complete!")
#     print(f"Best model: {save_path}")
#     print(f"Final model: {final_path}")
#     print(f"Best loss: {best_loss:.4f}")
#     print(f"{'='*60}")

# if __name__ == "__main__":
#     main()