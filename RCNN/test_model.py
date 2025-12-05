import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2
import numpy as np
import os
import json
import torchvision.transforms as T

# --- Configuration ---
base_path = "/courses/CS5330.202610/students/patel.harshil3/FinalProject"
model_path = os.path.join(base_path, "models/best_model.pth")
val_images_dir = os.path.join(base_path, "dataset/val")
val_annotations = os.path.join(base_path, "dataset/val_annotations.json")
output_dir = os.path.join(base_path, "final_predictions")
os.makedirs(output_dir, exist_ok=True)

NUM_CLASSES = 4
IMAGE_SIZE = 960
CONFIDENCE_THRESHOLD = 0.5

CATEGORIES = {
    1: 'human',
    2: 'trafficsign',
    3: 'vehicle'
}

COLORS = {
    1: (0, 255, 0),      # Green for human
    2: (0, 0, 255),      # Red for traffic sign
    3: (255, 0, 0)       # Blue for vehicle
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Load Model ---
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

print("\nLoading model...")
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("✓ Model loaded\n")

# --- Load Annotations ---
with open(val_annotations, 'r') as f:
    coco_data = json.load(f)

img_to_anns = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    if img_id not in img_to_anns:
        img_to_anns[img_id] = []
    img_to_anns[img_id].append(ann)

img_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}

# --- Run Prediction ---
def predict_image(image_path, model, device, conf_threshold=0.5):
    img_pil = Image.open(image_path).convert("RGB")
    orig_width, orig_height = img_pil.size
    
    # Resize to model input size
    img_resized = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img_tensor = T.ToTensor()(img_resized).to(device)
    
    with torch.no_grad():
        predictions = model([img_tensor])[0]
    
    keep = predictions['scores'] > conf_threshold
    boxes = predictions['boxes'][keep].cpu().numpy()
    labels = predictions['labels'][keep].cpu().numpy()
    scores = predictions['scores'][keep].cpu().numpy()
    
    # Scale boxes back to original size
    scale_x = orig_width / IMAGE_SIZE
    scale_y = orig_height / IMAGE_SIZE
    
    scaled_boxes = []
    for box in boxes:
        scaled_box = [
            box[0] * scale_x,
            box[1] * scale_y,
            box[2] * scale_x,
            box[3] * scale_y
        ]
        scaled_boxes.append(scaled_box)
    
    return np.array(scaled_boxes), labels, scores

# --- Draw Boxes ---
def draw_boxes(img, boxes, labels, scores, title_prefix=""):
    img_copy = img.copy()
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        class_name = CATEGORIES.get(label, 'unknown')
        color = COLORS.get(label, (255, 255, 255))
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
        
        label_text = f"{class_name}: {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            img_copy, 
            (x1, y1 - text_height - 10), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        cv2.putText(
            img_copy, 
            label_text, 
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
    
    return img_copy

def draw_ground_truth(img, annotations):
    img_copy = img.copy()
    
    for ann in annotations:
        x, y, w, h = ann['bbox']
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        label = ann['category_id']
        
        class_name = CATEGORIES.get(label, 'unknown')
        color = COLORS.get(label, (255, 255, 255))
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
        
        label_text = class_name
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            img_copy, 
            (x1, y1 - text_height - 10), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        cv2.putText(
            img_copy, 
            label_text, 
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
    
    return img_copy

# --- Main ---
print("="*60)
print("CREATING VISUALIZATION - GROUND TRUTH vs PREDICTIONS")
print("="*60)

# Process first 10 validation images
val_images = sorted([f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png'))])
num_to_process = min(10, len(val_images))

for i, img_name in enumerate(val_images[:num_to_process]):
    img_path = os.path.join(val_images_dir, img_name)
    
    # Load image
    img = cv2.imread(img_path)
    
    # Get ground truth
    img_id = None
    for id, name in img_id_to_name.items():
        if name == img_name:
            img_id = id
            break
    
    ground_truth = img_to_anns.get(img_id, []) if img_id else []
    
    # Get predictions
    pred_boxes, pred_labels, pred_scores = predict_image(
        img_path, model, device, CONFIDENCE_THRESHOLD
    )
    
    # Draw ground truth and predictions
    img_gt = draw_ground_truth(img, ground_truth)
    img_pred = draw_boxes(img, pred_boxes, pred_labels, pred_scores)
    
    # Combine side by side
    combined = np.hstack([img_gt, img_pred])
    
    # Add text labels
    cv2.putText(combined, "GROUND TRUTH", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(combined, "PREDICTIONS", (img.shape[1] + 50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Save
    output_path = os.path.join(output_dir, f"comparison_{i+1:02d}_{img_name}")
    cv2.imwrite(output_path, combined)
    
    print(f"[{i+1}/{num_to_process}] {img_name}")
    print(f"  Ground Truth: {len(ground_truth)} objects")
    print(f"  Predictions: {len(pred_boxes)} objects")
    print(f"  Saved: {output_path}")
    print()