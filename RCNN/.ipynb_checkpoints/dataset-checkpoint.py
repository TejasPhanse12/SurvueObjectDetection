import json
import os
import random
from shutil import copy2

# Set random seed for reproducibility
random.seed(42)

# --- Inputs ---
json_path = "instances.json"
image_dir = "results"

# --- Output directories ---
output_base = "dataset"
train_dir = os.path.join(output_base, "train")
val_dir = os.path.join(output_base, "val")

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

print("Loading annotations...")
with open(json_path, "r") as f:
    data = json.load(f)

print(f"Total images: {len(data['images'])}")
print(f"Total annotations: {len(data['annotations'])}")
print(f"Categories: {[cat['name'] for cat in data['categories']]}")

# --- Split images 80/20 ---
all_images = data['images'].copy()
random.shuffle(all_images)

split_idx = int(len(all_images) * 0.8)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

print(f"\nTrain images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")

# Create image_id sets for filtering
train_img_ids = {img['id'] for img in train_images}
val_img_ids = {img['id'] for img in val_images}

# --- Filter annotations for train/val ---
train_annotations = [ann for ann in data['annotations'] if ann['image_id'] in train_img_ids]
val_annotations = [ann for ann in data['annotations'] if ann['image_id'] in val_img_ids]

print(f"\nTrain annotations: {len(train_annotations)}")
print(f"Validation annotations: {len(val_annotations)}")

# --- Create train JSON ---
train_data = {
    'images': train_images,
    'annotations': train_annotations,
    'categories': data['categories']
}

train_json_path = os.path.join(output_base, "train_annotations.json")
with open(train_json_path, 'w') as f:
    json.dump(train_data, f)
print(f"\nSaved: {train_json_path}")

# --- Create validation JSON ---
val_data = {
    'images': val_images,
    'annotations': val_annotations,
    'categories': data['categories']
}

val_json_path = os.path.join(output_base, "val_annotations.json")
with open(val_json_path, 'w') as f:
    json.dump(val_data, f)
print(f"Saved: {val_json_path}")

# --- Copy images to train/val folders ---
print("\nCopying images to train folder...")
copied_train = 0
missing_train = []
for img in train_images:
    src = os.path.join(image_dir, img['file_name'])
    dst = os.path.join(train_dir, img['file_name'])
    if os.path.exists(src):
        copy2(src, dst)
        copied_train += 1
    else:
        missing_train.append(img['file_name'])

print(f"Copied {copied_train}/{len(train_images)} images")
if missing_train:
    print(f"Missing {len(missing_train)} images: {missing_train[:3]}...")

print("Copying images to validation folder...")
copied_val = 0
missing_val = []
for img in val_images:
    src = os.path.join(image_dir, img['file_name'])
    dst = os.path.join(val_dir, img['file_name'])
    if os.path.exists(src):
        copy2(src, dst)
        copied_val += 1
    else:
        missing_val.append(img['file_name'])

print(f"Copied {copied_val}/{len(val_images)} images")
if missing_val:
    print(f"Missing {len(missing_val)} images: {missing_val[:3]}...")

print("\n" + "="*50)
print("Dataset preparation complete!")
print("="*50)
print(f"\nDataset structure:")
print(f"  {output_base}/")
print(f"    ├── train/  ({copied_train} images)")
print(f"    ├── val/  ({copied_val} images)")
print(f"    ├── train_annotations.json")
print(f"    └── val_annotations.json")
print("\nReady for model training!")