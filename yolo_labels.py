import os
import json
import cv2

def coco_to_yolo(coco_json_path, images_dir, labels_dir):
    # map counter for train and val images counts
    map_counter = 0
    img_counter = 0
    missing_images = 0

    # Create output directory if not present
    os.makedirs(labels_dir, exist_ok=True)

    # Load COCO JSON
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Map image_id -> filename (ONLY for images that exist in images_dir)
    image_map = {}
    for img in coco["images"]:
        img_path = os.path.join(images_dir, img["file_name"])
        if os.path.exists(img_path):
            image_map[img["id"]] = img["file_name"]
            map_counter += 1
        else:
            missing_images += 1

    # Group annotations per image (ONLY for images that exist)
    ann_map = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id in image_map:  # Only process if image exists
            if img_id not in ann_map:
                ann_map[img_id] = []
            ann_map[img_id].append(ann)

    # Process each image
    for img_id, file_name in image_map.items():
        img_path = os.path.join(images_dir, file_name)

        # Load image with OpenCV 
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load: {img_path}")
            continue
        
        img_counter += 1
        h, w = img.shape[:2]

        # Output label file
        label_path = os.path.join(labels_dir, file_name.replace(".jpg", ".txt")
                                                      .replace(".png", ".txt")
                                                      .replace(".jpeg", ".txt"))

        with open(label_path, "w") as lf:
            if img_id in ann_map:
                for ann in ann_map[img_id]:
                    # Extract COCO bbox
                    x, y, bw, bh = ann["bbox"]
                    category_id = ann["category_id"]

                    # Convert to YOLO normalized format
                    x_center = (x + bw / 2) / w
                    y_center = (y + bh / 2) / h
                    bw_norm = bw / w
                    bh_norm = bh / h

                    # Write: class x_center y_center width height
                    lf.write(f"{category_id} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

    print(f"Total images in JSON: {len(coco['images'])}")
    print(f"Images found in directory: {map_counter}")
    print(f"Images successfully processed: {img_counter}")
    print(f"Missing images: {missing_images}")


# -----------------------------
# RUN FOR TRAIN
# -----------------------------
print("Processing TRAIN set...")
coco_to_yolo(
    coco_json_path="instances.json",
    images_dir="dataset/train/images",
    labels_dir="dataset/train/labels"
)

# -----------------------------
# RUN FOR VAL
# -----------------------------
print("\nProcessing VAL set...")
coco_to_yolo(
    coco_json_path="instances.json",
    images_dir="dataset/val/images",
    labels_dir="dataset/val/labels"
)

