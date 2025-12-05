import json
import os
import cv2

base_path = "/courses/CS5330.202610/students/patel.harshil3/FinalProject"
json_path = os.path.join(base_path, "instances.json")
image_dir = os.path.join(base_path, "images")
out_dir = os.path.join(base_path, "results")
os.makedirs(out_dir, exist_ok=True)

# ----- Load JSON -----
print("Loading annotations from:", json_path)
with open(json_path, "r") as f:
    data = json.load(f)

print(f"Found {len(data['images'])} images")
print(f"Found {len(data['annotations'])} annotations")
print(f"Found {len(data['categories'])} categories")

# Print category names
print("\nCategories:")
for cat in data["categories"]:
    print(f"  {cat['id']}: {cat['name']}")

# ----- Build image_id → file_name mapping -----
image_map = {img["id"]: img["file_name"] for img in data["images"]}

# ----- Build annotation list per image -----
ann_per_image = {}
for ann in data["annotations"]:
    img_id = ann["image_id"]
    ann_per_image.setdefault(img_id, []).append(ann)


count = 0
max_visualize = 500

for img_id, file_name in image_map.items():
    if count >= max_visualize:
        break
    
    img_path = os.path.join(image_dir, file_name)
    if not os.path.exists(img_path):
        print(f"Missing: {img_path}")
        continue
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read: {img_path}")
        continue
    
    anns = ann_per_image.get(img_id, [])
    
    for ann in anns:
        x, y, w, h = ann["bbox"]
        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))
        
        # Draw bounding box (green)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        
        # Put category name
        cat_id = ann["category_id"]
        label = [c["name"] for c in data["categories"] if c["id"] == cat_id][0]
        cv2.putText(img, label, (int(x), int(y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    out_path = os.path.join(out_dir, file_name)
    cv2.imwrite(out_path, img)
    print(f"Saved ({count+1}/{max_visualize}): {out_path}")
    count += 1

print(f"\nDone! Visualized {count} images in: {out_dir}")