import os

models = {
    "yolo11m": "runs/detect/train3/weights/best.pt",
    "yolo11s": "runs/detect/train4/weights/best.pt",
    "yolo11n": "runs/detect/train5/weights/best.pt",
    "yolo11n(imgsz=1280)" : "runs/detect/train6/weights/best.pt",
    "yolo11n(imgsz=1280)" : "runs/detect/train7/weights/best.pt",
    "yolo11n(imgsz=1280, NO-TS)" : "runs/detect/train8/weights/best.pt"
}

for name, path in models.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{name}: {size_mb:.2f} MB")
    else:
        print(f"{name}: file not found → {path}")
