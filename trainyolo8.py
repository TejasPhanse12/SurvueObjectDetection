from ultralytics import YOLO

# model = YOLO('best (1).pt')

# model.train(
#     data='data.yaml',
#     epochs=100,
#     imgsz=1280,
#     batch=10,
#     patience=15,
#     mosaic=1.0,
#     copy_paste=0.2,   
#     degrees=10,       
#     project='runs/detect',
#     name='yolo11ntesting'
# )
model = YOLO('yolov8n.pt')

model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    patience=15,
    mosaic=1.0,
    mixup=0.3,        
    copy_paste=0.2,   
    degrees=10,       
    scale=0.7,        
    project='runs/detect',
    name='yolo8_640'
)

# from ultralytics import YOLO

# model = YOLO('yolov8m.pt')

# model.train(
#     data='yolo_dataset/data.yaml',
#     epochs=100,
#     imgsz=1280,
#     batch=8, 
#     box=10.0,
#     dfl=2.0,
#     lr0=0.001,
#     ltf = 0.04
#     mosaic=1.0,
#     mixup=0.2,
#     degrees=10,
#     fliplr=0.5,
    
#     project='runs/detect',
#     name='yolov8m_aug_tightbox'
# )
# from ultralytics import YOLO

# model = YOLO('yolov8m.pt')

# model.train(
#     data='yolo_dataset/data.yaml',
#     epochs=100,
#     imgsz=960,
#     batch=10,
#     lrf=0.5,
#     mosaic=1.0,
#     degrees=10,
#     fliplr=0.5,
#     flipud=0.5,
#     shear=1,
#     name='yolo_m'
# )
# from ultralytics import YOLO

# model = YOLO('yolov8s.pt')

# model.train(
#     data='yolo_dataset/data.yaml',
#     epochs=100,
#     imgsz=1280,
#     batch=8,
#     patience=20,
    
#     # Learning rate
#     lr0=0.01,
#     lrf=0.04,
    
#     # Augmentation
#     mosaic=1.0,
#     mixup=0.3,
#     copy_paste=0.2,
#     degrees=10,
#     translate=0.2,
#     scale=0.7,
#     shear=5,
#     flipud=0.1,
#     fliplr=0.5,
#     # hsv_h=0.02,
#     # hsv_s=0.8,
#     # hsv_v=0.5,
    
#     project='runs/detect',
#     name='yolov8s_1280_tuned'
# )