from ultralytics import YOLO

model = YOLO("model/yolo11n.pt")

# # model.train(
# #     data="data.yaml",
# #     epochs=50,
# #     imgsz=640,
# #     batch=10
# # )

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=1280,
    batch=10,
    degrees=10,
    shear=1,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    lr0 = 0.001,
    lrf = 0.5,
    copy_paste = 0.2
)