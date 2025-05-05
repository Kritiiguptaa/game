from ultralytics import YOLO
model = YOLO("yolov8n.yaml")
results = model.train(data="C:/Isha/data.yaml", epochs=1)
