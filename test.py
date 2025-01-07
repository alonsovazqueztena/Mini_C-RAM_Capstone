from ultralytics import YOLO
model = YOLO("yolo_epoch_100.pt")
results = model.predict("00000001_jpg.rf.2d8fb1dff4c5ebf31f0c0e406b2b6c21.jpg", conf=0.5, imgsz=640)
print(results[0].boxes)  # Check if bounding boxes exist
