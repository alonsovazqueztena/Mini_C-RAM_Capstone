from ultralytics import YOLO
model = YOLO("yolo_epoch_100.pt")
results = model.predict("drone_test.jpg", conf=0.5, imgsz=640)
print(results[0].boxes)  # Check if bounding boxes exist
