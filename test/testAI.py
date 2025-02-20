# Alonso Vazquez Tena
# STG-452: Capstone Project II
# February 3, 2025
# This is my own code.

# We import the YOLO model's class here.
from ultralytics import YOLO

# We create an instance of the trained YOLO model here.
model = YOLO("..\src\yolo_epoch_100.pt")

# We use the model to infer on a test image here.

# The confidence threshold is set to 50%,
# the image size is set to 640, and the processed 
# image is displayed and saved.
results = model.predict(
    "..\images\drone_real_test_10.jpg", conf=0.5, 
    imgsz=640, show=True, save=True)

# We print the bounding boxes of the detected objects here.
print(
    results[0].boxes
    )