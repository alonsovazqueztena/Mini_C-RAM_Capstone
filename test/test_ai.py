# Alonso Vazquez Tena | STG-452: Capstone Project II | April 5, 2025
# This is my own code.

from ultralytics import YOLO # Import YOLO class for AI.

model = YOLO("..\src\drone_detector_12n.pt") # Create model instance.
results = model.predict("..\\test_images\drone_real_test_10.jpg", conf=0.5, imgsz=640, show=True, save=True) # Run inference (confidence, image size, display, save prediction).
print(results[0].boxes) # Print detected bounding boxes.