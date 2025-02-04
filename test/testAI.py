# Alonso Vazquez Tena
# STG-452: Capstone Project II
# February 2, 2025
# This is my own code.

from ultralytics import YOLO

model = YOLO(r"C:\Users\alons\.vscode\Mini_C-RAM_Capstone\Mini_C-RAM_Capstone\src\yolo_epoch_100.pt")
results = model.predict("drone_real_test_15.jpg", conf=0.5, imgsz=640, show=True, save=True)
print(results[0].boxes)  # Check if bounding boxes exist