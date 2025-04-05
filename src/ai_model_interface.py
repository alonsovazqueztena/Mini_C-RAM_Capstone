# Alonso Vazquez Tena | STG-452: Capstone Project II | April 4, 2025
# Source: https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8
# Daniel Saravia Source: https://grok.com/share/bGVnYWN5_52adc247-cde4-41e4-80bd-c70ef0c81dc9
# ai_model_interface.py
from ultralytics import YOLO
import torch

class AIModelInterface:
    """Optimized interface for YOLO drone detection."""

    def __init__(self, model_path="drone_detector_12n.pt", confidence_threshold=0.5):
        """Initialize YOLO model with minimal setup."""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict(self, frame):
        """Run inference and return minimal detection data."""
        results = self.model.predict(
            source=frame,
            imgsz=640,
            conf=self.confidence_threshold,
            half=True,
            device=self.device,
            verbose=False
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if self.model.names[int(box.cls[0])] != "drone":
                    continue
                bbox = box.xyxy[0].tolist()
                centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                detections.append({"bbox": bbox, "centroid": centroid})
        return detections