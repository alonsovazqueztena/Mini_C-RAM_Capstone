# Alonso Vazquez Tena | STG-452: Capstone Project II | April 4, 2025
# Source: https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8

from ultralytics import YOLO # YOLO model for object detection.

class AIModelInterface:
    """Interface for AI model inference and detection."""

    def __init__(self, model_path="drone_detector_12n.pt", confidence_threshold=0.5, target_classes=None):
        """Initalize with model path and confidence threshold."""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes if target_classes is not None else ["drone"]

    def predict(self, frame):
        """Runs inference on frame and extract detections."""
        results = self.model.predict(source=frame, imgsz=640, conf=self.confidence_threshold, verbose=False) # Run inference.
        detections = [{"bbox": box.xyxy[0].tolist(), "confidence": box.conf[0].item(), "class_id": int(box.cls[0].item()), "label": self.model.names[int(box.cls[0].item())]} for result in results if result.boxes is not None for box in result.boxes]
        return [{"bbox": det["bbox"], "confidence": det["confidence"], "class_id": det["class_id"], "label": det["label"], "centroid": ((det["bbox"][0] + det["bbox"][2]) / 2, (det["bbox"][1] + det["bbox"][3]) / 2)} for det in detections if not self.target_classes or det["label"] in self.target_classes]