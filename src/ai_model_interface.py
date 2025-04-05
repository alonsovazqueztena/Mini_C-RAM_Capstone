# Alonso Vazquez Tena | STG-452: Capstone Project II | April 4, 2025
# Source: https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8

import logging # For logging errors and info.

from ultralytics import YOLO # YOLO model for object detection.

class AIModelInterface:
    """Interface for AI model inference and detection."""

    def __init__(self, model_path="drone_detector_12n.pt", confidence_threshold=0.5):
        """Initalize with model path and confidence threshold."""
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s") # Configure logging.
        try:
            self.model = YOLO(self.model_path) # Load AI model.
        except Exception as e:
            logging.error(f"Failed to load AI model from {self.model_path}: {e}")
            raise

    def predict(self, frame):
        """Runs inference on frame and extract detections."""
        try:
            results = self.model.predict(source=frame, imgsz=640, conf=self.confidence_threshold, half=True, verbose=False) # Run inference.
            detections = []
            for result in results:
                if result.boxes is not None: # If boxes exist.
                    for box in result.boxes:
                        x_min, y_min, x_max, y_max = box.xyxy[0].tolist() # Extract bounding box.
                        confidence = box.conf[0].item() # Get confidence score.
                        class_id = int(box.cls[0].item()) # Get class ID.
                        label = self.model.names[class_id] # Get label.
                        if confidence >= self.confidence_threshold: # Validate detection
                            detections.append({"bbox": [x_min, y_min, x_max, y_max], "confidence": confidence, "class_id": class_id, "label": label})
            return detections
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return []