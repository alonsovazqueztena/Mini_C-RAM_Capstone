from ultralytics import YOLO
import numpy as np
import logging


class YOLOModelInterface:
    """
    Interface for YOLO model to run inference and process detections.
    """

    def __init__(self, model_path="yolo_epoch_100.pt", confidence_threshold=0.5):
        """
        Initializes the YOLO model interface.

        Args:
            model_path (str): Path to the YOLO model file.
            confidence_threshold (float): Minimum confidence score for detections (0-1).
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        # Load the YOLO model
        try:
            self.model = YOLO(self.model_path)
            logging.info(f"YOLO model loaded successfully from {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to load YOLO model from {self.model_path}: {e}")
            raise

    def predict(self, frame):
        """
        Runs inference on a single frame and extracts detections.

        Args:
            frame (np.ndarray): Input frame for YOLO.

        Returns:
            List[Dict]: List of detections containing bounding box, confidence, and class ID.
        """
        try:
            # Run inference
            results = self.model.predict(source=frame, imgsz=640, conf=self.confidence_threshold)

            # Process results
            detections = []
            for result in results:
                if result.boxes is not None:  # Check if boxes are available
                    for box in result.boxes:
                        # Extract bounding box, confidence, and class ID
                        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())  # Ensure class ID is an integer

                        # Append detection if confidence is above the threshold
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                "bbox": [x_min, y_min, x_max, y_max],
                                "confidence": confidence,
                                "class_id": class_id
                            })

            logging.info(f"Detections: {detections}")
            return detections

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return []

    def predict_batch(self, frames):
        """
        Runs inference on a batch of frames.

        Args:
            frames (List[np.ndarray]): List of input frames for YOLO.

        Returns:
            List[List[Dict]]: List of detections for each frame.
        """
        try:
            # Run batch inference
            results = self.model.predict(source=frames, imgsz=640, conf=self.confidence_threshold)

            all_detections = []
            for result in results:
                detections = []
                if result.boxes is not None:
                    for box in result.boxes:
                        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())

                        if confidence >= self.confidence_threshold:
                            detections.append({
                                "bbox": [x_min, y_min, x_max, y_max],
                                "confidence": confidence,
                                "class_id": class_id
                            })
                all_detections.append(detections)

            logging.info(f"Batch detections: {all_detections}")
            return all_detections

        except Exception as e:
            logging.error(f"Error during batch prediction: {e}")
            return []
