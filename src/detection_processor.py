# detection_processor.py

class DetectionProcessor:
    """
    Processes raw YOLO detections by filtering based on confidence or class
    and adding additional metadata like centroids.
    """

    def __init__(self, target_classes=None, confidence_threshold=0.5):
        """
        Args:
            target_classes (List[int], optional): List of class IDs to keep. 
                If empty or None, no class filtering is performed.
            confidence_threshold (float): Minimum confidence threshold to keep a detection.
        """
        self.target_classes = target_classes if target_classes is not None else []
        self.confidence_threshold = confidence_threshold

    def process_detections(self, detections):
        """
        Processes raw detections (list of dictionaries) from the YOLO model.

        Args:
            detections (List[Dict]): Each dictionary in this list has:
                {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "confidence": float,
                    "class_id": int
                }

        Returns:
            List[Dict]: Processed detections filtered by confidence and target_classes,
                        with an added "centroid" key.
        """
        filtered_detections = []

        for detection in detections:
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            class_id = detection["class_id"]

            # Unpack bounding box
            x_min, y_min, x_max, y_max = bbox

            # Filter by confidence and (optionally) class IDs
            if confidence >= self.confidence_threshold and \
               (not self.target_classes or class_id in self.target_classes):

                # Calculate centroid
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                # Build a processed detection dictionary
                processed_detection = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "class_id": class_id,
                    "centroid": (x_center, y_center),
                }

                filtered_detections.append(processed_detection)

        return filtered_detections
