class DetectionProcessor:

    # This initializes the detection processor.
    # Arguments for this include a list of class IDs to process
    # and minimum confidence score to keep a detection.
    def __init__(self, target_classes = None, confidence_threshold = 0.5):

        self.target_classes = target_classes if target_classes is not None else []
        self.confidence_threshold = confidence_threshold

    # This processes raw detections from the YOLO model.
    # Arguments include the list of raw detections from YOLO.
    # This is to return a list of processed detections with metadata.
    def process_detections(self, detections):

        filtered_detections = []

        for detection in detections:
            x_min, y_min, x_max, y_max, confidence, class_id = detection

            # This is to filter by confidence and class.
            if confidence >= self.confidence_threshold and (not self.target_classes or class_id in self.target_classes):

                # This is to calculate centroid of the bounding box.
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                # This is to format detection.
                processed_detection = {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "confidence": confidence,
                    "class_id": class_id,
                    "centroid": (x_center, y_center)
                }

                filtered_detections.append(processed_detection)

        return filtered_detections
