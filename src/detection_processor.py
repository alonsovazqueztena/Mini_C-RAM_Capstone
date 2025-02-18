# Alonso Vazquez Tena
# STG-452: Capstone Project II
# February 3, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8
# (used as starter code for basic functionality).


# This class processes raw YOLO detections by 
# filtering based on confidence or class.
class DetectionProcessor:
    """Processes raw YOLO detections by filtering 
    based on confidence or class and adding additional 
    metadata like centroids."""

    # This method initializes the detection processor.
    def __init__(
            self, target_classes=None, 
            confidence_threshold=0.5):
        """ Initializes the detection processor.

        Keyword arguments:
        target_classes -- list of class IDs to keep. 
        confidence_threshold -- minimum confidence 
        threshold to keep a detection.
        """
        self.target_classes = target_classes if target_classes is not None else []
        self.confidence_threshold = confidence_threshold

    def process_detections(self, detections):
        """ Processes raw detections from the YOLO model.

        Keyword arguments:
        self -- instance of the detection processor,
        detections -- serves as a list of dictionaries that takes
        in the bounding box, confidence, and class ID of each detection.
        """

        # Our filtered detections will be stored here.
        filtered_detections = []

        # Each detection is to have a bounding box, confidence, and class ID.
        for detection in detections:
            bbox = detection[
                "bbox"
                ]
            confidence = detection[
                "confidence"
                ]
            class_id = detection[
                "class_id"
                ]

            # The bounding box is unpacked into its components.
            x_min, y_min, x_max, y_max = bbox

            # This filters by confidence and class IDs.
            if confidence >= self.confidence_threshold and \
               (not self.target_classes or class_id in self.target_classes):

                # The centroid of the bounding box is calculated.
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                # This builds a processed detection dictionary.
                processed_detection = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "class_id": class_id,
                    "centroid": (x_center, y_center),
                }

                # The processed detection is appended to the filtered detections.
                filtered_detections.append(processed_detection)

        return filtered_detections