from ultralytics import YOLO
import numpy as mp

class YOLOModelInterface:

    # This initializes the YOLO model interface.
    def __init__(self, model_path = "yolo11n.pt", confidence_threshold = 5.5):

        # Arguments will include path to the YOLO model file
        # and minimum confidence score to consider a detection.
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        # The YOLO model is loaded here.
        self.model = YOLO(self.model_path)
        print(f"The YOLO model is loaded from {self.model_path}")

    def predict(self, frame):

        # This runs inferences on a preprocessed frame.
        # Arguments include preprocessed input frame for YOLO.
        # This returns a list of detections.
        results = self.model.predict(source = frame, imgsz = 640, 
                                 conf = self.confidence_threshold)
    
        # This is to extract detections.
        detections = []
        for result in results:
            for box in result.boxes:
            
                # This is to extract bounding box and confidence.
                x_min, y_min, x_max, y_max = box.xyxy.tolist()[0]
                confidence = box.conf.tolist()[0]
                class_id = box.cls.tolist()[0]

                # It is to append detection if confidence is above
                # the threshold.
                if confidence >= self.confidence_threshold:
                    detections.append([x_min, y_min, x_max, y_max, 
                                   confidence, class_id])
                
        return detections

