# file: frame_pipeline.py

import cv2 as cv
import numpy as np
import logging

from video_stream_manager import VideoStreamManager
from frame_processor import FrameProcessor
from yolo_model_interface import YOLOModelInterface
from detection_processor import DetectionProcessor
from tracking_system import TrackingSystem  # <-- Import your tracker here

class FramePipeline:
    """
    A pipeline that captures frames from a video stream, processes them,
    runs YOLO + detection filtering, then tracks objects over time.
    """

    def __init__(
        self,
        capture_device=0,
        frame_width=640,
        frame_height=480,
        target_width=640,
        target_height=640,
        model_path="yolo_epoch_100.pt",
        confidence_threshold=0.5,
        detection_processor=None,
        tracking_system=None
    ):
        self.video_stream = VideoStreamManager(
            capture_device=capture_device, 
            frame_width=frame_width, 
            frame_height=frame_height
        )
        self.frame_processor = FrameProcessor(
            target_width=target_width, 
            target_height=target_height
        )
        self.yolo_model_interface = YOLOModelInterface(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        # Either use provided detection processor or create one with default params
        self.detection_processor = detection_processor or DetectionProcessor(
            target_classes=None, 
            confidence_threshold=confidence_threshold
        )
        # Either use provided tracking system or create a default
        self.tracking_system = tracking_system or TrackingSystem(
            max_disappeared=50, 
            max_distance=50
        )

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and centroids on the frame.
        """
        for det in detections:
            bbox = det["bbox"]
            confidence = det["confidence"]
            class_id = det["class_id"]
            x_min, y_min, x_max, y_max = map(int, bbox)

            # Draw bounding box
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Label
            label = f"ID:{class_id} Conf:{confidence:.2f}"
            cv.putText(
                frame, label, (x_min, y_min - 5), 
                cv.FONT_HERSHEY_SIMPLEX, 0.5, 
                (0, 255, 0), 1
            )
            
            # Draw centroid, if present
            if "centroid" in det:
                cx, cy = det["centroid"]
                cv.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    def draw_tracked_objects(self, frame, tracked_objects):
        """
        Draw tracked object IDs and centroids. Each tracked object
        is an entry: object_id -> { ... detection dict ... }.
        """
        for object_id, detection in tracked_objects.items():
            # detection should contain bbox, centroid, class_id, etc.
            bbox = detection["bbox"]
            x_min, y_min, x_max, y_max = map(int, bbox)
            cx, cy = detection["centroid"]

            # Draw bounding box in a different color for tracking
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Put text with the tracker ID
            cv.putText(
                frame, f"Obj {object_id}", (x_min, y_min - 10),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
            )

            # Draw the centroid
            cv.circle(frame, (int(cx), int(cy)), 4, (255, 0, 0), -1)

    def run(self):
        """
        Captures frames, runs preprocessing + YOLO + detection processing,
        then updates the tracking system and displays both YOLO detections
        and tracked objects in real time.
        """
        try:
            with self.video_stream as stream:
                logging.info("Starting the pipeline with tracking...")

                while True:
                    frame = stream.get_frame()
                    if frame is None:
                        logging.warning("No frame captured. Exiting the pipeline.")
                        break

                    # Preprocess frame for YOLO (resize, normalize)
                    processed_frame = self.frame_processor.preprocess_frame(frame)
                    # YOLO expects shape (H,W,3)
                    raw_detections = self.yolo_model_interface.predict(processed_frame[0])
                    
                    # Filter + add centroid
                    processed_detections = self.detection_processor.process_detections(raw_detections)
                    
                    # Update tracking system with the processed detections
                    tracked_objects = self.tracking_system.update(processed_detections)

                    # Draw YOLO bounding boxes in green (for debugging)
                    self.draw_detections(frame, processed_detections)
                    # Draw tracked objects in blue
                    self.draw_tracked_objects(frame, tracked_objects)

                    cv.imshow("Frame with Tracking", frame)

                    # Press 'q' to quit
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break

        except Exception as e:
            logging.error(f"Error in FramePipeline run: {e}")
        finally:
            logging.info("Releasing resources and closing windows.")
            self.video_stream.release_stream()
            cv.destroyAllWindows()
