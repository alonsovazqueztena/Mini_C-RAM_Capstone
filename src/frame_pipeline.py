# Alonso Vazquez Tena | STG-452: Capstone Project II | April 5, 2025
# Source: https://chatgpt.com/share/67a17189-ca30-800e-858d-aac289e6cb56, https://chatgpt.com/share/67d77b29-c824-800e-ab25-2cc850596046.
# Daniel Saravia Source: https://grok.com/share/bGVnYWN5_f72f2b0a-cc36-43d1-b32a-3b62ed45820a
# frame_pipeline.py
import concurrent.futures
import logging
from ai_model_interface import AIModelInterface
from tracking_system import TrackingSystem
from video_stream_manager import VideoStreamManager
import cv2 as cv

class FramePipeline:
    """Pipeline: captures frames, runs AI detections, tracks objects, displays results."""

    def __init__(self, model_path="drone_detector_12n.pt", confidence_threshold=0.5):
        """Initialize video stream, AI interface, and tracking system with optional parameters."""
        self.video_stream = VideoStreamManager()
        self.ai_model_interface = AIModelInterface(model_path, confidence_threshold)
        self.tracking_system = TrackingSystem()

    def draw(self, frame, detections, tracked_objects):
        """Draw detections and tracked objects using minimal format (bbox and centroid only)."""
        for det in detections:
            x_min, y_min, x_max, y_max = map(int, det["bbox"])
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        for obj in tracked_objects.values():
            x_min, y_min, x_max, y_max = map(int, obj["bbox"])
            cx, cy = map(int, obj["centroid"])
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    def run(self):
        """Captures frames, runs detection and tracking, then displays results."""
        try:
            with self.video_stream as stream, concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                cv.namedWindow("AIegis Beam View", cv.WINDOW_NORMAL)
                cv.resizeWindow("AIegis Beam View", 800, 600)
                cv.setWindowProperty("AIegis Beam View", cv.WND_PROP_TOPMOST, 1)
                while True:
                    frame = stream.get_frame()
                    if frame is None:
                        logging.warning("No frame captured. Exiting the pipeline.")
                        break
                    future = executor.submit(self.ai_model_interface.predict, frame)
                    detections = future.result()
                    tracked_objects = self.tracking_system.update(detections)
                    self.draw(frame, detections, tracked_objects)
                    cv.imshow("AIegis Beam View", frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
        except Exception as e:
            logging.error(f"Error in FramePipeline run: {e}")
        finally:
            self.video_stream.release_stream()
            cv.destroyAllWindows()