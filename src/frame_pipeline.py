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
import pygame

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
        pygame.init()
        pygame.joystick.init()
        joystick = None
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
        else:
            logging.warning("No joystick detected. Falling back to keyboard control.")
        try:
            with self.video_stream as stream, concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                cv.namedWindow("AIegis Beam View", cv.WINDOW_NORMAL)
                cv.resizeWindow("AIegis Beam View", 600, 400)
                cv.setWindowProperty("AIegis Beam View", cv.WND_PROP_TOPMOST, 1)
                while True:
                    exit_pipeline = False
                    if joystick is not None:
                        for event in pygame.event.get():
                            if event.type == pygame.JOYBUTTONDOWN:
                                if event.button == 1:
                                    exit_pipeline = True
                                    break

                    if cv.waitKey(1) & 0xFF == ord('q'):
                        exit_pipeline = True
                    
                    if exit_pipeline:
                        break
                    
                    frame = stream.get_frame()
                    if frame is None:
                        logging.warning("No frame captured. Exiting the pipeline.")
                        break
                    future = executor.submit(self.ai_model_interface.predict, frame)

                    # Poll for exit events while waiting for inference to complete
                    while not future.done():
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            exit_pipeline = True
                            break
                        if joystick is not None:
                            for event in pygame.event.get():
                                if event.type == pygame.JOYBUTTONDOWN and event.button == 1:
                                    exit_pipeline = True
                                    break
                    # A short sleep could be added here to avoid busy waiting, e.g.:
                    # time.sleep(0.005)
                    if exit_pipeline:
                        break
                    detections = future.result()
                    tracked_objects = self.tracking_system.update(detections)
                    self.draw(frame, detections, tracked_objects)
                    cv.imshow("AIegis Beam View", frame)
        except Exception as e:
            logging.error(f"Error in FramePipeline run: {e}")
        finally:
            self.video_stream.release_stream()
            cv.destroyAllWindows()
            pygame.quit()