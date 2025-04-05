# Alonso Vazquez Tena | STG-452: Capstone Project II | April 5, 2025
# Source: https://chatgpt.com/share/67a17189-ca30-800e-858d-aac289e6cb56, https://chatgpt.com/share/67d77b29-c824-800e-ab25-2cc850596046.

import concurrent.futures # Asyncronous AI predictions using threads.
import logging # Logging for errors and status
from ai_model_interface import AIModelInterface # AI model interface
from tracking_system import TrackingSystem # Multi-object tracking
from video_stream_manager import VideoStreamManager # Video stream handling.
import cv2 as cv # OpenCV for computer vision.

class FramePipeline:
    """Pipeline: captures frames, run AI detections, track objects, display results."""

    def __init__(self):
        """Initialize video stream, run AI interface, and tracking system."""
        self.video_stream = VideoStreamManager() # Manage video stream.
        self.ai_model_interface = AIModelInterface() # Run AI detections.
        self.tracking_system = TrackingSystem() # Track detected objects.

    def draw(self, frame, detections, tracked_objects):
        """Draw detections and tracked objects with text on black background."""
        for det in detections:
            x_min, y_min, x_max, y_max = map(int, det["bbox"]) # Extract bounding box coordinates.
            label = f"drone {det['confidence']:.3f}" # Create label with confidence score.
            font = cv.FONT_HERSHEY_TRIPLEX
            font_scale = 1
            thickness = 2
            (text_width, text_height), baseline = cv.getTextSize(label, font, font_scale, thickness) # Calculate text dimensions.
            margin = 5
            cv.rectangle(frame, (x_min, y_min - text_height - baseline - margin), (x_min + text_width, y_min), (0, 0, 0), -1)  # Draw black background.
            cv.putText(frame, label, (x_min, y_min - margin), font, font_scale, (0, 255, 0), thickness)  # Draw green text label.
        for obj in tracked_objects.values():
            x_min, y_min, x_max, y_max = map(int, obj["bbox"]) # Extract tracked object box coordinates.
            cx, cy = map(int, obj["centroid"]) # Extract centroid coordinates.
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) # Draw green box around tracked object.
            cv.circle(frame, (cx, cy), 4, (0, 255, 0), -1)  # Draw green centroid dot.

    def run(self):
        """Captures frames, run detection and tracking, then display results."""
        try:
            with self.video_stream as stream, concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                cv.namedWindow("AIegis Beam View", cv.WINDOW_NORMAL) # Create window.
                cv.resizeWindow("AIegis Beam View", 800, 600) # Resize window.
                cv.setWindowProperty("AIegis Beam View", cv.WND_PROP_TOPMOST, 1) # Set window on top.
                while True:
                    frame = stream.get_frame() # Capture frame.
                    if frame is None:
                        logging.warning("No frame captured. Exiting the pipeline.")
                        break
                    future = executor.submit(self.ai_model_interface.predict, frame) # Run AI prediction asynchronously.
                    detections = future.result() # Get detection results.
                    tracked_objects = self.tracking_system.update(detections) # Update tracking.
                    self.draw(frame, detections, tracked_objects)
                    cv.imshow("AIegis Beam View", frame) # Display frame.
                    if cv.waitKey(1) & 0xFF == ord('q'): break # Quit on 'q' press.
        except Exception as e:
            logging.error(f"Error in FramePipeline run: {e}") # Log errors.
        finally:
            self.video_stream.release_stream() # Release video stream.
            cv.destroyAllWindows() # Close windows.