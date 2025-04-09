# Alonso Vazquez Tena | STG-452: Capstone Project II | April 5, 2025
# Source: https://chatgpt.com/share/67a17189-ca30-800e-858d-aac289e6cb56, https://chatgpt.com/share/67d77b29-c824-800e-ab25-2cc850596046.
# Daniel Saravia Source: https://grok.com/share/bGVnYWN5_f72f2b0a-cc36-43d1-b32a-3b62ed45820a
# frame_pipeline.py
import concurrent.futures # For running tasks concurrently.
import logging # For logging messages.
from ai_model_interface import AIModelInterface # Import AI model interface.
from tracking_system import TrackingSystem # Import tracking system for object tracking.
from video_stream_manager import VideoStreamManager # Import video stream manager for capturing frames.
import cv2 as cv # OpenCV for image processing.
import os # For operating system functions.
from contextlib import redirect_stdout # For suppressing output during pygame import.
with redirect_stdout(open(os.devnull, 'w')):
    import pygame # Import pygame for joystick control.

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
        pygame.init() # Initialize pygame modules.
        pygame.joystick.init() # Initialize joystick support.
        joystick = None # Default joystick variable.
        if pygame.joystick.get_count() > 0: # Check joystick availability.
            joystick = pygame.joystick.Joystick(0) # Select first joystick.
            joystick.init() # Initialize joystick.
        else:
            logging.warning("No joystick detected. Falling back to keyboard control.")
        try:
            with self.video_stream as stream, concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor: # Open video stream and create thread pool.
                cv.namedWindow("AIegis Beam View", cv.WINDOW_NORMAL) # Create a resizable window.
                cv.resizeWindow("AIegis Beam View", 600, 400) # Resize window to 600x400 pixels.
                cv.setWindowProperty("AIegis Beam View", cv.WND_PROP_TOPMOST, 1) # Set window to always be on top.
                while True: # Main loop for processing frames. 
                    exit_pipeline = False # Flag to exit pipeline.
                    if joystick is not None: # Check if joystick exists.
                        for event in pygame.event.get(): # Iterate over pygame events.
                            if event.type == pygame.JOYBUTTONDOWN: # Check for joystick button press.
                                if event.button == 1: # IF button 1 pressed.
                                    exit_pipeline = True # Set exit flag.
                                    break

                    if cv.waitKey(1) & 0xFF == ord('q'): # Check if 'q' key is pressed.
                        exit_pipeline = True # Set exit flag.
                    
                    if exit_pipeline: # Exit main loop is flag is set.
                        break
                    
                    frame = stream.get_frame() # Capture frame from video stream.
                    if frame is None: # If frame capture fails.
                        logging.warning("No frame captured. Exiting the pipeline.")
                        break
                    future = executor.submit(self.ai_model_interface.predict, frame) # Submit AI prediction task asynchronously.

                    # Poll for exit events while waiting for inference to complete
                    while not future.done():
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            exit_pipeline = True
                            break
                        if joystick is not None: # If joystick exists, process events.
                            for event in pygame.event.get(): # Iterate over joystick events.
                                if event.type == pygame.JOYBUTTONDOWN and event.button == 1: # If joystick button 1 is pressed. 
                                    exit_pipeline = True # Set exit flag.
                                    break
                    if exit_pipeline: # If exit flag is set after inference.
                        break
                    detections = future.result() # Retrieve AI detection results.
                    tracked_objects = self.tracking_system.update(detections) # Update tracking with detections.
                    self.draw(frame, detections, tracked_objects) # Draw detections and tracking on frame.
                    cv.imshow("AIegis Beam View", frame) # Display processed frame.
        except Exception as e:
            logging.error(f"Error in FramePipeline run: {e}") # Log any exception encountered.
        finally:
            self.video_stream.release_stream() # Release video stream resources.
            cv.destroyAllWindows() # Close all OpenCV windows.
            pygame.quit() # Quit pygame modules.