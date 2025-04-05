# Alonso Vazquez Tena
# STG-452: Capstone Project II
# March 16, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a17189-ca30-800e-858d-aac289e6cb56
# (used as starter code for basic functionality) and
# https://chatgpt.com/share/67d77b29-c824-800e-ab25-2cc850596046
# (used to improve the frame pipeline further).

# This allows for the asynchronous execution of the AI model
# predictions using threads.
import concurrent.futures

# This project requires the usage of logs for the developer
# to understand the conditions of the system, whether
# an error has occurred or the execution of the class was a success.
import logging

import time

# All the classes are imported from the src folder
# to be used in the frame pipeline class.
from ai_model_interface import AIModelInterface
from detection_processor import DetectionProcessor
from tracking_system import TrackingSystem
from video_stream_manager import VideoStreamManager

# This project requires the usage of computer vision.

# In this case, OpenCV will be used.
import cv2 as cv


# This class serves as a frame pipeline that 
# captures frames from a video stream,
# processes them, runs AI + detection filtering, 
# then tracks objects over time.
class FramePipeline:
    """A pipeline that captures frames from a video stream, processes them, 
    runs YOLO + detection filtering, then tracks objects over time."""

    # This method initializes the frame pipeline.
    def __init__(
        self,
        model_path="drone_detector_12n.pt",
        confidence_threshold=0.5,
        detection_processor=None,
        tracking_system=None
    ):
        """Initialize the frame pipeline.

        Keyword arguments:
        self -- instance of the frame pipeline,
        target_width -- target width for a preprocessed frame,
        target_height -- target height for a preprocessed frame,
        model_path -- path to the AI model file,
        confidence_threshold -- minimum confidence score for detections,
        detection_processor -- instance of DetectionProcessor 
        to filter detections,
        tracking_system -- instance of TrackingSystem to track objects.
        """
        # Set up the video stream manager.
        self.video_stream = VideoStreamManager()

        # Set up the AI model interface.
        self.ai_model_interface = AIModelInterface(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )

        # Use a provided detection processor or create 
        # one with default parameters.
        self.detection_processor = detection_processor or DetectionProcessor(
            target_classes=None
        )

        # Use a provided tracking system or create one 
        # with default parameters.
        self.tracking_system = tracking_system or TrackingSystem(
            max_disappeared=50, 
            max_distance=50
        )

    # This method draws the detections on the frame.
    def draw_detections(
            self, frame, 
            detections):
        """Draw bounding boxes and centroids on the frame."""

        # Each detection is iterated over. Their bounding box
        # and confidence are extracted.
        for det in detections:
            bbox = det[
                "bbox"]
            confidence = det[
                "confidence"]
            x_min, y_min, x_max, y_max = map(
                int, bbox)

            # The label is prepared with the confidence.
            label = f"drone {confidence:.3f}"
            font = cv.FONT_HERSHEY_TRIPLEX
            font_scale = 2
            thickness = 4

            # We get the size of the text box and the baseline for
            # the background.

            # A black background is drawn as background for the label.
            (text_width, text_height), baseline = cv.getTextSize(
                label, font, 
                font_scale, thickness)
            margin = 5
            cv.rectangle(frame,
                (x_min, y_min - text_height - baseline - margin),
                (x_min + text_width, y_min),
                (0, 0, 0), -1)

            # The text box is drawn in front of the rectangle.
            cv.putText(frame, label, 
                       (x_min, y_min - margin), 
                       font, font_scale, 
                       (0, 255, 0), thickness
                       )

    # This method draws the tracked objects on the frame.
    def draw_tracked_objects(
            self, frame, 
            tracked_objects):
        """Draws tracked object IDs and centroids."""

        # Each tracked object is iterated over.
        for detection in tracked_objects.values():

            # Each detection is to have a boundary box and centroid.
            bbox = detection[
                "bbox"]
            x_min, y_min, x_max, y_max = map(
                int, bbox)
            cx, cy = detection[
                "centroid"]

            # This draws bounding box in green for tracking.
            cv.rectangle(frame, (x_min, y_min), 
                         (x_max, y_max), (0, 255, 0), 2
                         )

            # This draws the centroid.
            cv.circle(frame, (int(cx), int(cy)), 
                      4, (0, 255, 0), -1
                      )

    # This method runs the frame pipeline.
    def run(
            self):
        """Captures frames, runs preprocessing + AI + detection processing,
        then updates the tracking system and displays both AI detections
        and tracked objects in real time."""

        try:
            # Start the video stream and create a thread
            # pool for concurrent execution of AI model predictions.
            with self.video_stream as stream, concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                
                # The window view to see the program execution
                # is through here.

                # The window is to be small enough for the user to
                # see and is meant to automatically popup.
                cv.namedWindow("AIegis Beam View", cv.WINDOW_NORMAL)
                cv.resizeWindow("AIegis Beam View", 800, 600)
                cv.setWindowProperty("AIegis Beam View", cv.WND_PROP_TOPMOST, 1)

                # Run as long as frames are available.
                while True:

                    start_time = time.time()

                    frame = stream.get_frame()
                    if frame is None:
                        logging.warning(
                            "No frame captured. Exiting the pipeline."
                            )
                        break

                    capture_time = time.time()
                    
                    # This runs the prediction in a separate thread.
                    future = executor.submit(
                        self.ai_model_interface.predict, 
                        frame)
                    prediction_time = time.time()

                    # The results of the prediction are retrieved.
                    raw_detections = future.result()
                    
                    # This filters detections and adds centroids.
                    processed_detections = self.detection_processor.process_detections(
                        raw_detections)
                    
                    detection_process_time = time.time()
                    
                    # This updates the tracking system with the processed detections.
                    tracked_objects = self.tracking_system.update(
                        processed_detections)
                    
                    tracking_time = time.time()

                    # This draws the YOLO bounding boxes.
                    self.draw_detections(
                        frame, processed_detections
                        )

                    # This draws tracked objects.
                    self.draw_tracked_objects(
                        frame, tracked_objects
                        )
                    
                    pipeline_time = time.time()

                    # This displays the frame with tracking.
                    cv.imshow(
                        "AIegis Beam View", frame
                        )
                    
                    end_time = time.time()

                    # Calculate latencies
                    capture_latency = capture_time - start_time
                    prediction_latency = prediction_time - capture_time
                    detection_process_latency = detection_process_time - prediction_time
                    tracking_latency = tracking_time - detection_process_time
                    pipeline_latency = pipeline_time - tracking_time
                    total_latency = end_time - start_time

                    # Log the latency values
                    logging.info(f"Capture latency: {capture_latency:.6f} sec")
                    logging.info(f"Prediction latency: {prediction_latency:.6f} sec")
                    logging.info(f"Detection Process latency: {detection_process_latency:.6f} sec")
                    logging.info(f"Tracking latency: {tracking_latency:.6f} sec")
                    logging.info(f"Pipeline latency: {pipeline_latency:.6f} sec")
                    logging.info(f"Total pipeline latency: {total_latency:.6f} sec")


                    # This handles the button 'q' to quit.
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
        
        # This handles exceptions and logs them.
        except Exception as e:
            logging.error(
                f"Error in FramePipeline run: {e}"
                )
            
        # This ensures that resources are released and windows are closed.
        finally:
            self.video_stream.release_stream()
            cv.destroyAllWindows()
