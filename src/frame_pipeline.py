# Alonso Vazquez Tena
# STG-452: Capstone Project II
# February 3, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a17189-ca30-800e-858d-aac289e6cb56
# (used as starter code for basic functionality).

# This project requires the usage of logs for the developer
# to understand the conditions of the system, whether
# an error has occurred or the execution of the class was a success.
import logging

# This project requires the usage of computer vision.

# In this case, OpenCV will be used.
import cv2 as cv

# All the classes are imported from the src folder
# to be used in the frame pipeline class.
from detection_processor import DetectionProcessor
from frame_processor import FrameProcessor
from tracking_system import TrackingSystem
from video_stream_manager import VideoStreamManager
from yolo_model_interface import YOLOModelInterface


# This class serves as a frame pipeline that 
# captures frames from a video stream,
# processes them, runs YOLO + detection filtering, 
# then tracks objects over time.
class FramePipeline:
    """A pipeline that captures frames from a video stream, processes them, 
    runs YOLO + detection filtering, then tracks objects over time."""

    # This method initializes the frame pipeline.
    def __init__(
        self,
        capture_device=0,
        frame_width=1920,
        frame_height=1080,
        target_width=1920,
        target_height=1080,
        model_path="yolo_epoch_100.pt",
        confidence_threshold=0.5,
        detection_processor=None,
        tracking_system=None
    ):
        """Initialize the frame pipeline.

        Keyword arguments:
        self -- instance of the frame pipeline,
        frame_width -- width of the video frame,
        frame_height -- height of the video frame,
        target_width -- target width for a preprocessed frame,
        target_height -- target height for a preprocessed frame,
        model_path -- path to the YOLO model file,
        confidence_threshold -- minimum confidence score for detections,
        detection_processor -- instance of DetectionProcessor 
        to filter detections,
        tracking_system -- instance of TrackingSystem to track objects.
        """
        # Set up the video stream manager.
        self.video_stream = VideoStreamManager(
            capture_device=capture_device, 
            frame_width=frame_width, 
            frame_height=frame_height
        )

        # Set up the frame processor.
        self.frame_processor = FrameProcessor(
            target_width=target_width, 
            target_height=target_height
        )

        # Set up the YOLO model interface.
        self.yolo_model_interface = YOLOModelInterface(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )

        # Use a provided detection processor or create 
        # one with default parameters.
        self.detection_processor = detection_processor or DetectionProcessor(
            target_classes=None, 
            confidence_threshold=confidence_threshold
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

        # Each detection is iterated over. Their bounding box,
        # confidence, and class ID are extracted.
        for det in detections:
            bbox = det[
                "bbox"]
            confidence = det[
                "confidence"]
            class_id = det[
                "class_id"]
            x_min, y_min, x_max, y_max = map(
                int, bbox)

            # The bounding box is drawn in green.
            cv.rectangle(
                frame, (x_min, y_min), 
                (x_max, y_max), (0, 255, 0), 2
                )

             # Prepare the label text.
            label = f"drone {confidence:.2f}"
            font = cv.FONT_HERSHEY_TRIPLEX
            font_scale = 2
            thickness = 4

            # Get the size of the text box and the baseline.
            (text_width, text_height), baseline = cv.getTextSize(label, font, font_scale, thickness)
            margin = 5

            # Draw a filled rectangle (black) as background for the label.
            cv.rectangle(frame,
                (x_min, y_min - text_height - baseline - margin),
                (x_min + text_width, y_min),
                (0, 0, 0), -1)

            # Draw the text (green) on top of the rectangle.
            cv.putText(frame, label, (x_min, y_min - margin), font, font_scale, (0, 255, 0), thickness)
            
            # This draws the centroid, if present.
            if "centroid" in det:
                cx, cy = det[
                    "centroid"]
                cv.circle(
                    frame, (int(cx), int(cy)), 
                    3, (0, 0, 255), -1
                    )

    # This method draws the tracked objects on the frame.
    def draw_tracked_objects(
            self, frame, 
            tracked_objects):
        """Draws tracked object IDs and centroids."""

        # Each tracked object is iterated over.
        for object_id, detection in tracked_objects.items():

            # Each detection is to have a boundary box, centroid, class_id, etc.
            bbox = detection[
                "bbox"]
            x_min, y_min, x_max, y_max = map(
                int, bbox)
            cx, cy = detection[
                "centroid"]

            # This draws bounding box in red for tracking.
            cv.rectangle(frame, (x_min, y_min), 
                         (x_max, y_max), (255, 0, 0), 2
                         )

            # This puts the text with the tracker ID.
            # cv.putText(
            #    frame, f"Drone {object_id}", (x_min, y_min - 10),
            #    cv.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 4
            #    )

            # This draws the centroid.
            cv.circle(frame, (int(cx), int(cy)), 
                      4, (255, 0, 0), -1
                      )

    # This method runs the frame pipeline.
    def run(
            self):
        """Captures frames, runs preprocessing + YOLO + detection processing,
        then updates the tracking system and displays both YOLO detections
        and tracked objects in real time."""

        try:
            # Start the video stream.
            with self.video_stream as stream:
                logging.info(
                    "Starting the pipeline with tracking..."
                    )
                
                cv.namedWindow("Frame with Tracking", cv.WINDOW_NORMAL)
                cv.resizeWindow("Frame with Tracking", 800, 600)  # Adjust width and height as needed

                # Run as long as frames are available.
                while True:
                    frame = stream.get_frame()
                    if frame is None:
                        logging.warning(
                            "No frame captured. Exiting the pipeline."
                            )
                        break

                    # This preprocesses frame for the YOLO model.
                    processed_frame = self.frame_processor.preprocess_frame(
                        frame)

                    # This predicts detections using the YOLO model.
                    raw_detections = self.yolo_model_interface.predict(
                        processed_frame[0])
                    
                    # This filters detections and adds centroids.
                    processed_detections = self.detection_processor.process_detections(
                        raw_detections)
                    
                    # This updates the tracking system with the processed detections.
                    tracked_objects = self.tracking_system.update(
                        processed_detections)

                    # This draws the YOLO bounding boxes.
                    self.draw_detections(
                        frame, processed_detections
                        )

                    # This draws tracked objects.
                    self.draw_tracked_objects(
                        frame, tracked_objects
                        )

                    # This displays the frame with tracking.
                    cv.imshow(
                        "Frame with Tracking", frame
                        )

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
            logging.info(
                "Releasing resources and closing windows."
                )
            self.video_stream.release_stream()
            cv.destroyAllWindows()