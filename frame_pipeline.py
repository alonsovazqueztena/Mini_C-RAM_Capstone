import cv2 as cv
import numpy as np
import logging

from video_stream_manager import VideoStreamManager
from frame_processor import FrameProcessor
from yolo_model_interface import YOLOModelInterface
from detection_processor import DetectionProcessor  # Make sure this is in your PYTHONPATH

class FramePipeline:
    """
    A pipeline that captures frames from a video stream, processes them, 
    runs YOLO inference, and applies detection filtering.
    """

    def __init__(
        self,
        capture_device=0,
        frame_width=640,            # Updated to 480p width
        frame_height=480,          # Updated to 480p height
        target_width=640, 
        target_height=640,
        model_path="yolo_epoch_100.pt",
        confidence_threshold=0.5,
        detection_processor=None
    ):
        """
        Initializes the FramePipeline.

        Args:
            capture_device (int): Index of the video capture device.
            frame_width (int): Desired width of the captured frames (640 for 480p).
            frame_height (int): Desired height of the captured frames (480 for 480p).
            target_width (int): Width to which frames will be resized for YOLO (often 640).
            target_height (int): Height to which frames will be resized for YOLO (often 640).
            model_path (str): Path to the YOLO model file.
            confidence_threshold (float): Minimum confidence for YOLO detections.
            detection_processor (DetectionProcessor, optional): 
                An instance of DetectionProcessor for filtering and processing detections.
        """

        # 1. Video stream manager (captures 640x480 frames)
        self.video_stream = VideoStreamManager(
            capture_device=capture_device, 
            frame_width=frame_width, 
            frame_height=frame_height
        )
        
        # 2. Frame processor for resizing, normalizing, etc. (to 640x640 by default)
        self.frame_processor = FrameProcessor(
            target_width=target_width, 
            target_height=target_height
        )

        # 3. YOLO model interface for running inference
        self.yolo_model_interface = YOLOModelInterface(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )

        # 4. Detection processor for filtering detections (if none provided, create a default)
        self.detection_processor = detection_processor or DetectionProcessor(
            target_classes=None,          # or specify classes you want to keep
            confidence_threshold=confidence_threshold
        )

    def draw_detections(self, frame, detections):
        """
        Draws detection bounding boxes on the original frame.

        Args:
            frame (np.ndarray): The original frame (BGR).
            detections (List[Dict]): List of detections returned by the DetectionProcessor. 
                Each dict contains "bbox", "confidence", "class_id", and possibly "centroid".
        """
        for det in detections:
            x_min, y_min, x_max, y_max = det["bbox"]
            confidence = det["confidence"]
            class_id = det["class_id"]

            # Convert float coords to int
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

            # Draw the bounding box
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Put a label with class_id and confidence
            label = f"ID:{class_id} Conf:{confidence:.2f}"
            cv.putText(
                frame, label, (x_min, y_min - 5), 
                cv.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 1
            )

            # (Optional) If you want to visualize centroid
            if "centroid" in det:
                cx, cy = det["centroid"]
                cv.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    def run(self):
        """
        Runs the frame pipeline:
        1. Opens the video stream.
        2. Continuously captures frames.
        3. Preprocesses the frame for YOLO.
        4. Performs YOLO inference.
        5. Processes detections with DetectionProcessor.
        6. Draws detections on the original frame.
        7. Displays frames and exits on 'q'.
        """
        try:
            with self.video_stream as stream:
                logging.info("Starting the frame processing pipeline at 640x480.")

                while True:
                    # 1. Capture frame from the video stream
                    frame = stream.get_frame()
                    if frame is None:
                        logging.warning("No frame captured. Exiting the pipeline.")
                        break

                    # 2. Preprocess the frame for YOLO (resize, normalize, etc.)
                    #    This returns shape (1, target_height, target_width, 3)
                    processed_frame = self.frame_processor.preprocess_frame(frame)

                    # 3. Run YOLO inference on the first (and only) preprocessed frame
                    #    YOLO expects (H, W, 3)
                    raw_detections = self.yolo_model_interface.predict(processed_frame[0])

                    # 4. Process detections (filter out low confidence or unwanted classes)
                    processed_detections = self.detection_processor.process_detections(raw_detections)

                    # 5. Draw detections on the original (BGR) frame for visualization
                    self.draw_detections(frame, processed_detections)

                    # 6. Display the original frame with detections
                    cv.imshow("Original Frame with Detections", frame)

                    # (Optional) Display the processed frame for debugging
                    processed_display_frame = (processed_frame[0] * 255).astype(np.uint8)
                    processed_display_frame = cv.cvtColor(processed_display_frame, cv.COLOR_RGB2BGR)
                    cv.imshow("Processed Frame (Debug)", processed_display_frame)

                    # 7. Break on 'q'
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        logging.info("Exiting the frame processing pipeline on user request.")
                        break

        except RuntimeError as e:
            logging.error(f"An error occurred: {e}")
        finally:
            logging.info("Releasing resources and closing windows.")
            self.video_stream.release_stream()
            cv.destroyAllWindows()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Example usage for 480p input, 640x640 YOLO inference, and confidence threshold of 0.5
    pipeline = FramePipeline(
        capture_device=0,
        frame_width=640,       # 480p width
        frame_height=480,      # 480p height
        target_width=640,      # YOLO input width
        target_height=640,     # YOLO input height
        model_path="yolo_epoch_100.pt",
        confidence_threshold=0.5
    )
    pipeline.run()
