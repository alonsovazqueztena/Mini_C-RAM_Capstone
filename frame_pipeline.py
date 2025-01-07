import cv2 as cv
import numpy as np
import logging

from video_stream_manager import VideoStreamManager
from frame_processor import FrameProcessor
from yolo_model_interface import YOLOModelInterface  # Make sure this module is in your PYTHONPATH

class FramePipeline:
    """
    A pipeline that captures frames from a video stream, processes them, and runs YOLO inference.
    """

    def __init__(
        self,
        capture_device=0,
        frame_width=1280,
        frame_height=720,
        target_width=640,
        target_height=640,
        model_path="yolo_epoch_100.pt",
        confidence_threshold=0.5
    ):
        """
        Initializes the FramePipeline.

        Args:
            capture_device (int): Index of the video capture device.
            frame_width (int): Desired width of the captured frames.
            frame_height (int): Desired height of the captured frames.
            target_width (int): Width to which frames will be resized for processing.
            target_height (int): Height to which frames will be resized for processing.
            model_path (str): Path to the YOLO model file.
            confidence_threshold (float): Minimum confidence for YOLO detections.
        """

        # Video stream manager for capturing frames
        self.video_stream = VideoStreamManager(capture_device, frame_width, frame_height)
        
        # Frame processor for resizing, normalizing, etc.
        self.frame_processor = FrameProcessor(target_width, target_height)

        # YOLO model interface for running inference
        self.yolo_model_interface = YOLOModelInterface(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )

    def draw_detections(self, frame, detections):
        """
        Draws detection bounding boxes on the original frame.

        Args:
            frame (np.ndarray): The original frame (BGR).
            detections (List[Dict]): List of detections returned by YOLO. 
                                     Each dict contains "bbox", "confidence", "class_id".
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

    def run(self):
        """
        Runs the frame pipeline:
        1. Opens the video stream.
        2. Continuously captures frames.
        3. Preprocesses the frame for YOLO.
        4. Performs YOLO inference.
        5. Draws detections on the original frame.
        6. Displays both the original and processed frames.
        7. Exits on 'q'.
        """
        try:
            with self.video_stream as stream:
                logging.info("Starting the frame processing pipeline.")

                while True:
                    # 1. Capture frame from the video stream
                    frame = stream.get_frame()
                    if frame is None:
                        logging.warning("No frame captured. Exiting the pipeline.")
                        break

                    # 2. Preprocess the frame for YOLO (resize, color convert, normalize)
                    #    This returns shape (1, target_height, target_width, 3)
                    processed_frame = self.frame_processor.preprocess_frame(frame)

                    # 3. Run YOLO inference
                    #    YOLO expects (H, W, 3) for a single frame, so pass processed_frame[0]
                    detections = self.yolo_model_interface.predict(processed_frame[0])

                    # 4. Draw detections on the original (BGR) frame for visualization
                    self.draw_detections(frame, detections)

                    # 5. Display the original frame with detections
                    cv.imshow("Original Frame with Detections", frame)

                    # 6. (Optional) display the processed frame for debugging
                    #    Convert from [0,1], RGB => [0,255], BGR
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

    # Example usage
    pipeline = FramePipeline(
        capture_device=0,
        frame_width=1280,
        frame_height=720,
        target_width=640,
        target_height=640,
        model_path="yolo_epoch_100.pt",
        confidence_threshold=0.5
    )
    pipeline.run()

