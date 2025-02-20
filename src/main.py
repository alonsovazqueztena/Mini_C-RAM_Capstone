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
from frame_pipeline import FramePipeline
from frame_processor import FrameProcessor
from tracking_system import TrackingSystem
from video_stream_manager import VideoStreamManager
from yolo_model_interface import YOLOModelInterface

# This method tests the video stream manager 
# by attempting to capture a single frame.
def test_video_stream_manager():
    """Test the VideoStreamManager by attempting 
    to capture a single frame."""

    logging.info(
        "Testing VideoStreamManager..."
        )
    try:
        # This initializes the video stream manager.
        video_stream = VideoStreamManager(
            capture_device=0, frame_width=1920, 
            frame_height=1080)
        
        # This captures a single frame.
        with video_stream as stream:
            frame = stream.get_frame()

            # If the frame is empty, an error is raised.
            if frame is None:
                raise RuntimeError(
                    "Failed to capture frame in VideoStreamManager test."
                    )
            
            # If the frame is captured successfully, a log is created.
            logging.info(
                f"Frame captured successfully with shape: {frame.shape}"
                )
            
    # If an exception is raised, the error is logged.
    except Exception as e:
        logging.error(
            f"VideoStreamManager test failed: {e}"
            )

# This method tests the frame processor by processing a dummy image.
def test_frame_processor():
    """Test the FrameProcessor by processing a dummy image."""

    logging.info(
        "Testing FrameProcessor..."
        )
    try:
        # This initializes the frame processor.
        processor = FrameProcessor(
            target_width=1920, target_height=1080)

        # This loads a dummy frame for testing.
        dummy_frame = cv.imread(
            "../images/drone_mock_test_1.jpg")
        
        # If the dummy frame is empty or cannot be found, an error is raised.
        if dummy_frame is None:
            raise ValueError(
                "Failed to load test image. Provide a valid image path."
                )

        # This preprocesses the dummy frame.
        processed_frame = processor.preprocess_frame(
            dummy_frame)
        
        # If the processed frame is empty, an error is raised.
        if processed_frame is None or processed_frame.size == 0:
            raise RuntimeError(
                "Preprocessed frame is None or empty."
                )

        # If the processed frame is successfully created, a log is created.
        logging.info(
            f"Preprocessed frame shape: {processed_frame.shape}"
            )
        
    # If an exception is raised, the error is logged.
    except Exception as e:
        logging.error(
            f"FrameProcessor test failed: {e}"
            )

# This method tests the YOLO model interface by 
# running inference on a sample image.
def test_yolo_model_interface():
    """Test the YOLOModelInterface by 
    running inference on a sample image."""

    logging.info("Testing YOLOModelInterface...")
    try:
        # This initializes the YOLO model interface.
        yolo_interface = YOLOModelInterface(
            model_path="yolo_epoch_100.pt", 
            confidence_threshold=0.5)

        # A test image is loaded for YOLO.
        test_img = cv.imread(
            "../images/drone_mock_test_1.jpg")
        
        # If the test image is empty or cannot be found, an error is raised.
        if test_img is None:
            raise ValueError(
                "Failed to load test image for YOLO. Provide a valid image path."
                )

        # This runs inference on the test image.
        detections = yolo_interface.predict(
            test_img)
        
        # A log is created with the raw YOLO detections.
        logging.info(
            f"Raw YOLO detections: {detections}"
            )
        
    # If an exception is raised, the error is logged.
    except Exception as e:
        logging.error(
            f"YOLOModelInterface test failed: {e}"
            )

# This method tests the detection processor by 
# running YOLO on a sample image
# and then processing the raw detections.
def test_detection_processor():
    """ Test the DetectionProcessor by running YOLO on a sample image
    and then processing the raw detections."""

    logging.info("Testing DetectionProcessor...")
    try:

        # The YOLO model interface is initialized.
        yolo_interface = YOLOModelInterface(
            model_path="yolo_epoch_100.pt", 
            confidence_threshold=0.5)

        # The test image is loaded for YOLO.
        test_img = cv.imread(
            "../images/drone_mock_test_1.jpg")
        
        # If the test image is empty or cannot be found, an error is raised.
        if test_img is None:
            raise ValueError(
                "Failed to load test image for YOLO. Provide a valid image path."
                )

        # This runs inference on the test image.
        raw_detections = yolo_interface.predict(
            test_img)
        
        # A log is created with the raw YOLO detections.
        logging.info(
            f"Raw detections from YOLO: {raw_detections}"
            )

        # The detection processor is initialized.
        detection_processor = DetectionProcessor(
            target_classes=None,
            confidence_threshold=0.5
        )

        # Detections are processed by the detection processor.
        processed_detections = detection_processor.process_detections(
            raw_detections)
        logging.info(
            f"Processed detections: {processed_detections}"
            )

    # If an exception is raised, the error is logged.
    except Exception as e:
        logging.error(
            f"DetectionProcessor test failed: {e}"
            )

# This method tests the frame pipeline by 
# running a continuous video stream at 640x480,
# processing each frame, and running YOLO detection.
def test_frame_pipeline():
    """Test the FramePipeline by 
    running a continuous video stream at 640x480,
    processing each frame, and running YOLO detection."""

    logging.info(
        "Testing FramePipeline..."
        )
    try:

        # The frame pipeline is initialized.
        pipeline = FramePipeline(
            capture_device=0, 
            frame_width=1920, 
            frame_height=1080, 
            target_width=1920, 
            target_height=1080,
            model_path="yolo_epoch_100.pt",
            confidence_threshold=0.5
        )

        # The frame pipeline is run and stops when the user quits.
        pipeline.run()
        logging.info(
            "FramePipeline test completed successfully."
            )
        
    # If an exception is raised, the error is logged.
    except Exception as e:
        logging.error(
            f"FramePipeline test failed: {e}"
            )

# This method tests the frame pipeline by 
# running a continuous video stream at 640x480,
# and includes detection and tracking.
def test_frame_pipeline_with_tracking():
    """Test the FramePipeline by running a 
    continuous video stream at 640x480,
    including detection and tracking."""

    logging.info(
        "Testing FramePipeline with TrackingSystem..."
        )
    try:

        # Initialize the TrackingSystem for the FramePipeline.
        tracker = TrackingSystem(
            max_disappeared=50, max_distance=50)

        # Initialize the FramePipeline and pass the tracker.
        pipeline = FramePipeline(
            capture_device=0, 
            frame_width=1920, 
            frame_height=1080, 
            target_width=1920, 
            target_height=1080,
            model_path="yolo_epoch_100.pt",
            confidence_threshold=0.5,
            detection_processor=None,
            tracking_system=tracker
        )

        # The frame pipeline is run and stops when the user quits.
        pipeline.run()
        logging.info(
            "FramePipeline with TrackingSystem test completed successfully."
            )
        
    # If an exception is raised, the error is logged.
    except Exception as e:
        logging.error(
            f"FramePipeline with TrackingSystem test failed: {e}"
            )

# This method is the main entry point for testing all modules in a single script.
def main():
    """Main entry point for testing all modules in a single script."""

    # This configures logging for the entire script.
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
        )
    logging.info(
        "Starting all module tests..."
        )

    # This tests the VideoStreamManager (basic frame capture).
    test_video_stream_manager()

    # This tests the FrameProcessor (image preprocessing).
    test_frame_processor()

    # This tests the YOLOModelInterface (model loading and inference).
    test_yolo_model_interface()

    # This tests the DetectionProcessor (filter & add centroids).
    test_detection_processor()

    # This tests the FramePipeline 
    # (real-time video + YOLO detection + 640x480).
    test_frame_pipeline()

    # This tests the FramePipeline with tracking 
    # (real-time detection + tracking).
    test_frame_pipeline_with_tracking()

    logging.info(
        "All module tests completed."
        )

# This runs the main method if the script is executed.
if __name__ == "__main__":
    main()