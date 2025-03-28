# Alonso Vazquez Tena
# STG-452: Capstone Project II
# March 16, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a17189-ca30-800e-858d-aac289e6cb56
# (used as starter code for basic functionality).
# capture_device=0 for gopro and phone
# This project requires the usage of logs for the developer
# to understand the conditions of the system, whether
# an error has occurred or the execution of the class was a success.
import logging

# This project requires the usage of computer vision.
# In this case, OpenCV will be used.
import cv2 as cv

# All the classes are imported from the src folder
# to be used in the frame pipeline class.
from ai_model_interface import AIModelInterface
from detection_processor import DetectionProcessor
from frame_pipeline import FramePipeline
from frame_processor import FrameProcessor
from tracking_system import TrackingSystem
from video_stream_manager import VideoStreamManager

# This method tests the video stream manager 
# by attempting to capture a single frame.
def test_video_stream_manager():
    """Test the VideoStreamManager by attempting 
    to capture a single frame."""

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
            
    # If an exception is raised, the error is logged.
    except Exception as e:
        logging.error(
            f"VideoStreamManager test failed: {e}"
            )

# This method tests the frame processor by processing a dummy image.
def test_frame_processor():
    """Test the FrameProcessor by processing a dummy image."""

    try:
        # This initializes the frame processor.
        processor = FrameProcessor(
            target_width=1920, target_height=1080)

        # This loads a dummy frame for testing.
        dummy_frame = cv.imread(
            "../test-images/drone_mock_test_1.jpg")
        
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
        
    # If an exception is raised, the error is logged.
    except Exception as e:
        logging.error(
            f"FrameProcessor test failed: {e}"
            )

# This method tests the AI model interface by 
# running inference on a sample image.
def test_ai_model_interface():
    """Test the AIModelInterface by 
    running inference on a sample image."""

    try:
        # This initializes the AI model interface.
        ai_interface = AIModelInterface(
            model_path="drone_detector_12m.pt", 
            confidence_threshold=0.5)

        # A test image is loaded for AI.
        test_img = cv.imread(
            "../test-images/drone_mock_test_1.jpg")
        
        # If the test image is empty or cannot be found, an error is raised.
        if test_img is None:
            raise ValueError(
                "Failed to load test image for AI. Provide a valid image path."
                )

        # This runs inference on the test image.
        ai_interface.predict(
            test_img
            )
        
    # If an exception is raised, the error is logged.
    except Exception as e:
        logging.error(
            f"AIModelInterface test failed: {e}"
            )

# This method tests the detection processor by 
# running AI on a sample image
# and then processing the raw detections.
def test_detection_processor():
    """ Test the DetectionProcessor by running AI on a sample image
    and then processing the raw detections."""

    try:
        # The YOLO model interface is initialized.
        ai_interface = AIModelInterface(
            model_path="drone_detector_12m.pt", 
            confidence_threshold=0.5)

        # The test image is loaded for YOLO.
        test_img = cv.imread(
            "../test-images/drone_mock_test_2.jpg")
        
        # If the test image is empty or cannot be found, an error is raised.
        if test_img is None:
            raise ValueError(
                "Failed to load test image for AI. Provide a valid image path."
                )

        # This runs inference on the test image.
        raw_detections = ai_interface.predict(
            test_img
            )

        # The detection processor is initialized.
        detection_processor = DetectionProcessor(
            target_classes=None
            )

        # The raw detections are processed.
        detection_processor.process_detections(
            raw_detections
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
    running a continuous video stream at full HD,
    processing each frame, and running AI detection."""

    try:
        # The frame pipeline is initialized.
        pipeline = FramePipeline(
            capture_device=0, 
            frame_width=1920, 
            frame_height=1080, 
            target_width=1920, 
            target_height=1080,
            model_path="drone_detector_12m.pt",
            confidence_threshold=0.5
        )

        # The frame pipeline is run and stops when the user quits.
        pipeline.run()
        
    # If an exception is raised, the error is logged.
    except Exception as e:
        logging.error(
            f"FramePipeline test failed: {e}"
            )

# This method is the main entry point for 
# testing all modules in a single script.
def main():
    """Main entry point for testing all modules in a single script."""

    # This configures logging for the entire script.
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
        )

    # This tests the VideoStreamManager (basic frame capture).
    test_video_stream_manager()

    # This tests the FrameProcessor (image preprocessing).
    test_frame_processor()

    # This tests the AIModelInterface (model loading and inference).
    test_ai_model_interface()

    # This tests the DetectionProcessor (filter & add centroids).
    test_detection_processor()

    # This tests the FramePipeline 
    # (all modules tested together).
    test_frame_pipeline()

# This runs the main method if the script is executed.
if __name__ == "__main__":
    main()