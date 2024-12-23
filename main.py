import cv2 as cv
import logging
from video_stream_manager import VideoStreamManager
from frame_processor import FrameProcessor
from frame_pipeline import FramePipeline  # Assuming all these modules are saved and accessible.

def test_video_stream_manager():
    """Test the VideoStreamManager module."""
    try:
        logging.info("Testing VideoStreamManager...")
        video_stream = VideoStreamManager(capture_device=0, frame_width=640, frame_height=480)
        with video_stream as stream:
            # Test frame capture
            frame = stream.get_frame()
            if frame is None:
                raise RuntimeError("Failed to capture frame in VideoStreamManager test.")
            logging.info(f"Frame captured successfully with shape: {frame.shape}")

    except Exception as e:
        logging.error(f"VideoStreamManager test failed: {e}")

def test_frame_processor():
    """Test the FrameProcessor module."""
    try:
        logging.info("Testing FrameProcessor...")
        processor = FrameProcessor(target_width=640, target_height=640)

        # Create a dummy frame for testing
        dummy_frame = cv.imread("test_image.jpg")  # Replace with an actual image path or dummy data.
        if dummy_frame is None:
            raise ValueError("Failed to load test image.")

        # Process the frame
        processed_frame = processor.preprocess_frame(dummy_frame)
        if processed_frame is None:
            raise RuntimeError("Preprocessed frame is None.")

        logging.info(f"Preprocessed frame shape: {processed_frame.shape}")

    except Exception as e:
        logging.error(f"FrameProcessor test failed: {e}")

def test_frame_pipeline():
    """Test the FramePipeline module."""
    try:
        logging.info("Testing FramePipeline...")
        pipeline = FramePipeline(capture_device=0, frame_width=640, frame_height=480, target_width=640, target_height=640)
        pipeline.run()  # The pipeline runs continuously until 'q' is pressed.
        logging.info("FramePipeline test completed successfully.")

    except Exception as e:
        logging.error(f"FramePipeline test failed: {e}")

if __name__ == "__main__":
    # Configure logging for the test script.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Run tests
    logging.info("Starting module tests...")
    test_video_stream_manager()
    test_frame_processor()
    test_frame_pipeline()
    logging.info("Module tests completed.")

