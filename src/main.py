import cv2 as cv
import logging

from video_stream_manager import VideoStreamManager
from frame_processor import FrameProcessor
from frame_pipeline import FramePipeline
from yolo_model_interface import YOLOModelInterface
from detection_processor import DetectionProcessor
from tracking_system import TrackingSystem  # <-- Import your tracker here

def test_video_stream_manager():
    """
    Test the VideoStreamManager by attempting to capture a single frame.
    """
    logging.info("Testing VideoStreamManager...")
    try:
        video_stream = VideoStreamManager(capture_device=0, frame_width=640, frame_height=480)
        with video_stream as stream:
            frame = stream.get_frame()
            if frame is None:
                raise RuntimeError("Failed to capture frame in VideoStreamManager test.")
            logging.info(f"Frame captured successfully with shape: {frame.shape}")
    except Exception as e:
        logging.error(f"VideoStreamManager test failed: {e}")

def test_frame_processor():
    """
    Test the FrameProcessor by processing a dummy image.
    """
    logging.info("Testing FrameProcessor...")
    try:
        processor = FrameProcessor(target_width=640, target_height=640)

        # Load or create a dummy frame for testing.
        # Replace 'test_image.jpg' with a valid path to your image.
        dummy_frame = cv.imread("test_image.jpg")
        if dummy_frame is None:
            raise ValueError("Failed to load test image. Provide a valid image path.")

        processed_frame = processor.preprocess_frame(dummy_frame)
        if processed_frame is None or processed_frame.size == 0:
            raise RuntimeError("Preprocessed frame is None or empty.")

        logging.info(f"Preprocessed frame shape: {processed_frame.shape}")
    except Exception as e:
        logging.error(f"FrameProcessor test failed: {e}")

def test_yolo_model_interface():
    """
    Test the YOLOModelInterface by running inference on a sample image.
    """
    logging.info("Testing YOLOModelInterface...")
    try:
        # Adjust model_path if your YOLO model is in a different location.
        yolo_interface = YOLOModelInterface(model_path="yolo_epoch_100.pt", confidence_threshold=0.5)

        # Replace 'test_image.jpg' with any valid image file for real testing.
        test_img = cv.imread("test_image.jpg")
        if test_img is None:
            raise ValueError("Failed to load test image for YOLO. Provide a valid image path.")

        detections = yolo_interface.predict(test_img)
        logging.info(f"Raw YOLO detections: {detections}")
    except Exception as e:
        logging.error(f"YOLOModelInterface test failed: {e}")

def test_detection_processor():
    """
    Test the DetectionProcessor by running YOLO on a sample image
    and then processing the raw detections.
    """
    logging.info("Testing DetectionProcessor...")
    try:
        # 1. Initialize YOLO interface
        yolo_interface = YOLOModelInterface(model_path="yolo_epoch_100.pt", confidence_threshold=0.3)

        # 2. Load a test image
        test_img = cv.imread("test_image.jpg")
        if test_img is None:
            raise ValueError("Failed to load test image for YOLO. Provide a valid image path.")

        # 3. Get raw YOLO detections
        raw_detections = yolo_interface.predict(test_img)
        logging.info(f"Raw detections from YOLO: {raw_detections}")

        # 4. Initialize DetectionProcessor (filter by confidence >= 0.3, for example)
        detection_processor = DetectionProcessor(
            target_classes=None,  # e.g., [0,1] if you only want classes 0 and 1
            confidence_threshold=0.3
        )

        # 5. Process detections
        processed_detections = detection_processor.process_detections(raw_detections)
        logging.info(f"Processed detections: {processed_detections}")

    except Exception as e:
        logging.error(f"DetectionProcessor test failed: {e}")

def test_frame_pipeline():
    """
    Test the FramePipeline by running a continuous video stream at 640x480,
    processing each frame, and running YOLO detection.
    Press 'q' to stop the pipeline.
    """
    logging.info("Testing FramePipeline...")
    try:
        pipeline = FramePipeline(
            capture_device=0, 
            frame_width=640, 
            frame_height=480, 
            target_width=640, 
            target_height=640,
            model_path="yolo_epoch_100.pt",
            confidence_threshold=0.5
        )
        pipeline.run()  # Runs until 'q' is pressed or no frames are captured.
        logging.info("FramePipeline test completed successfully.")
    except Exception as e:
        logging.error(f"FramePipeline test failed: {e}")

def test_frame_pipeline_with_tracking():
    """
    Test the FramePipeline by running a continuous video stream at 640x480,
    including detection + tracking. Press 'q' to stop the pipeline.
    """
    logging.info("Testing FramePipeline with TrackingSystem...")
    try:
        # Create a TrackingSystem with desired parameters
        tracker = TrackingSystem(max_disappeared=50, max_distance=50)

        # Initialize the FramePipeline, passing the tracker in
        pipeline = FramePipeline(
            capture_device=0, 
            frame_width=640, 
            frame_height=480, 
            target_width=640, 
            target_height=640,
            model_path="yolo_epoch_100.pt",
            confidence_threshold=0.5,
            detection_processor=None,  # or specify if you want custom
            tracking_system=tracker
        )
        pipeline.run()  # Runs until 'q' is pressed or no frames are captured.
        logging.info("FramePipeline with TrackingSystem test completed successfully.")
    except Exception as e:
        logging.error(f"FramePipeline with TrackingSystem test failed: {e}")

def main():
    """
    Main entry point for testing all modules in a single script.
    """
    # Configure logging for the entire script.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("Starting all module tests...")

    # 1. Test VideoStreamManager (basic frame capture).
    test_video_stream_manager()

    # 2. Test FrameProcessor (image preprocessing).
    test_frame_processor()

    # 3. Test YOLOModelInterface (model loading and inference).
    test_yolo_model_interface()

    # 4. Test DetectionProcessor (filter & add centroids).
    test_detection_processor()

    # 5. Test the FramePipeline (real-time video + YOLO detection + 640x480).
    test_frame_pipeline()

    # 6. Test the FramePipeline WITH TRACKING (real-time detection + tracking).
    test_frame_pipeline_with_tracking()

    logging.info("All module tests completed.")

if __name__ == "__main__":
    main()



