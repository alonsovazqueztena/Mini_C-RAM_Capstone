# import cv2 as cv
# import logging
# from video_stream_manager import VideoStreamManager
# from frame_processor import FrameProcessor
# from frame_pipeline import FramePipeline
# from yolo_model_interface import YOLOModelInterface

# def test_video_stream_manager():
#     """
#     Test the VideoStreamManager by attempting to capture a single frame.
#     """
#     logging.info("Testing VideoStreamManager...")
#     try:
#         video_stream = VideoStreamManager(capture_device=0, frame_width=640, frame_height=480)
#         with video_stream as stream:
#             frame = stream.get_frame()
#             if frame is None:
#                 raise RuntimeError("Failed to capture frame in VideoStreamManager test.")
#             logging.info(f"Frame captured successfully with shape: {frame.shape}")
#     except Exception as e:
#         logging.error(f"VideoStreamManager test failed: {e}")

# def test_frame_processor():
#     """
#     Test the FrameProcessor by processing a dummy image.
#     """
#     logging.info("Testing FrameProcessor...")
#     try:
#         processor = FrameProcessor(target_width=640, target_height=640)

#         # Load or create a dummy frame for testing.
#         # You can replace 'test_image.jpg' with a path to an actual image.
#         dummy_frame = cv.imread("test_image.jpg")
#         if dummy_frame is None:
#             raise ValueError("Failed to load test image. Provide a valid image path.")

#         processed_frame = processor.preprocess_frame(dummy_frame)
#         if processed_frame is None or processed_frame.size == 0:
#             raise RuntimeError("Preprocessed frame is None or empty.")

#         logging.info(f"Preprocessed frame shape: {processed_frame.shape}")
#     except Exception as e:
#         logging.error(f"FrameProcessor test failed: {e}")

# def test_yolo_model_interface():
#     """
#     Test the YOLOModelInterface by running inference on a dummy or sample image.
#     """
#     logging.info("Testing YOLOModelInterface...")
#     try:
#         # Adjust the model_path if your YOLO model is in a different location.
#         yolo_interface = YOLOModelInterface(model_path="yolo_epoch_100.pt", confidence_threshold=0.5)
        
#         # Load a test image. Replace with any valid image file for real testing.
#         test_img = cv.imread("test_image.jpg")
#         if test_img is None:
#             raise ValueError("Failed to load test image for YOLO. Provide a valid image path.")

#         detections = yolo_interface.predict(test_img)
#         logging.info(f"YOLO detections: {detections}")
#     except Exception as e:
#         logging.error(f"YOLOModelInterface test failed: {e}")

# def test_frame_pipeline():
#     """
#     Test the FramePipeline by running a continuous video stream,
#     processing each frame, and running YOLO detection.
#     Press 'q' to stop the pipeline.
#     """
#     logging.info("Testing FramePipeline...")
#     try:
#         pipeline = FramePipeline(
#             capture_device=0, 
#             frame_width=640, 
#             frame_height=480, 
#             target_width=640, 
#             target_height=640,
#             model_path="yolo_epoch_100.pt",      # Update path if needed
#             confidence_threshold=0.5
#         )
#         pipeline.run()  # This will run until 'q' is pressed or no frames are captured.
#         logging.info("FramePipeline test completed successfully.")
#     except Exception as e:
#         logging.error(f"FramePipeline test failed: {e}")

# def main():
#     """
#     Main entry point for testing all modules in a single script.
#     """
#     # Configure logging for the entire script.
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#     logging.info("Starting all module tests...")

#     # 1. Test the VideoStreamManager (basic frame capture).
#     test_video_stream_manager()

#     # 2. Test the FrameProcessor (image preprocessing).
#     test_frame_processor()

#     # 3. Test the YOLOModelInterface (model loading and inference).
#     test_yolo_model_interface()

#     # 4. Test the FramePipeline (real-time video + YOLO detection).
#     test_frame_pipeline()

#     logging.info("All module tests completed.")

# if __name__ == "__main__":
#     main()


