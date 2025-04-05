# Alonso Vazquez Tena
# STG-452: Capstone Project II
# March 16, 2025
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
from ai_model_interface import AIModelInterface
from frame_pipeline import FramePipeline
from tracking_system import TrackingSystem
from video_stream_manager import VideoStreamManager

# This method tests the AI model interface by 
# running inference on a sample image.
def test_ai_model_interface():
    """Test the AIModelInterface by 
    running inference on a sample image."""

    try:
        # This initializes the AI model interface.
        ai_interface = AIModelInterface()

        # A test image is loaded for AI.
        test_img = cv.imread("../test_images/drone_mock_test_1.jpg")
        
        # If the test image is empty or cannot be found, an error is raised.
        if test_img is None:
            raise ValueError("Failed to load test image for AI. Provide a valid image path.")

        # This runs inference on the test image.
        ai_interface.predict(test_img)
        
    # If an exception is raised, the error is logged.
    except Exception as e:
        logging.error(f"AIModelInterface test failed: {e}")

# This method tests the frame pipeline by 
# running a continuous video stream at 640x480,
# processing each frame, and running YOLO detection.
def test_frame_pipeline():
    """Test the FramePipeline by 
    running a continuous video stream at full HD,
    processing each frame, and running AI detection."""

    try:
        # The frame pipeline is initialized.
        pipeline = FramePipeline()

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

    # This tests the AIModelInterface (model loading and inference).
    test_ai_model_interface()

    # This tests the FramePipeline 
    # (all modules tested together).
    test_frame_pipeline()

# This runs the main method if the script is executed.
if __name__ == "__main__":
    main()