# Alonso Vazquez Tena
# STG-452: Capstone Project II
# March 15, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8
# (used as starter code for basic functionality).

# This project requires the usage of logs for the developer
# to understand the conditions of the system, if an
# an error has occurred.
import logging

# This project requires the usage of computer vision.

# In this case, OpenCV will be used.
import cv2 as cv

# Utilizing AI models requires the usage of arrays and matrices
# for data processing.
import numpy as np


# This class serves to process frames for the AI model.

# This ensures that the frames can be processed to ensure
# more accurate object detection.
class FrameProcessor:
    """Creates and sets up the frame processor."""

    # This method initializes the frame processor.
    
    # The target width and height of the frame are taken in as arguments.

    # This can be adjusted as necessary, in this case, we keep
    # it as the full HD resolution.
    def __init__(
            self, target_width=1920, 
            target_height=1080):
        """Initialize the frame processor.

        Keyword arguments:
        self -- instance of the frame processor
        target_width -- target width of the frame (default 1920)
        target_height -- target height of the frame (default 1080)
        """
        self.target_width = target_width
        self.target_height = target_height

        # The logging is configured here. 
        
        # Basic info is put in, which includes the time, 
        # level name, and messages.
        logging.basicConfig(
            level=logging.WARNING, 
            format="%(asctime)s - %(levelname)s - %(message)s"
            )
    
    # This preprocesses a single frame for AI input.
    def preprocess_frame(
            self, frame):
        """Preprocess a single frame for AI input."""

        # If the frame is invalid or empty, an error is logged and raised.
        if frame is None or frame.size == 0:
            logging.error(
                "Invalid frame provided for preprocessing."
                )
            raise ValueError(
                "ERROR: Invalid frame provided."
                )

        # Resize the frame to the target dimensions.
        resized_frame = cv.resize(
            frame, (self.target_width, 
            self.target_height))

        # This adds a batch dimension as the AI model expects 4D input: 
        # batch, width, height, and channels.

        # We are looking at one frame at a time for the batch.
        preprocessed_frame = np.expand_dims(
            resized_frame, axis=0)
        
        # We return the preprocessed frame.
        return preprocessed_frame
