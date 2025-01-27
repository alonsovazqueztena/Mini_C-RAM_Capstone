import cv2 as cv
import numpy as np
import logging

class FrameProcessor:
    # This initializes the frame processor.
    # Arguments include the width and height to resize frames for YOLO input.
    def __init__(self, target_width=640, target_height=640):
        self.target_width = target_width
        self.target_height = target_height
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # This preprocesses a single frame for YOLO input.
    # Takes in the original video frame.
    # Returns the preprocessed frame ready for YOLO input.
    def preprocess_frame(self, frame):
        if frame is None or frame.size == 0:
            logging.error("Invalid frame provided for preprocessing.")
            raise ValueError("Invalid frame provided for preprocessing.")

        logging.info(f"Original frame size: {frame.shape}")

        # Resize the frame to the target dimensions.
        resized_frame = cv.resize(frame, (self.target_width, self.target_height))
        logging.info(f"Resized frame to: {self.target_width}x{self.target_height}")

        # Convert BGR to RGB for YOLO input.
        rgb_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
        logging.info("Converted frame from BGR to RGB.")

        # Normalize pixel values to range [0, 1].
        normalized_frame = rgb_frame / 255.0
        logging.info("Normalized frame pixel values to range [0, 1].")

        # Add a batch dimension as YOLO expects 4D input: batch_size, width, height, and channels.
        preprocessed_frame = np.expand_dims(normalized_frame, axis=0)
        logging.info(f"Added batch dimension. Preprocessed frame shape: {preprocessed_frame.shape}")

        return preprocessed_frame

    # This preprocesses multiple frames for YOLO input.
    # Takes a list of frames and returns a batch of preprocessed frames.
    def preprocess_frames(self, frames):
        if not frames or not isinstance(frames, list):
            logging.error("Invalid list of frames provided for batch preprocessing.")
            raise ValueError("Invalid list of frames provided for batch preprocessing.")
        
        logging.info(f"Processing a batch of {len(frames)} frames.")

        preprocessed_frames = [self.preprocess_frame(frame) for frame in frames]

        # Combine all preprocessed frames into a single batch.
        batch_frames = np.vstack(preprocessed_frames)
        logging.info(f"Batch of preprocessed frames shape: {batch_frames.shape}")

        return batch_frames

    
