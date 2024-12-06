import cv2 as cv
import numpy as np

class FrameProcessor:

    # This initalizes the frame processor.
    # Arguments include that of the width and height to resize
    # the frames for YOLO input.
    def __init__(self, target_width = 640, target_height = 640):
        self.target_width = target_width
        self.target_height = target_height

    # This preprocesses a frame for YOLO input.
    # We will take in the original video frame.
    # The preprocessed frame will be returned.
    def preprocess_frame(self, frame):

        # This resizes the frame to the target dimension.
        resized_frame = cv.resize(frame, (self.target_width, self.target_height))

        # This converts BGF to RGB for YOLO input.
        rgb_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)

        # This normalizes pixel values to range [0, 1] for YOLO input.
        normalized_frame = rgb_frame / 255.0

        # This adds a batch dimension as YOLO expects 4D input: batch_size, width,
        # height, and channels.
        preprocessed_frame = np.expand_dims(normalized_frame, axis = 0)

        return preprocessed_frame
    
    
