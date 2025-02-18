# Alonso Vazquez Tena
# STG-452: Capstone Project II
# January 13, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8
# (used as starter code for basic functionality).

# This project requires the usage of logs for the developer
# to understand the conditions of the system, whether
# an error has occurred or the execution of the class was a success.
import logging

# This project requires the usage of computer vision.

# In this case, OpenCV will be used.
import cv2 as cv


# This class serves as code for a video stream manager.

# This serves to handle the logic to properly
# take in a video stream connected to the Pi 5 from a camera
# through an HDMI capture card.
class VideoStreamManager:
    """Creates and sets up the video stream manager."""

    # This method initializes the video stream manager.
    
    # The captured device is taken in as an index,
    # the matching width and height of the lowest available video frame
    # resolution from the GoPro Hero 5 Black is also taken 
    # in, all as arguments.
    def __init__(
            self, capture_device=0, 
            frame_width=1920, frame_height=1080):
        """Initialize the video stream manager.
        
        Keyword arguments:
        self -- instance of the video stream manager
        capture_device -- index of video capture stream device (default 0)
        frame_width -- width of video capture stream frame (default 848)
        frame_height -- height of video capture stream frame (default 480)
        """
        self.capture_device = capture_device
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.capture = None

        # The logging is configured here. 
        
        # Basic info is put in, which includes the time, 
        # level name, and messages.
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
            )

    # This initializes the video stream taken in from the
    # HDMI capture card off the GoPro Hero 5 Black.
    def initialize_stream(
            self):
        """Initialize the video stream."""
        # The log includes a message displaying that the 
        # video stream is initializing.
        logging.info(
            "The video stream is initializing..."
            )

        # OpenCV is executed on the video stream.
        self.capture = cv.VideoCapture(
            self.capture_device)

        # Captured frame width and height is set here.
        self.capture.set(
            cv.CAP_PROP_FRAME_WIDTH, self.frame_width
            )
        self.capture.set(
            cv.CAP_PROP_FRAME_HEIGHT, self.frame_height
            )

        # Any hardware acceleration available on the 
        # device the code is running on is to be leveraged.
        self.capture.set(
            cv.CAP_PROP_HW_ACCELERATION, cv.VIDEO_ACCELERATION_ANY
            )

        # This checks if the HDMI capture card was able 
        # to be connected to and opened.
        if not self.capture.isOpened():
            logging.error(
                "HDMI capture card open failed."
                )
            raise RuntimeError(
                "ERROR: Cannot open the HDMI capture card."
                )

        # This message is displayed through a log that 
        # the stream has been initialized with a certain resolution.
        logging.info(
            f"The video stream has been initialized with resolution "
            f"{self.frame_width} by {self.frame_height}."
            )

    # This method gets a frame from the video stream and
    # returns it in the program.
    def get_frame(
            self):
        """Retrieve the frame from the video stream."""

        # If the HDMI capture card cannot be opened or 
        # there is no HDMI capture card detected,
        # an error is raised and output in a log.
        if not self.capture or not self.capture.isOpened():
            logging.error(
                "The video stream is not initialized."
                )
            raise RuntimeError(
                "ERROR: The video stream cannot be initialized."
                )

        # A boolean condition is checked if the
        # frame from the HDMI capture card can be received.
        ret, frame = self.capture.read()
        if not ret:
            logging.error(
                "Failed to capture the frame."
                )
            return None

        # If an invalid frame is received, an error is 
        # raised and output in a log.
        if frame is None:
            logging.error(
                "The captured frame is None (invalid)."
                )
            return None

        # When a frame is successfully received, the height, width,
        # and channels are logged. 
        logging.info(
            f"Captured frame of size: {frame.shape}"
            )
        return frame

    # This method releases the video stream resources upon
    # key from the user to terminate.
    def release_stream(
            self):
        """Release the video stream."""

        # If the HDMI capture card is opened and detected
        # (thus, a video stream exists), the video stream will end and
        # this will be output in a log.
        if self.capture and self.capture.isOpened():
            self.capture.release()
            logging.info(
                "The video stream was released."
                )
            print(
                "The video stream was successfully released."
                )

    # This is a simple enter method that only initializes the stream.
    def __enter__(
            self):
        """Initialize the stream."""
        self.initialize_stream()
        return self

    # This is a simple exit method that only releases the stream.
    
    # An exit message is displayed as well 
    # as the traceback through the terminal.
    def __exit__(
            self, exc_type, 
            exc_value, traceback):
        """Exit the video stream.
        
        self -- instance of the video stream manager
        exc_type -- class of the exception that occurred during execution
        exc_value -- actual exception instance that occurred
        traceback -- contains stack trace where exception was raised
        """
        self.release_stream()