# Alonso Vazquez Tena
# STG-452: Capstone Project II
# March 16, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8
# (used as starter code for basic functionality) and
# https://chatgpt.com/share/67d77b29-c824-800e-ab25-2cc850596046
# (used to improve the video stream manager further).

# This project requires the usage of logs for the developer
# to understand the conditions of the system if
# an error has occurred.
import logging

# Queueing is used to work with the threads to better
# handle the video stream.
import queue

# This project uses threads to handle the video stream.
import threading

# Time is used to handle the sleep time for the threads.
import time

# This project requires the usage of computer vision.

# In this case, OpenCV will be used.
import cv2 as cv


# This class serves as code for a video stream manager.

# This serves to handle the logic to properly
# take in a video stream from a device.
class VideoStreamManager:
    """Creates and sets up the video stream manager."""

    # This method initializes the video stream manager.
    
    # The captured device is taken in as an index and the
    # frame is expected to be in full HD resolution.

    # Adjust the capture device index accordingly to
    # to your device as well as the resolution of
    # your camera.
    def __init__(
            self, capture_device=1, 
            frame_width=1920, frame_height=1080,
            max_queue_size=5):
        """Initialize the video stream manager.
        
        Keyword arguments:
        self -- instance of the video stream manager
        capture_device -- index of video capture stream device (default 0)
        frame_width -- width of video capture stream frame (default 1920)
        frame_height -- height of video capture stream frame (default 1080)
        max_queue_size -- maximum size of the frame queue (default 10)
        """
        self.capture_device = capture_device
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.capture = None

        # A first-in, first-out queue is to be used to store the frames.
        self.frame_queue = queue.Queue(
            maxsize=max_queue_size)
        self.stopped = False

         # A thread will be used to grab frames from the video stream.
        self.grabber_thread = None

        # The logging is configured here. 
        
        # Basic info is put in, which includes the time, 
        # level name, and messages.
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
            )

    # This initializes the video stream taken in from the
    # HDMI capture card off the device.
    def initialize_stream(
            self):
        """Initialize the video stream."""

        # OpenCV is executed on the video stream.
        self.capture = cv.VideoCapture(
            self.capture_device)

        # Captured frame width and height is set here.
        if not self.capture.set(
            cv.CAP_PROP_FRAME_WIDTH, 
            self.frame_width):
            logging.warning(
                "Failed to set the frame width."
                )
        if not self.capture.set(
            cv.CAP_PROP_FRAME_HEIGHT, 
            self.frame_height):
            logging.warning(
                "Failed to set the frame height."
                )

        # Any hardware acceleration available on the 
        # device the code is running on is to be leveraged.
        self.capture.set(
            cv.CAP_PROP_HW_ACCELERATION, cv.VIDEO_ACCELERATION_ANY
            )

        # This checks if capture device can be connected to
        # and opened.
        if not self.capture.isOpened():
            logging.critical(
                "Capture device open failed."
                )
            raise RuntimeError(
                "ERROR: Cannot open the capture device."
                )
        
        # A thread is used to grab frames from the video stream.

        # The thread is set as a daemon so it will exit when 
        # the main program exits.
        self.stopped = False
        self.grabber_thread = threading.Thread(
            target=self._frame_grabber, 
            daemon=True)
        self.grabber_thread.start()

    # This method grabs frames from the video stream 
    # and puts them in the queue.
    def _frame_grabber(
            self):
        """Grabs frames from the video stream and puts them in a queue."""

        # As long as the thread is not flagged to be stopped,
        # this loop will run and grab frames from the video stream.
        while not self.stopped:

            # This captures a frame from the video stream.
            ret, frame = self.capture.read()

            # If the frame is not captured, an error is logged.
            if not ret:
                logging.error(
                    "Failed to capture a frame."
                    )
                continue
            try:
                # If the frame queue is full, remove the oldest frame.
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                # Add the new frame to the queue.
                self.frame_queue.put(
                    frame, block=False
                    )

            # If the queue is full after removal, log a warning.
            except queue.Full:
                logging.warning(
                    "The frame queue is full, dropping a frame."
                    )

            # Sleep for a short time to avoid busy waiting. 
            time.sleep(
                0.001
                )

    # This method gets a frame from the video stream and
    # returns it in the program.
    def get_frame(
            self):
        """Retrieve the frame from the video stream."""

        # If there is no way to capture the video stream anymore,
        # whether through a webcam or capture card, an error is
        # raised and output in a log.
        if not self.capture or not self.capture.isOpened():
            logging.critical(
                "The video stream is not initialized."
                )
            raise RuntimeError(
                "ERROR: The video stream cannot be initialized."
                )
        
        # Try to get a frame from the queue, if available.
        try:
            frame = self.frame_queue.get(
                timeout=0.5)
            logging.debug(
                f"Retrieved frame of size: {frame.shape}"
                )
            return frame
        
        # If the queue is empty, log an error and raise an exception.
        except queue.Empty:
            logging.error(
                "No frame available in the queue"
                )
            raise RuntimeError(
                "ERROR: No frame available."
                )

    # This method releases the video stream resources upon
    # key from the user to terminate.
    def release_stream(
            self):
        """Release the video stream."""

        # The thread is flagged to stop and the thread is joined
        # upon the stopped execution.
        self.stopped = True
        if self.grabber_thread is not None:
            self.grabber_thread.join(
                timeout=2.0
                )
            
        # If the device is opened and detected
        # (thus, a video stream exists), the video stream will end and
        # this will be output in a log.
        if self.capture and self.capture.isOpened():
            self.capture.release()

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
        if exc_type:
            logging.error(f"Exception occurred: {exc_value}", 
                          exc_info=(exc_type, exc_value, traceback)
                          )
        self.release_stream()
