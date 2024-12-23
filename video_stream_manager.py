import cv2 as cv
import logging

class VideoStreamManager:

    # This function initializes the video stream manager.
    # Arguments include the device index for the HDMI capture card,
    # width of the video frames, and height of the video frames.
    def __init__(self, capture_device = 0, frame_width = 1280, frame_height = 720):
        self.capture_device = capture_device
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.capture = None

        # This configures the logging.
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # This initializes the video stream from the HDMI capture card.
    def initialize_stream(self):

        logging.info("Initializing video stream..")
        self.capture = cv.VideoCapture(self.capture_device)

        # Here, we set the frame width and height.
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        self.capture.set(cv.CAP_PROP_HW_ACCELERATION, cv.VIDEO_ACCELERATION_ANY)

        # This verifies the video capture device has been opened.
        if not self.capture.isOpened():
            logging.error("Cannot open HDMI capture card.")
            raise RuntimeError("ERROR: Cannot open HDMI capture card.")
        
        logging.info(f"The video stream has been initialized with resolution{self.frame_width} by {self.frame_height}.")
        
    # This retrieves a frame from the video stream.
    # This returns the current video frame.
    def get_frame(self):

        if not self.capture or not self.capture.isOpened():
            logging.error("The video stream is not initialized.")
            raise RuntimeError("ERROR: The video stream is not initialized.")

        ret, frame = self.capture.read()
        if not ret:
            logging.error("Failed to capture the frame.")
            return None

        if frame is None:
            logging.error("Captured frame is None.")
            return None

        logging.info(f"Captured frame of size: {frame.shape}")
        return frame
    
    # This releases the video stream resources.
    def release_stream(self):

        if self.capture and self.capture.isOpened():
            self.capture.release()
            logging.info("The video stream was released.")
            print("The video stream was released.")

    def __enter__(self):
        self.initialize_stream()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release_stream()

    