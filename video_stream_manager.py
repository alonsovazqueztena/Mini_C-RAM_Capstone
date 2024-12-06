import cv2 as cv

class VideoStreamManager:

    # This function initializes the video stream manager.
    # Arguments include the device index for the HDMI capture card,
    # width of the video frames, and height of the video frames.
    def __init__(self, capture_device = 0, frame_width = 1280, frame_height = 720):
        self.capture_device = capture_device
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.capture = None
    
    # This initializes the video stream from the HDMI capture card.
    def initialize_stream(self):
        self.capture = cv.VideoCapture(self.capture_device)

        # Here, we set the frame width and height.
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # This verifies the video capture device has been opened.
        if not self.capture.isOpened():
            raise RuntimeError("ERROR: Cannot open HDMI capture card.")
        
        print(f"The video stream has been initialized with resolution 
              {self.frame_width} by {self.frame_height}.")
        
    # This retrieves a frame from the video stream.
    # This returns the current video frame.
    def get_frame(self):

        if not self.capture or not self.capture.isOpened():
            raise RuntimeError("ERROR: The video stream is not initialized.")

        ret, frame = self.capture.read()
        if not ret:
            raise RuntimeError("ERROR: Failed to read the frame from video stream.")
        
        return frame
    
    # This releases the video stream resources.
    def release_stream(self):

        if self.capture and self.capture.isOpened():
            self.capture.release()
            print("The video stream was released.")

    