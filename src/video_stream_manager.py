# Alonso Vazquez Tena
# STG-452: Capstone Project II
# April 4, 2025
# I used source code from the following websites:
# https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8 (basic functionality),
# https://chatgpt.com/share/67d77b29-c824-800e-ab25-2cc850596046 (improvements)

import queue # For managing thread-safe frame queues
import threading # For running video stream operations in separate threads.
import cv2 as cv # For computer vision using OpenCV

class VideoStreamManager:
    """Manages video stream capture and processing."""

    def __init__(self, capture_device=1, max_queue_size=10):
        """Initialize with capture device index and frame queue size."""
        self.capture_device = capture_device
        self.capture = None
        self.frame_queue = queue.Queue(maxsize=max_queue_size) # FIFO queue for frames
        self.stopped = False
        self.grabber_thread = None # Thread for grabbing frames.

    def initialize_stream(self):
        """Start video stream and frame grabber thread."""
        self.capture = cv.VideoCapture(self.capture_device) # Open video capture.
        if not self.capture.isOpened(): # Check if capture open successfully.
            raise RuntimeError("ERROR: Cannot open the capture device.")
        self.capture.set(cv.CAP_PROP_BUFFERSIZE, 1) # Set minimal buffer size.
        self.capture.set(cv.CAP_PROP_HW_ACCELERATION, cv.VIDEO_ACCELERATION_ANY) # Use hardware acceleration if available.
        self.grabber_thread = threading.Thread(target=self._frame_grabber, daemon=True) # Create daemon thread.
        self.grabber_thread.start() # Start frame grabber.

    def _frame_grabber(self):
        """Continuously grab frames and update the frame queue."""
        while not self.stopped:
            ret, frame = self.capture.read() # Read a frame.
            if ret:
                with self.frame_queue.mutex:
                    self.frame_queue.queue.clear() # Clear queue to remove old frames.
                try:
                    self.frame_queue.put_nowait(frame) # Add new frame to queue.
                except queue.Full:
                    pass # Ignore if queue is full.

    def get_frame(self):
        """Retrieve frame from the video stream."""
        try:
            return self.frame_queue.get(timeout=0.5) # Get frame with timeout.
        except queue.Empty:
            return None

    def release_stream(self):
        """Stop frame grabbing and release video capture."""
        self.stopped = True
        if self.grabber_thread is not None:
            self.grabber_thread.join(timeout=0.5)
        if self.capture:
            self.capture.release() # Release the capture device.

    def __enter__(self):
        """Initialize the stream."""
        self.initialize_stream()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the video stream."""
        self.release_stream()