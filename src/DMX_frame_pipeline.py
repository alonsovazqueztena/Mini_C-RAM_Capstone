# Daniel Saravia Source: https://grok.com/share/bGVnYWN5_52adc247-cde4-41e4-80bd-c70ef0c81dc9
# DMX_frame_pipeline.py
import socket
import concurrent.futures
import websocket
import cv2 as cv
from pynput import keyboard
import logging
import time
import numpy as np

from frame_pipeline import FramePipeline

class DMXFramePipeline(FramePipeline):
    """
    Extends FramePipeline with DMX control for drone tracking in two modes:
      - Manual: Arrow keys update DMX (via pynput).
      - Automatic/Scanning: If a drone is detected, the beam locks on;
        if detections are momentarily lost (less than the threshold), the beam holds its last DMX values;
        otherwise, it follows a predefined scanning path.
    """

    def __init__(self, model_path="drone_detector_12n.pt", confidence_threshold=0.5):
        super().__init__(model_path, confidence_threshold)
        # Automatic mode parameters (internal pan/tilt values)
        self.pan = 0.0
        self.tilt = 0.0
        self.k_pan = 0.005
        self.k_tilt = 0.005
        self.max_delta = 2.5

        # Manual mode parameters (DMX values in the 0–255 range)
        self.manual_mode = False
        self.keyboard_increment = 5
        self.current_pan = 127   # Starting at midpoint.
        self.current_tilt = 127

        # Variables for lock-on hysteresis.
        self.last_detection_time = None      # Time when a detection was last seen.
        self.lock_loss_threshold = 1.0         # Seconds to wait before switching to scanning.
        self.last_lock_dmx_pan = None          # Last DMX value sent for pan when locked.
        self.last_lock_dmx_tilt = None         # Last DMX value sent for tilt when locked.
        
        # Timer for scanning when no detection is found.
        self.scan_start_time = None

        self.ws = None
        self.init_dmx()

    def init_dmx(self):
        """Quick DMX WebSocket setup."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            self.ws = websocket.WebSocket()
            self.ws.connect(f"ws://{ip}:9999/qlcplusWS")
            logging.info(f"Connected to DMX controller at ws://{ip}:9999/qlcplusWS")
        except Exception as e:
            logging.error(f"DMX initialization failed: {e}")
            self.ws = None

    def send_dmx(self, channel, value):
        """Send DMX value with no error checking."""
        if self.ws:
            try:
                self.ws.send(f"CH|{channel}|{int(value)}")
            except Exception as e:
                logging.error(f"Error sending DMX value: {e}")

    def draw(self, frame, detections, tracked_objects):
        """Minimal drawing for speed (overrides base draw)."""
        for det in detections:
            x_min, y_min, x_max, y_max = map(int, det["bbox"])
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        for obj in tracked_objects.values():
            cx, cy = map(int, obj["centroid"])
            cv.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

    def update_dmx(self, centroid, frame):
        """Fast DMX update from detections (lock-on mode)."""
        h, w = frame.shape[:2]
        error_x = centroid[0] - w / 2
        error_y = centroid[1] - h / 2

        delta_pan = max(-self.max_delta, min(self.k_pan * error_x, self.max_delta))
        delta_tilt = max(-self.max_delta, min(self.k_tilt * error_y, self.max_delta))

        self.pan = max(0, min(self.pan - delta_pan, 540))
        self.tilt = max(0, min(self.tilt - delta_tilt, 205))

        dmx_pan = (self.pan / 540.0) * 255.0
        dmx_tilt = (self.tilt / 205.0) * 255.0

        # Store last lock-on DMX values.
        self.last_lock_dmx_pan = dmx_pan
        self.last_lock_dmx_tilt = dmx_tilt

        self.send_dmx(1, dmx_pan)
        self.send_dmx(3, dmx_tilt)

    def on_press(self, key):
        """
        Callback for the keyboard listener.
        When manual mode is enabled, arrow key presses update the DMX values.
        """
        if self.manual_mode:
            try:
                if key == keyboard.Key.up:
                    self.current_tilt = min(self.current_tilt + self.keyboard_increment, 255)
                    self.send_dmx(3, self.current_tilt)
                    print(f"Manual Mode: Tilt increased to {self.current_tilt}")
                elif key == keyboard.Key.down:
                    self.current_tilt = max(self.current_tilt - self.keyboard_increment, 0)
                    self.send_dmx(3, self.current_tilt)
                    print(f"Manual Mode: Tilt decreased to {self.current_tilt}")
                elif key == keyboard.Key.right:
                    self.current_pan = min(self.current_pan + self.keyboard_increment, 255)
                    self.send_dmx(1, self.current_pan)
                    print(f"Manual Mode: Pan increased to {self.current_pan}")
                elif key == keyboard.Key.left:
                    self.current_pan = max(self.current_pan - self.keyboard_increment, 0)
                    self.send_dmx(1, self.current_pan)
                    print(f"Manual Mode: Pan decreased to {self.current_pan}")
            except Exception as e:
                logging.error(f"Error in on_press: {e}")

    def start_keyboard_listener(self):
        """Starts the pynput keyboard listener in a separate thread."""
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.keyboard_listener.start()

    def run(self):
        """
        Run the DMX pipeline with two user-selectable modes:
          - Manual: Arrow keys update DMX.
          - Automatic/Scanning: If a drone is detected, lock on;
            if detections are momentarily lost (less than the threshold), continue holding lock;
            otherwise, follow the predefined scanning path.
        """
        self.start_keyboard_listener()

        cv.namedWindow("View", cv.WINDOW_NORMAL)
        cv.resizeWindow("View", 640, 480)

        # Create a toggle button for manual mode.
        def toggle_manual_mode(state, userdata=None):
            if state:
                print("Manual Mode: ON")
                logging.info("Manual Mode enabled.")
                self.manual_mode = True
            else:
                print("Manual Mode: OFF (Automatic/Scanning Mode)")
                logging.info("Automatic/Scanning Mode enabled.")
                self.manual_mode = False
                # Reset timers when switching modes.
                self.scan_start_time = None
                self.last_detection_time = None

        cv.createButton("Manual Mode", toggle_manual_mode, None, cv.QT_CHECKBOX, 0)

        try:
            with self.video_stream as stream, concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                while True:
                    frame = stream.get_frame()
                    if frame is None:
                        break

                    if self.manual_mode:
                        # Manual mode: use stored DMX values from keyboard listener.
                        cv.putText(frame, "Manual Mode", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                                   1, (0, 0, 255), 2)
                        self.send_dmx(1, self.current_pan)
                        self.send_dmx(3, self.current_tilt)
                    else:
                        # Automatic/Scanning mode:
                        future = executor.submit(self.ai_model_interface.predict, frame)
                        detections = future.result()
                        if detections:
                            # Drone detected: lock on.
                            self.last_detection_time = time.time()
                            tracked_objects = self.tracking_system.update(detections)
                            self.draw(frame, detections, tracked_objects)
                            self.update_dmx(detections[-1]["centroid"], frame)
                            cv.putText(frame, "Locking On", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                                       1, (0, 255, 0), 2)
                            # Reset scanning timer when lock-on occurs.
                            self.scan_start_time = None
                        else:
                            # No detection available.
                            # If we were recently locked on and the loss duration is less than threshold,
                            # continue holding the last lock-on DMX values.
                            if (self.last_detection_time is not None and 
                                (time.time() - self.last_detection_time) < self.lock_loss_threshold):
                                if self.last_lock_dmx_pan is not None and self.last_lock_dmx_tilt is not None:
                                    self.send_dmx(1, self.last_lock_dmx_pan)
                                    self.send_dmx(3, self.last_lock_dmx_tilt)
                                cv.putText(frame, "Locking On (holding)", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                                           1, (0, 255, 0), 2)
                            else:
                                # Detection lost for longer than threshold, switch to scanning.
                                if self.scan_start_time is None:
                                    self.scan_start_time = time.time()
                                t = time.time() - self.scan_start_time
                                # Scanning path equations (starting from DMX 0,0):
                                # Use a slower period: 20 seconds for both pan and tilt.
                                T_pan = 20.0
                                pan_deg = 270 * (1 - np.cos(2 * np.pi * t / T_pan))
                                T_tilt = 20.0
                                tilt_deg = 102.5 * (((1 - np.cos(2 * np.pi * t / T_tilt)) / 2) ** 2)
                                dmx_pan = (pan_deg / 540.0) * 255.0
                                dmx_tilt = (tilt_deg / 205.0) * 255.0
                                self.send_dmx(1, dmx_pan)
                                self.send_dmx(3, dmx_tilt)
                                cv.putText(frame, "Scanning...", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                                           1, (255, 0, 0), 2)

                    cv.imshow("View", frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.video_stream.release_stream()
            cv.destroyAllWindows()
            if self.ws:
                self.ws.close()
            if hasattr(self, 'keyboard_listener'):
                self.keyboard_listener.stop()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    pipeline = DMXFramePipeline(model_path="drone_detector_12n.pt", confidence_threshold=0.5)
    pipeline.run()
