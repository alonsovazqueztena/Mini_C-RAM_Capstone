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
import os
from contextlib import redirect_stdout
import pygame
import threading

from frame_pipeline import FramePipeline

class DMXFramePipeline(FramePipeline):
    """
    DMXFramePipeline now uses a state machine to manage mode transitions:
      - MANUAL: User-controlled via keyboard.
      - LOCKED: Drone is detected; the beam locks on.
      - HOLD: Detections have just dropped out; continue holding the lock.
      - SCANNING: No detection for a while; resume scanning from last offset.
    """

    def __init__(self, model_path="drone_detector_12n.pt", confidence_threshold=0.5):
        super().__init__(model_path, confidence_threshold)
        # Internal variables for lock-on mode.
        self.pan = 0.0
        self.tilt = 0.0
        self.k_pan = 0.005
        self.k_tilt = 0.005
        self.max_delta = 2.5

        # Manual mode DMX values.
        self.manual_mode = False
        self.keyboard_increment = 3
        self.current_pan = 127  # Starting at midpoint.
        self.current_tilt = 127

        # State machine.
        self.state = "SCANNING"  # One of "MANUAL", "LOCKED", "HOLD", "SCANNING"
        self.last_state = None

        # Lock-hold parameters.
        self.last_detection_time = None
        self.lock_loss_threshold = 2.0  # seconds to hold lock after detection loss.
        self.consecutive_no_detection = 0
        self.detection_loss_threshold = 3  # frames with no detection before switching to SCANNING.
        self.last_lock_dmx_pan = None
        self.last_lock_dmx_tilt = None

        # Scanning parameters.
        self.scan_start_time = None
        self.scan_offset = 0.0  # persists scanning progress

        # Logging throttling.
        self.last_scan_log = 0
        self.last_lock_log = 0

        self.ws = None
        self.init_dmx()

        self.joystick = None
        self.running = True

    def init_dmx(self):
        """Establish DMX WebSocket connection."""
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
        """Send DMX value to the controller."""
        if self.ws:
            try:
                self.ws.send(f"CH|{channel}|{int(value)}")
            except Exception as e:
                logging.error(f"Error sending DMX value: {e}")

    def draw(self, frame, detections, tracked_objects):
        """Draw detections on the frame."""
        for det in detections:
            x_min, y_min, x_max, y_max = map(int, det["bbox"])
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        for obj in tracked_objects.values():
            cx, cy = map(int, obj["centroid"])
            cv.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

    def update_dmx(self, centroid, frame):
        """Compute DMX values for lock-on based on detection and save them."""
        h, w = frame.shape[:2]
        error_x = centroid[0] - w / 2
        error_y = centroid[1] - h / 2

        delta_pan = max(-self.max_delta, min(self.k_pan * error_x, self.max_delta))
        delta_tilt = max(-self.max_delta, min(self.k_tilt * error_y, self.max_delta))

        self.pan = max(0, min(self.pan - delta_pan, 540))
        self.tilt = max(0, min(self.tilt - delta_tilt, 205))

        dmx_pan = (self.pan / 540.0) * 255.0
        dmx_tilt = (self.tilt / 205.0) * 255.0

        self.last_lock_dmx_pan = dmx_pan
        self.last_lock_dmx_tilt = dmx_tilt

        self.send_dmx(1, dmx_pan)
        self.send_dmx(3, dmx_tilt)

    def on_press(self, key):
        """Manual mode DMX update via keyboard listener."""
        if self.manual_mode:
            try:
                if key == keyboard.Key.up:
                    self.current_tilt = min(self.current_tilt + self.keyboard_increment, 255)
                    self.send_dmx(3, self.current_tilt)
                elif key == keyboard.Key.down:
                    self.current_tilt = max(self.current_tilt - self.keyboard_increment, 0)
                    self.send_dmx(3, self.current_tilt)
                elif key == keyboard.Key.right:
                    self.current_pan = min(self.current_pan + self.keyboard_increment, 255)
                    self.send_dmx(1, self.current_pan)
                elif key == keyboard.Key.left:
                    self.current_pan = max(self.current_pan - self.keyboard_increment, 0)
                    self.send_dmx(1, self.current_pan)
            except Exception as e:
                logging.error(f"Error in on_press: {e}")

    def start_keyboard_listener(self):
        """Start the keyboard listener in a separate thread."""
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.keyboard_listener.start()

    def _update_state(self, detections):
        """Update the internal state based on detection results."""
        if self.manual_mode:
            new_state = "MANUAL"
        elif detections:
            # Reset no-detection counter and update state to LOCKED.
            self.consecutive_no_detection = 0
            new_state = "LOCKED"
        else:
            self.consecutive_no_detection += 1
            if (self.last_detection_time is not None and 
                (time.time() - self.last_detection_time) < self.lock_loss_threshold and
                self.consecutive_no_detection < self.detection_loss_threshold):
                new_state = "HOLD"
            else:
                new_state = "SCANNING"
        if new_state != self.state:
            logging.info(f"State change: {self.state} -> {new_state}")
            self.state = new_state

    def handle_controller(self):
        while self.running:
            if self.joystick is not None:
                for event in pygame.event.get():
                    if event.type == pygame.JOYBUTTONDOWN:
                        if event.button == 1:
                            self.running = False
                            break
                        elif event.button == 3:
                            self.manual_mode = not self.manual_mode
                            if self.manual_mode:
                                logging.info("Manual Mode enabled.")
                            else:
                                logging.info("Automatic/Scanning Mode enabled.")
                                self.last_detection_time = None
                                self.consecutive_no_detection = 0
                if self.manual_mode:
                    axis_x = self.joystick.get_axis(0)
                    axis_y = self.joystick.get_axis(1) 
                    deadzone = 0.2
                    sensitivity = 0.2
                    if abs(axis_x) > deadzone:
                        self.current_pan += int(self.keyboard_increment * axis_x * sensitivity)
                        self.current_pan = max(0, min(self.current_pan, 255))
                        self.send_dmx(1, self.current_pan)
                    if abs(axis_y) > deadzone:
                        self.current_tilt += int(self.keyboard_increment * -axis_y * sensitivity)
                        self.current_tilt = max(0, min(self.current_tilt, 255))
                        self.send_dmx(3, self.current_tilt)
            time.sleep(0.03)  # Avoid busy waiting.
    def run(self):
        """
        Run the DMX pipeline using a state machine:
          - MANUAL: Direct control via keyboard.
          - LOCKED: A drone is detected and the system locks on.
          - HOLD: Brief loss of detection; maintain last DMX values.
          - SCANNING: Prolonged loss of detection; resume scanning from last offset.
        """
        self.start_keyboard_listener()

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logging.info(f"Initialized joystick: {self.joystick.get_name()}")
        else:
            logging.warning("No Xbox controller detected. Using keyboard only for manual mode.")

        controller_thread = threading.Thread(target=self.handle_controller, daemon=True)
        controller_thread.start()

        cv.namedWindow("View", cv.WINDOW_NORMAL)
        cv.resizeWindow("View", 640, 480)

        # Toggle button for manual mode.
        def toggle_manual_mode(state, userdata=None):
            if state:
                print("Manual Mode: ON")
                logging.info("Manual Mode enabled.")
                self.manual_mode = True
            else:
                print("Manual Mode: OFF (Automatic/Scanning Mode)")
                logging.info("Automatic/Scanning Mode enabled.")
                self.manual_mode = False
                # When leaving manual, reset detection parameters.
                self.last_detection_time = None
                self.consecutive_no_detection = 0

        cv.createButton("Manual Mode", toggle_manual_mode, None, cv.QT_CHECKBOX, 0)

        try:
            with self.video_stream as stream, concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                while self.running:
                    frame = stream.get_frame()
                    if frame is None:
                        break

                    future = executor.submit(self.ai_model_interface.predict, frame)
                    detections = future.result()

                    # Update state based on detection results.
                    self._update_state(detections)

                    # Execute behavior based on current state.
                    if self.state == "MANUAL":
                        cv.putText(frame, "Manual Mode", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                                   1, (0, 0, 255), 2)
                    elif self.state in ("LOCKED", "HOLD"):
                        if detections:
                            self.last_detection_time = time.time()
                            # If scanning was active, record scan progress.
                            if self.scan_start_time is not None:
                                self.scan_offset = time.time() - self.scan_start_time
                                self.scan_start_time = None
                            tracked_objects = self.tracking_system.update(detections)
                            self.draw(frame, detections, tracked_objects)
                            self.update_dmx(detections[-1]["centroid"], frame)
                            cv.putText(frame, "Locking On", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                                       1, (0, 255, 0), 2)
                        else:
                            # HOLD state: maintain last DMX values.
                            if self.last_lock_dmx_pan is not None and self.last_lock_dmx_tilt is not None:
                                self.send_dmx(1, self.last_lock_dmx_pan)
                                self.send_dmx(3, self.last_lock_dmx_tilt)
                            cv.putText(frame, "Locking On (holding)", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                                       1, (0, 255, 0), 2)
                    elif self.state == "SCANNING":
                        # If transitioning to scanning, initialize scan_start_time if not already.
                        if self.scan_start_time is None:
                            self.scan_start_time = time.time() - self.scan_offset
                            logging.info("Switching to scanning mode.")
                        t = time.time() - self.scan_start_time
                        T_pan = 20.0  # 20-second period for slower scanning.
                        pan_deg = 270 * (1 - np.cos(2 * np.pi * t / T_pan))
                        T_tilt = 20.0
                        tilt_deg = 102.5 * (((1 - np.cos(2 * np.pi * t / T_tilt)) / 2) ** 2)
                        dmx_pan = (pan_deg / 540.0) * 255.0
                        dmx_tilt = (tilt_deg / 205.0) * 255.0
                        self.send_dmx(1, dmx_pan)
                        self.send_dmx(3, dmx_tilt)
                        cv.putText(frame, "Scanning...", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                                   1, (255, 0, 0), 2)
                        # Log scanning DMX values every 5 seconds.
                        if time.time() - self.last_scan_log >= 5.0:
                            logging.info(f"Scanning DMX: Pan: {dmx_pan:.2f}, Tilt: {dmx_tilt:.2f} (t = {t:.2f}s)")
                            self.last_scan_log = time.time()
                        self.scan_offset = t

                    cv.imshow("View", frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.running = False
            self.video_stream.release_stream()
            cv.destroyAllWindows()
            if self.ws:
                self.ws.close()
            if hasattr(self, 'keyboard_listener'):
                self.keyboard_listener.stop()
            pygame.quit()

if __name__ == "__main__":
    # Set logging to INFO for key state transitions and occasional DMX summaries.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    pipeline = DMXFramePipeline(model_path="drone_detector_12n.pt", confidence_threshold=0.5)
    pipeline.run()
