# Daniel Saravia Source: https://grok.com/share/bGVnYWN5_52adc247-cde4-41e4-80bd-c70ef0c81dc9
# Alonso Vazquez Tena Source: https://grok.com/share/bGVnYWN5_10a649d1-27c0-4492-9545-3a3f518fd3c3
# DMX_frame_pipeline.py
import socket # For network connections
import concurrent.futures # For running tasks asynchronously
import websocket # For WebSocket communication
import cv2 as cv # For computer vision operations
from pynput import keyboard # For capturing keyboard inputs.
import logging # For logging messages.
import time # For time-based functions.
import numpy as np # For numerical calculations.
import os # For operating system functions.
from contextlib import redirect_stdout # Supress output when importing pygame
with redirect_stdout(open(os.devnull, 'w')):
    import pygame # For controller functionality.
import threading # For threading operations.
from frame_pipeline import FramePipeline # Base class for frame processing.

class DMXFramePipeline(FramePipeline):
    """
    DMXFramePipeline now uses a state machine to manage mode transitions:
      - MANUAL: User-controlled via keyboard/joystick (AI detection is still shown, but DMX values are not updated automatically).
      - LOCKED: Drone is detected; the beam locks on.
      - HOLD: Detections have just dropped out; continue holding the lock.
      - SCANNING: No detection for a while; resume scanning from last offset.
    """

    def __init__(self, model_path="drone_detector_12n.pt", confidence_threshold=0.5):
        super().__init__(model_path, confidence_threshold) # Initialize base class
        # Internal variables for lock-on mode.
        self.pan = 0.0
        self.tilt = 0.0
        self.k_pan = 0.005
        self.k_tilt = 0.005
        self.max_delta = 2.5

        # Manual mode DMX values.
        self.manual_mode = False
        self.keyboard_increment = 5
        self.current_pan = 127  # Starting at midpoint.
        self.current_tilt = 127

        # State machine.
        self.state = "SCANNING"  # One of "MANUAL", "LOCKED", "HOLD", "SCANNING"
        self.last_state = None

        # Lock-hold parameters with improved tolerances.
        self.last_detection_time = None
        self.lock_loss_threshold = 3.0  # Hold lock for 3 seconds after detection loss.
        self.consecutive_no_detection = 0
        self.detection_loss_threshold = 5  # Require 5 consecutive missed frames.
        self.last_lock_dmx_pan = None
        self.last_lock_dmx_tilt = None

        # Scanning parameters.
        self.scan_start_time = None
        self.scan_offset = 0.0  # persists scanning progress

        # Logging throttling.
        self.last_scan_log = 0
        self.last_lock_log = 0

        self.ws = None # WebSocket for DMX connection 
        self.init_dmx() # Initialize DMX connection

        self.joystick = None # Joystick object placeholder
        self.running = True # Pipeline running flag

    def init_dmx(self):
        """Establish DMX WebSocket connection."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80)) # Connect to external server to determine local IP
                ip = s.getsockname()[0] # Get local IP address
            self.ws = websocket.WebSocket() # Create WebSocket object
            self.ws.connect(f"ws://{ip}:9999/qlcplusWS") # Connect to DMX controller WebSocket
            logging.info(f"Connected to DMX controller at ws://{ip}:9999/qlcplusWS") # Log successful connection
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
        for det in detections: # Loop over detection results.
            x_min, y_min, x_max, y_max = map(int, det["bbox"]) # Convert bbox coordinates to int.
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1) # Draw bounding box.
        for obj in tracked_objects.values(): # Loop over tracked objects.
            cx, cy = map(int, obj["centroid"])  # Get centroid as integers.
            cv.circle(frame, (cx, cy), 2, (0, 255, 0), -1) # Draw centroid on frame.

    def update_dmx(self, centroid, frame):
        """Compute DMX values for lock-on based on detection and send them."""
        h, w = frame.shape[:2] # Get frame dimensions.
        error_x = centroid[0] - w / 2 # Calculate horizontal error from center.
        error_y = centroid[1] - h / 2 # Calculate vertical error from center.

        # Calculate delta changes with clamping to max_delta.
        delta_pan = max(-self.max_delta, min(self.k_pan * error_x, self.max_delta))
        delta_tilt = max(-self.max_delta, min(self.k_tilt * error_y, self.max_delta))

        self.pan = max(0, min(self.pan - delta_pan, 540)) # Update pan with clamped value.
        self.tilt = max(0, min(self.tilt - delta_tilt, 205)) # Update tilt with clamped value.

        # Convert pan and tilt to DMX scale (0-255).
        dmx_pan = (self.pan / 540.0) * 255.0
        dmx_tilt = (self.tilt / 205.0) * 255.0

        self.last_lock_dmx_pan = dmx_pan # Save last pan DMX value.
        self.last_lock_dmx_tilt = dmx_tilt # Save last tilt DMX value.

        self.send_dmx(1, dmx_pan) # Send pan DMX value to channel 1.
        self.send_dmx(3, dmx_tilt) # Send tilt DMX value to channel 3.

    def on_press(self, key):
        """Manual mode DMX update via keyboard listener.
           Reversed left and right controls: 
             - Left key now increases the pan value.
             - Right key now decreases the pan value.
        """
        if self.manual_mode: # Process only in manual mode.
            try:
                if key == keyboard.Key.up:
                    self.current_tilt = min(self.current_tilt + self.keyboard_increment, 255)
                    self.send_dmx(3, self.current_tilt) # Send updated tilt.
                elif key == keyboard.Key.down:
                    self.current_tilt = max(self.current_tilt - self.keyboard_increment, 0)
                    self.send_dmx(3, self.current_tilt) # Send updated tilt.
                elif key == keyboard.Key.right:
                    # Reversed: Right decreases pan instead of increasing.
                    self.current_pan = max(self.current_pan - self.keyboard_increment, 0)
                    self.send_dmx(1, self.current_pan) # Send updated pan.
                elif key == keyboard.Key.left:
                    # Reversed: Left increases pan instead of decreasing.
                    self.current_pan = min(self.current_pan + self.keyboard_increment, 255)
                    self.send_dmx(1, self.current_pan) # Send updated pan.
            except Exception as e:
                logging.error(f"Error in on_press: {e}")

    def start_keyboard_listener(self):
        """Start the keyboard listener in a separate thread."""
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press) # Create keyboard listener.
        self.keyboard_listener.start() # Start listener thread.

    def _update_state(self, detections):
        """Update the internal state based on detection results."""
        current_time = time.time()
        if self.manual_mode:
            new_state = "MANUAL"
        elif detections:
            # Reset no-detection counter and record time of detection.
            self.consecutive_no_detection = 0
            self.last_detection_time = current_time
            new_state = "LOCKED"
        else:
            self.consecutive_no_detection += 1
            # Remain LOCKED if misses are within tolerance.
            if self.consecutive_no_detection < self.detection_loss_threshold:
                new_state = "LOCKED"
            # Move to HOLD if within the time threshold.
            elif self.last_detection_time is not None and (current_time - self.last_detection_time) < self.lock_loss_threshold:
                new_state = "HOLD"
            else:
                new_state = "SCANNING"
        if new_state != self.state:
            logging.info(f"State change: {self.state} -> {new_state}")
            self.state = new_state

    def handle_controller(self):
        """Handle joystick and manual mode switching from controller events."""
        while self.running: # Run while pipeline is active.
            if self.joystick is not None:
                for event in pygame.event.get(): # Process pygame events.
                    if event.type == pygame.JOYBUTTONDOWN:
                        if event.button == 1:
                            self.running = False # Stop pipeline on button 1 press.
                            break
                        elif event.button == 3:
                            self.manual_mode = not self.manual_mode # Toggle manual mode.
                            if self.manual_mode:
                                logging.info("Manual Mode enabled.") # Log manual mode activation.
                            else:
                                logging.info("Automatic/Scanning Mode enabled.") # Log automatic mode activation.
                                self.last_detection_time = None # Reset detection time.
                                self.consecutive_no_detection = 0 # Reset detection counter.
                if self.manual_mode:
                    axis_x = self.joystick.get_axis(0) # Get horizontal axis value.
                    axis_y = self.joystick.get_axis(1) # Get vertical axis value.
                    deadzone = 0.2 # Define joystick deadzone.
                    sensitivity = 0.5 # Define sensitivity factor.
                    # Reverse the horizontal control: multiply axis_x by -1.
                    if abs(axis_x) > deadzone:
                        self.current_pan += int(self.keyboard_increment * (-axis_x) * sensitivity) # Adjust pan based on joystick.
                        self.current_pan = max(0, min(self.current_pan, 255)) # Clamp pan value.
                        self.send_dmx(1, self.current_pan) # Send updated pan DMX value.
                    if abs(axis_y) > deadzone:
                        self.current_tilt += int(self.keyboard_increment * -axis_y * sensitivity) # Adjust tilt based on joystick.
                        self.current_tilt = max(0, min(self.current_tilt, 255)) # Clamp tilt value.
                        self.send_dmx(3, self.current_tilt) # Send updated tilt DMX value.
            time.sleep(0.01)  # Avoid busy waiting. 

    def run(self):
        """
        Run the DMX pipeline using a state machine:
          - MANUAL: Direct control via keyboard/joystick (DMX values are not auto-updated, but AI detection is displayed).
          - LOCKED: A drone is detected and the system locks on.
          - HOLD: Brief loss of detection; maintain last DMX values.
          - SCANNING: Prolonged loss of detection; resume scanning from last offset.
        """
        self.start_keyboard_listener() # Start keyboard input listening.

        pygame.init() # Initialize pygame modules.
        pygame.joystick.init() # Initialize joystick support.
        if pygame.joystick.get_count() > 0: 
            self.joystick = pygame.joystick.Joystick(0) # Get the first joystick.
            self.joystick.init() # Initialize the joystick.
            logging.info(f"Initialized joystick: {self.joystick.get_name()}") # Log joystick name.
        else:
            logging.warning("No Xbox controller detected. Using keyboard only for manual mode.")

        controller_thread = threading.Thread(target=self.handle_controller, daemon=True) # Create controller thread.
        controller_thread.start() # Start the controller thread.

        cv.namedWindow("View", cv.WINDOW_NORMAL) # Create a resizable window.
        cv.resizeWindow("View", 640, 480) # Set window size.

        # Toggle button for manual mode.
        def toggle_manual_mode(state, userdata=None):
            if state:
                print("Manual Mode: ON") # Print manual mode ON status.
                logging.info("Manual Mode enabled.") # Log manual model enabled.
                self.manual_mode = True # Enable manual mode.
            else:
                print("Manual Mode: OFF (Automatic/Scanning Mode)") # Print manual mode OFF status.
                logging.info("Automatic/Scanning Mode enabled.") # Log automatic mode enabled.
                self.manual_mode = False # Disabled manual mode.
                # Reset detection parameters when leaving manual mode.
                self.last_detection_time = None
                self.consecutive_no_detection = 0

        cv.createButton("Manual Mode", toggle_manual_mode, None, cv.QT_CHECKBOX, 0) # Create a GUI button for manual mode.

        try:
            with self.video_stream as stream, concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                while self.running: # Main loop runs while active.
                    frame = stream.get_frame() # Get frame from video stream.
                    if frame is None:
                        break

                    future = executor.submit(self.ai_model_interface.predict, frame) # Run detection asynchronously.
                    detections = future.result() # Get detection results.

                    # Update state based on detection results.
                    self._update_state(detections)

                    # Execute behavior based on current state.
                    if self.state == "MANUAL":
                        # In MANUAL mode, the AI detection remains running for display.
                        cv.putText(frame, "Manual Mode", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                                   1, (0, 0, 255), 2)
                        tracked_objects = self.tracking_system.update(detections) # Update tracking system.
                        self.draw(frame, detections, tracked_objects) # Draw detections and tracking info.
                    elif self.state in ("LOCKED", "HOLD"):
                        if detections:
                            self.last_detection_time = time.time() # Update last detection time.
                            if self.scan_start_time is not None:
                                self.scan_offset = time.time() - self.scan_start_time # Update scan offset.
                                self.scan_start_time = None # Reset scan start time.
                            tracked_objects = self.tracking_system.update(detections) # Update tracking.
                            self.draw(frame, detections, tracked_objects) # Draw detection results.
                            self.update_dmx(detections[-1]["centroid"], frame) # Update DMX with last detection's centroid.
                            cv.putText(frame, "Locking On", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                                       1, (0, 255, 0), 2)
                        else:
                            if self.last_lock_dmx_pan is not None and self.last_lock_dmx_tilt is not None:
                                self.send_dmx(1, self.last_lock_dmx_pan) # Resend last known pan DMX value.
                                self.send_dmx(3, self.last_lock_dmx_tilt) # Resend last known tilt DMX value.
                            cv.putText(frame, "Locking On (holding)", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                                       1, (0, 255, 0), 2)
                    elif self.state == "SCANNING":
                        if self.scan_start_time is None:
                            self.scan_start_time = time.time() - self.scan_offset # Start or resume scanning timer.
                            logging.info("Switching to scanning mode.") # Log scanning mode transition.
                        t = time.time() - self.scan_start_time # Compute elapsed time for scanning.
                        T_pan = 20.0  # 20-second period for slower scanning.
                        pan_deg = 270 * (1 - np.cos(2 * np.pi * t / T_pan)) # Calculate pan angle using cosine.
                        T_tilt = 20.0 # Set scanning period for tilt.
                        tilt_deg = 102.5 * (((1 - np.cos(2 * np.pi * t / T_tilt)) / 2) ** 2) # Calculate tilt angle using cosine squared.
                        dmx_pan = (pan_deg / 540.0) * 255.0 # Convert pan degrees to DMX value.
                        dmx_tilt = (tilt_deg / 205.0) * 255.0 # Convert tilt degrees to DMX value.
                        self.send_dmx(1, dmx_pan) # Send scanning pan DMX value.
                        self.send_dmx(3, dmx_tilt) # Send scanning tilt DMX value.
                        cv.putText(frame, "Scanning...", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                                   1, (255, 0, 0), 2)
                        if time.time() - self.last_scan_log >= 5.0:
                            logging.info(f"Scanning DMX: Pan: {dmx_pan:.2f}, Tilt: {dmx_tilt:.2f} (t = {t:.2f}s)")
                            self.last_scan_log = time.time() # Update last scan log timestamp.
                        self.scan_offset = t # Update scan offset.

                    cv.imshow("View", frame) # Show the video frame.
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break # Exit loop if 'q' is pressed.
        finally:
            self.running = False # Stop the pipeline.
            self.video_stream.release_stream() # Release video stream resources.
            cv.destroyAllWindows() # Close all OpenCV windows.
            if self.ws:
                self.ws.close() # Close WebSocket connection if open.
            if hasattr(self, 'keyboard_listener'):
                self.keyboard_listener.stop() # Stop keyboard listener if active.
            pygame.quit() # Quit pygame modules.

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, # Set logging level to INFO.
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    pipeline = DMXFramePipeline(model_path="drone_detector_12n.pt", confidence_threshold=0.5) # Create DMXFramePipeline instance.
    pipeline.run() # Run the DMX pipeline.