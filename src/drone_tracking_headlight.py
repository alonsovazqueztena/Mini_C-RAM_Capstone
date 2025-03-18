import cv2 as cv
import numpy as np
import websocket
import socket
import logging
import concurrent.futures
from ai_model_interface import AIModelInterface
from detection_processor import DetectionProcessor
from frame_processor import FrameProcessor
from tracking_system import TrackingSystem
from video_stream_manager import VideoStreamManager
from frame_pipeline import FramePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# WebSocket setup for QLC+
def get_host_ip():
    """Retrieve the local host computer's IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:
        logging.error(f"Error determining host IP: {e}")
        return "127.0.0.1"

QLC_IP = get_host_ip()
QLC_WS_URL = f"ws://{QLC_IP}:9999/qlcplusWS"
ws = websocket.WebSocket()
try:
    ws.connect(QLC_WS_URL)
    logging.info(f"Connected to QLC+ WebSocket at {QLC_WS_URL}")
except Exception as e:
    logging.error(f"Failed to connect to QLC+: {e}")
    raise

def send_dmx_value(channel, value):
    """Send a DMX value to the specified channel through the WebSocket."""
    try:
        value = int(np.clip(value, 0, 255))  # Ensure value is in valid DMX range
        ws.send(f"CH|{channel}|{value}")
        logging.debug(f"Sent DMX: CH|{channel}|{value}")
    except Exception as e:
        logging.error(f"Failed to send DMX value: {e}")

class DroneTrackingHeadlight:
    def __init__(self, capture_device=1, model_path="drone_detector_ai.pt", 
                 fov_horiz=60, fov_vert=45, smoothing_alpha=0.3):
        """Initialize the drone tracking system with headlight control.

        Args:
            capture_device (int): Camera index (e.g., 1 for secondary USB camera).
            model_path (str): Path to the YOLO model file.
            fov_horiz (float): Camera horizontal field of view in degrees.
            fov_vert (float): Camera vertical field of view in degrees.
            smoothing_alpha (float): Smoothing factor for DMX values (0-1).
        """
        self.pipeline = FramePipeline(
            capture_device=capture_device,
            frame_width=1920,
            frame_height=1080,
            target_width=1920,
            target_height=1080,
            model_path=model_path,
            confidence_threshold=0.5
        )
        self.fov_horiz = fov_horiz
        self.fov_vert = fov_vert
        self.smoothing_alpha = smoothing_alpha  # For exponential moving average
        self.prev_dmx_pan = 127.5  # Initial center value (0-255 range)
        self.prev_dmx_tilt = 127.5
        self.frame_width = 1920
        self.frame_height = 1080

    def pixel_to_dmx(self, x, y):
        """Convert pixel coordinates to smoothed DMX pan/tilt values."""
        # Normalize to [-0.5, 0.5] range, then scale to camera FOV
        pan_deg = (x / self.frame_width - 0.5) * self.fov_horiz
        tilt_deg = (0.5 - y / self.frame_height) * self.fov_vert
        # Map to DMX ranges (pan: 0-540°, tilt: 0-205°)
        raw_dmx_pan = (pan_deg + 270) / 540 * 255  # Center at 270°
        raw_dmx_tilt = (tilt_deg + 102.5) / 205 * 255  # Center at 102.5°
        # Apply exponential smoothing
        dmx_pan = (self.smoothing_alpha * raw_dmx_pan + 
                   (1 - self.smoothing_alpha) * self.prev_dmx_pan)
        dmx_tilt = (self.smoothing_alpha * raw_dmx_tilt + 
                    (1 - self.smoothing_alpha) * self.prev_dmx_tilt)
        # Update previous values
        self.prev_dmx_pan = dmx_pan
        self.prev_dmx_tilt = dmx_tilt
        return np.clip(dmx_pan, 0, 255), np.clip(dmx_tilt, 0, 255)

    def select_target_drone(self, tracked_objects):
        """Select the most prominent drone to track based on confidence."""
        if not tracked_objects:
            return None
        # Choose drone with highest confidence
        return max(tracked_objects.values(), 
                  key=lambda d: d.get("confidence", 0), 
                  default=None)

    def run(self):
        """Run the drone tracking and headlight control loop."""
        try:
            with self.pipeline.video_stream as stream, \
                 concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                cv.namedWindow("Drone Tracking", cv.WINDOW_NORMAL)
                cv.resizeWindow("Drone Tracking", 800, 600)
                logging.info("Starting drone tracking with headlight control...")

                while True:
                    frame = stream.get_frame()
                    if frame is None:
                        logging.warning("No frame captured. Exiting...")
                        break

                    # Process frame and get tracked objects
                    processed_frame = self.pipeline.frame_processor.preprocess_frame(frame)
                    future = executor.submit(self.pipeline.ai_model_interface.predict, 
                                          processed_frame[0])
                    raw_detections = future.result()
                    processed_detections = self.pipeline.detection_processor.process_detections(
                        raw_detections)
                    tracked_objects = self.pipeline.tracking_system.update(processed_detections)

                    # Draw detections and tracked objects
                    self.pipeline.draw_detections(frame, processed_detections)
                    self.pipeline.draw_tracked_objects(frame, tracked_objects)

                    # Control headlight if drones are tracked
                    target_drone = self.select_target_drone(tracked_objects)
                    if target_drone:
                        cx, cy = target_drone["centroid"]
                        dmx_pan, dmx_tilt = self.pixel_to_dmx(cx, cy)
                        send_dmx_value(1, dmx_pan)  # Pan
                        send_dmx_value(3, dmx_tilt)  # Tilt
                        logging.info(f"Tracking drone at ({cx:.1f}, {cy:.1f}) -> "
                                   f"DMX Pan: {dmx_pan:.1f}, Tilt: {dmx_tilt:.1f}")
                    else:
                        logging.debug("No drones detected.")

                    cv.imshow("Drone Tracking", frame)
                    key = cv.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logging.info("User requested exit.")
                        break

        except Exception as e:
            logging.error(f"Error in DroneTrackingHeadlight run: {e}")
        finally:
            ws.close()
            cv.destroyAllWindows()
            logging.info("Resources released and WebSocket closed.")

def test_drone_tracking_headlight():
    """Test the DroneTrackingHeadlight class."""
    logging.info("Testing DroneTrackingHeadlight...")
    try:
        tracker = DroneTrackingHeadlight()
        tracker.run()
        logging.info("DroneTrackingHeadlight test completed successfully.\n")
    except Exception as e:
        logging.error(f"DroneTrackingHeadlight test failed: {e}")

if __name__ == "__main__":
    test_drone_tracking_headlight()