# src/DMX_frame_pipeline.py
# Daniel Saravia
# STG-452: Capstone Project II
# March 28, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a17189-ca30-800e-858d-aac289e6cb56


import concurrent.futures
import logging
import cv2 as cv
import socket
import websocket

from ai_model_interface import AIModelInterface
from detection_processor import DetectionProcessor
from frame_processor import FrameProcessor
from tracking_system import TrackingSystem
from video_stream_manager import VideoStreamManager

class FramePipeline:
    """A pipeline that captures frames from a video stream, processes them, 
    runs YOLO + detection filtering, then tracks objects over time.
    Additionally, it uses the latest detection's centroid to control a moving head light via DMX."""
    
    def __init__(
        self,
        capture_device=1,
        frame_width=1920,
        frame_height=1080,
        target_width=1920,
        target_height=1080,
        model_path="drone_detector_12x.pt",
        confidence_threshold=0.5,
        detection_processor=None,
        tracking_system=None
    ):
        # Video and processing setup.
        self.video_stream = VideoStreamManager(
            capture_device=capture_device, 
            frame_width=frame_width, 
            frame_height=frame_height
        )
        self.frame_processor = FrameProcessor(
            target_width=target_width, 
            target_height=target_height
        )
        self.ai_model_interface = AIModelInterface(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        self.detection_processor = detection_processor or DetectionProcessor(
            target_classes=None
        )
        self.tracking_system = tracking_system or TrackingSystem(
            max_disappeared=50, 
            max_distance=50
        )
        
        # Initialize DMX control parameters to 0.
        # Pan: 0° to 540° (DMX 0-255), Tilt: 0° to 205° (DMX 0-255).
        # Starting at 0 degrees for both.
        self.current_pan = 0.0    
        self.current_tilt = 0.0   
        
        # Proportional gain factors (tune these for your system)
        # Lower gains help reduce aggressive corrections.
        self.k_pan = 0.005    # reduced pan gain: degrees per pixel error in x
        self.k_tilt = 0.005    # tilt gain remains the same
        
        # Maximum allowed change per update (damping factor)
        self.max_delta_pan = 2.5   # reduced maximum degrees change per frame for pan
        self.max_delta_tilt = 2.5   # maximum degrees change per frame for tilt
        
        # Initialize DMX connection via QLC+ WebSocket.
        self.init_dmx_connection()
    
    def init_dmx_connection(self):
        """Initializes the WebSocket connection to the QLC+ DMX controller."""
        try:
            self.QLC_IP = self.get_host_ip()
            self.QLC_WS_URL = f"ws://{self.QLC_IP}:9999/qlcplusWS"
            self.ws = websocket.WebSocket()
            self.ws.connect(self.QLC_WS_URL)
            logging.info(f"Connected to QLC+ WebSocket at {self.QLC_WS_URL}.")
        except Exception as e:
            logging.error(f"Error initializing DMX connection: {e}")
            self.ws = None

    def get_host_ip(self):
        """Retrieve the local host IP address."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception as e:
            logging.error(f"Error determining host IP: {e}")
            return "127.0.0.1"

    def send_dmx_value(self, channel, value):
        """Send a DMX value to the specified channel through the WebSocket connection."""
        if self.ws is not None:
            try:
                int_value = int(value)
                self.ws.send(f"CH|{channel}|{int_value}")
            except Exception as e:
                logging.error(f"Error sending DMX value: {e}")

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and centroids on the frame."""
        for det in detections:
            bbox = det["bbox"]
            confidence = det["confidence"]
            x_min, y_min, x_max, y_max = map(int, bbox)
            label = f"drone {confidence:.3f}"
            font = cv.FONT_HERSHEY_TRIPLEX
            font_scale = 2
            thickness = 4
            (text_width, text_height), baseline = cv.getTextSize(label, font, font_scale, thickness)
            margin = 5
            cv.rectangle(frame,
                         (x_min, y_min - text_height - baseline - margin),
                         (x_min + text_width, y_min),
                         (0, 0, 0), -1)
            cv.putText(frame, label, (x_min, y_min - margin), font, font_scale, (0, 255, 0), thickness)

    def draw_tracked_objects(self, frame, tracked_objects):
        """Draws tracked object bounding boxes and centroids."""
        for detection in tracked_objects.values():
            bbox = detection["bbox"]
            x_min, y_min, x_max, y_max = map(int, bbox)
            cx, cy = detection["centroid"]
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)

    def update_dmx_control(self, centroid, frame_dims):
        """Calculates the error between the centroid and the frame center,
        computes a damped adjustment, updates the pan and tilt angles,
        and sends new DMX values."""
        frame_width, frame_height = frame_dims
        center_x, center_y = frame_width / 2, frame_height / 2
        error_x = centroid[0] - center_x  # positive if object is to the right
        error_y = centroid[1] - center_y  # positive if object is below center

        # Compute raw adjustments based on proportional gain.
        delta_pan = self.k_pan * error_x
        delta_tilt = self.k_tilt * error_y

        # Clamp the adjustments to the maximum allowed change per update.
        if delta_pan > self.max_delta_pan:
            delta_pan = self.max_delta_pan
        elif delta_pan < -self.max_delta_pan:
            delta_pan = -self.max_delta_pan

        if delta_tilt > self.max_delta_tilt:
            delta_tilt = self.max_delta_tilt
        elif delta_tilt < -self.max_delta_tilt:
            delta_tilt = -self.max_delta_tilt

        # Update the current angles.
        self.current_pan -= delta_pan
        self.current_tilt -= delta_tilt

        # Clamp angles to valid ranges.
        self.current_pan = max(0, min(self.current_pan, 540))
        self.current_tilt = max(0, min(self.current_tilt, 205))

        # Convert angles to DMX values.
        dmx_pan = (self.current_pan / 540.0) * 255.0
        dmx_tilt = (self.current_tilt / 205.0) * 255.0

        # Send DMX values on channel 1 (pan) and channel 3 (tilt).
        self.send_dmx_value(1, dmx_pan)
        self.send_dmx_value(3, dmx_tilt)

    def run(self):
        """Captures frames, runs preprocessing + AI + detection processing,
        updates the tracking system, draws annotations, and controls the moving head light."""
        try:
            with self.video_stream as stream, concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                cv.namedWindow("Mini C-RAM View", cv.WINDOW_NORMAL)
                cv.resizeWindow("Mini C-RAM View", 800, 600)
                cv.setWindowProperty("Mini C-RAM View", cv.WND_PROP_TOPMOST, 1)

                while True:
                    frame = stream.get_frame()
                    if frame is None:
                        logging.warning("No frame captured. Exiting the pipeline.")
                        break

                    # Preprocess frame and run AI model.
                    processed_frame = self.frame_processor.preprocess_frame(frame)
                    future = executor.submit(self.ai_model_interface.predict, processed_frame[0])
                    raw_detections = future.result()
                    processed_detections = self.detection_processor.process_detections(raw_detections)
                    tracked_objects = self.tracking_system.update(processed_detections)

                    # Draw detection and tracking annotations.
                    self.draw_detections(frame, processed_detections)
                    self.draw_tracked_objects(frame, tracked_objects)

                    # Use only the latest detection's centroid to update DMX control.
                    if processed_detections:
                        latest_det = processed_detections[-1]
                        centroid = latest_det["centroid"]
                        self.update_dmx_control(centroid, (stream.frame_width, stream.frame_height))

                    cv.imshow("Mini C-RAM View", frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except Exception as e:
            logging.error(f"Error in FramePipeline run: {e}")
        
        finally:
            self.video_stream.release_stream()
            cv.destroyAllWindows()
            if self.ws is not None:
                try:
                    self.ws.close()
                    logging.info("DMX WebSocket closed.")
                except Exception as e:
                    logging.error(f"Error closing DMX WebSocket: {e}")
