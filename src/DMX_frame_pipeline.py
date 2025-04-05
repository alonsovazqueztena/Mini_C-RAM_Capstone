# Daniel Saravia Source: https://grok.com/share/bGVnYWN5_52adc247-cde4-41e4-80bd-c70ef0c81dc9
# DMX_frame_pipeline.py
import socket
import concurrent
import websocket
import cv2 as cv
from frame_pipeline import FramePipeline

class DMXFramePipeline(FramePipeline):
    """Extends FramePipeline with DMX control for drone tracking."""

    def __init__(self, model_path="drone_detector_12n.pt", confidence_threshold=0.5):
        """Initialize base pipeline and add DMX-specific setup."""
        super().__init__(model_path, confidence_threshold)  # Inherit video, AI, tracking

        # DMX setup
        self.pan = 0.0
        self.tilt = 0.0
        self.k_pan = 0.005
        self.k_tilt = 0.005
        self.max_delta = 2.5
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
        except Exception:
            self.ws = None

    def send_dmx(self, channel, value):
        """Send DMX value with no error checking."""
        if self.ws:
            self.ws.send(f"CH|{channel}|{int(value)}")

    def draw(self, frame, detections, tracked_objects):
        """Minimal drawing for speed (overrides base draw)."""
        for det in detections:
            x_min, y_min, x_max, y_max = map(int, det["bbox"])
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        for obj in tracked_objects.values():
            cx, cy = map(int, obj["centroid"])
            cv.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

    def update_dmx(self, centroid, frame):
        """Fast DMX update."""
        h, w = frame.shape[:2]
        error_x = centroid[0] - w / 2
        error_y = centroid[1] - h / 2

        delta_pan = max(-self.max_delta, min(self.k_pan * error_x, self.max_delta))
        delta_tilt = max(-self.max_delta, min(self.k_tilt * error_y, self.max_delta))

        self.pan = max(0, min(self.pan - delta_pan, 540))
        self.tilt = max(0, min(self.tilt - delta_tilt, 205))

        self.send_dmx(1, (self.pan / 540.0) * 255.0)
        self.send_dmx(3, (self.tilt / 205.0) * 255.0)

    def run(self):
        """Run with DMX control and optimized settings."""
        try:
            with self.video_stream as stream, concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                cv.namedWindow("View", cv.WINDOW_NORMAL)
                cv.resizeWindow("View", 640, 480)
                while True:
                    frame = stream.get_frame()
                    if frame is None:
                        break
                    future = executor.submit(self.ai_model_interface.predict, frame)
                    detections = future.result()
                    tracked_objects = self.tracking_system.update(detections)
                    self.draw(frame, detections, tracked_objects)
                    if detections:
                        self.update_dmx(detections[-1]["centroid"], frame)
                    cv.imshow("View", frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.video_stream.release_stream()
            cv.destroyAllWindows()
            if self.ws:
                self.ws.close()