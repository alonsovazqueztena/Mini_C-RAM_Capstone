# Alonso Vazquez Tena | STG-452: Capstone Project II | April 4, 2025
# Source: https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8, https://grok.com/share/bGVnYWN5_7008140a-9936-4b7f-b83d-0760c7ea866c
# Daniel Saravia Source: https://grok.com/share/bGVnYWN5_52adc247-cde4-41e4-80bd-c70ef0c81dc9

from ultralytics import YOLO # YOLO model for object detection
import torch # PyTorch for GPU support
from contextlib import redirect_stdout # Suppress output when importing pygame
import os # OS module for file operations
with redirect_stdout(open(os.devnull, 'w')):
    import pygame # Use pygame for audio playback
import time # Time module for time operations
import threading # Threading for non-blocking audio playback

class AIModelInterface:
    """Optimized interface for YOLO drone detection."""
    def __init__(self, model_path="drone_detector_12n.pt", confidence_threshold=0.5, audio_path="drone_detected.mp3"):
        """Initialize YOLO model with minimal setup."""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # Choose GPU if available, else CPU
        self.audio_path = audio_path
        self.last_alert_time = 0 # Initialize last alert timestamp
        self.alert_timeout = 3 # Set minimum seconds between alerts
        self.audio_channel = None # Initialize audio channel variable
        pygame.mixer.init() # Initialize pygame mixer for audio playback
        pygame.mixer.set_num_channels(1) # Limit to one audio channel

    def _play_audio(self):
        """Play the pre-existing audio file in a separate thread."""
        try:
            sound = pygame.mixer.Sound(self.audio_path) # Load sound file
            self.audio_channel = sound.play() # Start audio playback
            if self.audio_channel is None:
                print("Skipping playback") # Inform audio did not play
        except pygame.error as e:
            print(f"Error playing audio: {e}") # Print error if audio playback fails

    def _trigger_alert(self):
        """Trigger the voice alert if the timeout has passed."""
        current_time = time.time() # Get current timestamp
        if current_time - self.last_alert_time >= self.alert_timeout:  # Check if timeout passed
            self.last_alert_time = current_time # Update last alert time
            threading.Thread(target=self._play_audio, daemon=True).start() # Start audio in new thread

    def predict(self, frame):
        """Run inference and return minimal detection data."""
        results = self.model.predict(
            source=frame, # Input frame for detection.
            imgsz=640, # Resize frame to 640 pixels.
            conf=self.confidence_threshold, # Use defined confidence threshold.
            half=True, # Use half precision for faster inference.
            device=self.device, # Run on GPU or CPU.
            verbose=False # Disable logging.
        )
        detections = [] # Initialize empty list for detections
        for result in results:
            if result.boxes is None:
                continue # Skip if no boxes detected.
            for box in result.boxes:
                if self.model.names[int(box.cls[0])] != "drone":
                    continue
                bbox = box.xyxy[0].tolist() # Extract bounding box coordinates as list.
                centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) # Calculate centroid.
                detections.append({"bbox": bbox, "centroid": centroid}) # Append detection info.
        if detections:
            self._trigger_alert() # Trigger alert if drone detected.
        return detections