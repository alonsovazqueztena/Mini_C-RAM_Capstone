# Alonso Vazquez Tena | STG-452: Capstone Project II | April 4, 2025
# Source: https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8
# Daniel Saravia Source: https://grok.com/share/bGVnYWN5_52adc247-cde4-41e4-80bd-c70ef0c81dc9
# ai_model_interface.py
from ultralytics import YOLO
import torch
from contextlib import redirect_stdout
import os
with redirect_stdout(open(os.devnull, 'w')):
    import pygame
import time
import threading

class AIModelInterface:
    """Optimized interface for YOLO drone detection."""

    def __init__(self, model_path="drone_detector_12n.pt", confidence_threshold=0.5, audio_path="drone_detected.mp3"):
        """Initialize YOLO model with minimal setup."""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_path = audio_path
        self.last_alert_time = 0
        self.alert_timeout = 3
        self.audio_channel = None
        pygame.mixer.init()
        pygame.mixer.set_num_channels(1)

    def _play_audio(self):
        """Play the pre-existing audio file in a separate thread."""
        try:
            sound = pygame.mixer.Sound(self.audio_path)
            self.audio_channel = sound.play()
            if self.audio_channel is None:
                print("Skipping playback")
        except pygame.error as e:
            print(f"Error playing audio: {e}")

    def _trigger_alert(self):
        """Trigger the voice alert if the timeout has passed."""
        current_time = time.time()
        if current_time - self.last_alert_time >= self.alert_timeout:
            self.last_alert_time = current_time
            # Run audio playback in a separate thread to avoid blocking
            threading.Thread(target=self._play_audio, daemon=True).start()

    def predict(self, frame):
        """Run inference and return minimal detection data."""
        results = self.model.predict(
            source=frame,
            imgsz=640,
            conf=self.confidence_threshold,
            half=True,
            device=self.device,
            verbose=False
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if self.model.names[int(box.cls[0])] != "drone":
                    continue
                bbox = box.xyxy[0].tolist()
                centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                detections.append({"bbox": bbox, "centroid": centroid})

            # If any drones are detected, trigger the voice alert
        if detections:
            self._trigger_alert()
        return detections