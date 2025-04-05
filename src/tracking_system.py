# Alonso Vazquez Tena | STG-452: Capstone Project II | March 16, 2025
# Source: https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8, https://chatgpt.com/share/67d77b29-c824-800e-ab25-2cc850596046.

from collections import OrderedDict # For ordered storage of tracked objects.
import numpy as np # For numerical operations.
from scipy.optimize import linear_sum_assignment # Hungarian algorithm for optimal matching.
from scipy.spatial import distance as dist # To compute distances between centroids

class TrackingSystem:
    """Multi-object tracking system using centroid-based matching."""

    def __init__(self, max_disappeared=50, max_distance=50, smoothing_alpha=0.5):
        """Initialize tracking with disappearance and distance limits."""
        self.next_object_id = 0 # Next object ID to assign
        self.objects = OrderedDict() # Dict for current tracked objects
        self.disappeared = OrderedDict() # Dict for tracking disappearance counts
        self.max_disappeared = max_disappeared # Max frames object can vanish before removal
        self.max_distance = max_distance # Max distance for matching detections
        self.smoothing_alpha = smoothing_alpha # Smoothing factor for updates

    def register(self, detection):
        """Register a new detecton as a tracked object."""
        detection['trajectory'] = [detection['centroid']] # Initialize trajectory.
        detection['velocity'] = (0, 0) # Initialize velocity.
        self.objects[self.next_object_id] = detection # Add object.
        self.disappeared[self.next_object_id] = 0 # Reset disappearance count.
        self.next_object_id += 1 # Increment ID counter

    def deregister(self, object_id):
        """Remove a tracked object."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """Update tracked objects with new detections."""
        if len(detections) == 0: # No detections: update disappearance counts
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array([d["centroid"] for d in detections]) # Get new centroids

        if len(self.objects) == 0: # No objects: register all detections 
            for det in detections:
                self.register(det)
        else:
            object_ids = list(self.objects.keys()) # Get current object IDs.
            object_centroids = np.array([self.objects[obj_id]["centroid"] for obj_id in object_ids]) # Get centroids of tracked objects.
            distance_matrix = dist.cdist(object_centroids, input_centroids) # Compute distance matrix between tracked and new centroids.
            row_indices, col_indices = linear_sum_assignment(distance_matrix) # Optimal assignment using Hungarian algorithm.
            used_rows, used_cols = set(), set() # Track matched rows and columns.
            for row, col in zip(row_indices, col_indices):
                if row in used_rows or col in used_cols: # Skip if row or column already matched.
                    continue
                if distance_matrix[row, col] > self.max_distance: # Skip if row or column already matched.
                    continue
                object_id = object_ids[row] # Get the object ID.
                prev_centroid = self.objects[object_id]["centroid"] # Previous centroid.
                new_centroid = detections[col]["centroid"] # New detection centroid
                smoothed_centroid = (self.smoothing_alpha * new_centroid[0] + (1 - self.smoothing_alpha) * prev_centroid[0], self.smoothing_alpha * new_centroid[1] + (1 - self.smoothing_alpha) * prev_centroid[1]) # Smooth centroid.
                raw_velocity = (smoothed_centroid[0] - prev_centroid[0], smoothed_centroid[1] - prev_centroid[1]) # Calculate raw velocity.
                prev_velocity = self.objects[object_id]["velocity"] # Previous velocity.
                smoothed_velocity = (self.smoothing_alpha * raw_velocity[0] + (1 - self.smoothing_alpha) * prev_velocity[0], self.smoothing_alpha * raw_velocity[1] + (1 - self.smoothing_alpha) * prev_velocity[1]) # Smooth velocity.
                detections[col]["velocity"] = smoothed_velocity # Update detection with new velocity.
                trajectory = self.objects[object_id].get("trajectory", []) # Get existing trajectory.
                trajectory.append(smoothed_centroid) # Append new centroid to trajectory.
                detections[col]["trajectory"] = trajectory # Update detection with trajectory.
                detections[col]["centroid"] = smoothed_centroid # Update detection with smoothed centroid.
                self.objects[object_id] = detections[col] # Update tracked object.
                self.disappeared[object_id] = 0 # Reset disappearance count.
                used_rows.add(row) # Mark row as used.
                used_cols.add(col) # Mark column as used.
            unused_rows = set(range(0, distance_matrix.shape[0])).difference(used_rows) # Unmatched tracked objects.
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1 # Incremenet disappearance count.
                if self.disappeared[object_id] > self.max_disappeared: # Remove if count exceeds threshold.
                    self.deregister(object_id)
            unused_cols = set(range(0, distance_matrix.shape[1])).difference(used_cols) # Unmatched detections.
            for col in unused_cols:
                self.register(detections[col]) # Register new objects.
        return self.objects