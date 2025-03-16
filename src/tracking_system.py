# Alonso Vazquez Tena
# STG-452: Capstone Project II
# February 3, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8
# (used as starter code for basic functionality).

# This import ensures there is order for the tracked objects.
from collections import OrderedDict

# Utilizing AI models requires the usage of arrays and matrices
# for data processing.
import numpy as np

# This import is used to calculate the distance between centroids.
from scipy.spatial import distance as dist

from scipy.optimize import linear_sum_assignment


# This class serves as a tracking system for multiple objects.
class TrackingSystem:
    """Creates a multi-object tracking system using centroid-based matching."""

    # This method initializes the tracking system.
    def __init__(
            self, max_disappeared=50, 
            max_distance=50,
            smoothing_alpha=0.5):
        """Initializes the tracking system.
        
        Keyword arguments:
        self -- instance of the tracking system,
        max_disappeared -- Maximum number of consecutive frames 
        an object may go missing before it is deregistered,
        max_distance -- Maximum allowed centroid distance 
        for matching an existing object to a new detection.
        """

        # This sets the default value for the next object's ID and
        # creates dictionaries to store objects and disappeared objects.
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.smoothing_alpha = smoothing_alpha

    # This method registers a new object in the tracking system.
    def register(
            self, detection):
        """Register a new object (detection) in the tracking system."""
        
        detection['trajectory'] = [detection['centroid']]
        detection['velocity'] = (0, 0)

        # The new object is stored in the objects dictionary.
        self.objects[self.next_object_id] = detection
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    # This method deregisters an object from the tracking system.
    def deregister(self, object_id):
        """Remove an object from the tracking system."""

        del self.objects[object_id]
        del self.disappeared[object_id]

    # This updates the tracked objects with new detection data.
    def update(self, detections):
        """Update tracked objects with new detection data."""

        # If there are no new detections, mark existing objects as disappeared.
        if len(detections) == 0:
            for object_id in list(
                    self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(
                        object_id)
            return self.objects

        # For any new detections, extract centroids.
        input_centroids = np.array(
            [d["centroid"] for d in detections])

        # If no objects are being tracked, register all new detections.
        if len(self.objects) == 0:
            for det in detections:
                self.register(
                    det)
        else:

            # Prepare to match current tracked objects to new detections via centroid distance.
            object_ids = list(
                self.objects.keys())
            object_centroids = np.array([
                self.objects[obj_id]["centroid"] for obj_id in object_ids])

            # Compute distance matrix between tracked centroids and new detection centroids.
            distance_matrix = dist.cdist(
                object_centroids, input_centroids)

            row_indices, col_indices = linear_sum_assignment(distance_matrix)

            # Keep track of matched rows & columns to avoid double assignment.
            used_rows, used_cols = set(), set()

            for row, col in zip(
                    row_indices, col_indices):
                
                # If we’ve already matched this row or column, skip.
                if row in used_rows or col in used_cols:
                    continue

                # If distance is too much, ignore this match.
                if distance_matrix[row, col] > self.max_distance:
                    continue

                # Update the tracked object with the new detection data.
                object_id = object_ids[row]
                prev_centroid = self.objects[object_id]["centroid"]
                new_centroid = detections[col]["centroid"]

                smoothed_centroid = (
                    self.smoothing_alpha * new_centroid[0] + (1 - self.smoothing_alpha) * prev_centroid[0],
                    self.smoothing_alpha * new_centroid[1] + (1 - self.smoothing_alpha) * prev_centroid[1]
                )

                raw_velocity = (
                    smoothed_centroid[0] - prev_centroid[0],
                    smoothed_centroid[1] - prev_centroid[1]
                )

                prev_velocity = self.objects[object_id]["velocity"]

                smoothed_velocity = (
                    self.smoothing_alpha * raw_velocity[0] + (1 - self.smoothing_alpha) * prev_velocity[0],
                    self.smoothing_alpha * raw_velocity[1] + (1 - self.smoothing_alpha) * prev_velocity[1]
                )
                
                detections[col]["velocity"] = smoothed_velocity
                
                trajectory = self.objects[object_id].get("trajectory", [])
                trajectory.append(smoothed_centroid)
                detections[col]["trajectory"] = trajectory

                detections[col]["centroid"] = smoothed_centroid
                
                self.objects[object_id] = detections[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # For any unmatched tracked objects, increment disappearance count.
            unused_rows = set(range(0, distance_matrix.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[
                    row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # For any unmatched detections, register them as new objects.
            unused_cols = set(range(0, distance_matrix.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(
                    detections[col])

        return self.objects