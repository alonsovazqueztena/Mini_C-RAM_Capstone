from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist

class TrackingSystem:
    """
    Simple multi-object tracking system using centroid-based matching.
    """

    def __init__(self, max_disappeared=50, max_distance=50):
        """
        Args:
            max_disappeared (int): Maximum number of consecutive frames 
                an object may go missing before it is deregistered.
            max_distance (float): Maximum allowed centroid distance 
                for matching an existing object to a new detection.
        """
        self.next_object_id = 0
        self.objects = OrderedDict()      # object_id -> detection dict
        self.disappeared = OrderedDict()  # object_id -> number of consecutive missed frames
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, detection):
        """
        Register a new object (detection) in the tracking system.
        
        Args:
            detection (dict): A single detection dict containing at least
                              { 'centroid': (x_center, y_center), 'bbox': [...], ... }
        """
        self.objects[self.next_object_id] = detection
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """
        Remove an object from the tracking system.
        """
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        Update tracked objects with new detection data.

        Args:
            detections (List[dict]): Each dict is typically produced by
                DetectionProcessor and includes at least:
                {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "confidence": float,
                    "class_id": int,
                    "centroid": (x_center, y_center)
                }

        Returns:
            OrderedDict: Current tracked objects as {object_id: detection_dict}
        """
        # 1. If there are no new detections, mark existing objects as disappeared.
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # 2. Extract centroids from the new detections.
        input_centroids = np.array([d["centroid"] for d in detections])

        # 3. If no objects are being tracked, register all new detections.
        if len(self.objects) == 0:
            for det in detections:
                self.register(det)
        else:
            # 4. Prepare to match current tracked objects to new detections via centroid distance.
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id]["centroid"] for obj_id in object_ids]
            object_centroids = np.array(object_centroids)

            # 5. Compute distance matrix between tracked centroids and new detection centroids.
            distance_matrix = dist.cdist(object_centroids, input_centroids)

            # 6. For each tracked object, find the closest new detection in ascending order.
            rows = distance_matrix.min(axis=1).argsort()
            cols = distance_matrix.argmin(axis=1)[rows]

            # Keep track of matched rows & columns to avoid double assignment.
            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                # If we’ve already matched this row or column, skip.
                if row in used_rows or col in used_cols:
                    continue

                # If distance is too great, ignore this match.
                if distance_matrix[row, col] > self.max_distance:
                    continue

                # Update the tracked object with the new detection data.
                object_id = object_ids[row]
                self.objects[object_id] = detections[col]  # store full detection dict
                self.disappeared[object_id] = 0            # reset disappearance count

                used_rows.add(row)
                used_cols.add(col)

            # 7. For any unmatched tracked objects, increment disappearance count.
            unused_rows = set(range(0, distance_matrix.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # 8. For any unmatched detections, register them as new objects.
            unused_cols = set(range(0, distance_matrix.shape[1])) - used_cols
            for col in unused_cols:
                self.register(detections[col])

        return self.objects


    
