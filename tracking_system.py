from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist

class TrackingSystem:

    # This initializes the tracking system.
    # Arguments for this includes the maximum number of consecutive
    # frames an object can be missing before it is deregistered.
    def __init__(self, max_disappeared = 50):

        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    # This registers a new object with the tracking system.
    def register(self, centroid):

        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    # This deregisters an object from the tracking system.
    def deregister(self, object_id):

        del self.objects[object_id]
        del self.disappeared[object_id]

    # This updates the tracking system with new detections.
    # Arguments for this includes a list of detections.
    # This returns current tracked objects.
    def update(self, detections):
        
        if len(detections) == 0:
            # This marks all current objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects
        
        # This extracts centroids from detection.
        input_centroids = np.array([det["centroid"] for det in detections])

        # If there are no existing objects that are tracked, register all centroids.
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:

            # This is to match existing objects to new detections.
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # This is to compute the distance matrix.
            distance_matrix = dist.cdist(np.array(object_centroids), input_centroids)

            # This is to find the smallest distance for each object.
            rows = distance_matrix.min(axis = 1).argsort()
            cols = distance_matrix.argmin(axis = 1)[rows]

            # This is to keep track of matched rows and columns.
            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if distance_matrix[row, col] > self.max_disappeared:
                    continue

                # This is to update the centroid of the matched object.
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # This is to check for unmatched rows (disappeared objects).
            unused_rows = set(range(len(object_ids))) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # This is to check for unmatched columns (new objects).
            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

    
