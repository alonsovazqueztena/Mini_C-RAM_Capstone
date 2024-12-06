class ControlOutputManager:
    
    # This initializes the control output manager.
    # Arguments include functions to control the laser state
    # and to control camera movement and zoom.
    def __init__(self, laser_control = None, camera_control = None):

        self.laser_control = laser_control
        self.camera_control = camera_control
        self.laser_state = False
        self.system_power = False

    # This updates visual indicators or performs based on tracking system data.
    # Arguments include that of a dictionary of tracked objects where
    # the key is the object ID and the value is its centroid (x_center, y_center).
    def update_from_tracking(self, tracked_objects):

        if not tracked_objects:
            print("There are no objects being tracked.")
            return
        
        for object_id, centroid in tracked_objects.items():
            x_center, y_center = centroid

            print(f"Tracked Object ID: {object_id}, Centroid: ({x_center:.2f}, {y_center:.2f})")

            # This is to update visual indicators.
            self.update_visual_indicator(object_id, centroid)

            # This is to perform additional actions if necessary.
            self.perform_action_based_on_location(object_id, centroid)

        
