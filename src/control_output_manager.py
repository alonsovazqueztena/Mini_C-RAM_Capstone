# Alonso Vazquez Tena
# STG-452: Capstone Project II
# February 3, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8
# (used as starter code for basic functionality).


# This class is largely incomplete. 
# This is a placeholder for the ControlOutputManager class.

# This class is responsible for managing the control outputs.
class ControlOutputManager:
    """Manages control outputs based on tracking data."""
    
    # This initializes the control output manager.
    def __init__(
            self, laser_control = None, 
            camera_control = None):
        """Initialize the control output manager.
        
        Keyword arguments:
        self -- instance of the control output manager,
        laser_control -- control variable of the laser,
        camera_control -- control variable of the camera,
        laser_state -- state of the laser,
        system_power -- state of the system power.
        """
        self.laser_control = laser_control
        self.camera_control = camera_control
        self.laser_state = False
        self.system_power = False

    # This updates visual indicators or performs 
    # based on tracking system data.
    def update_from_tracking(
            self, tracked_objects):

        # If there are no tracked objects, a message is printed.
        if not tracked_objects:
            print(
                "There are no objects being tracked."
                )
            return
        
        # This iterates through the tracked objects.
        for object_id, centroid in tracked_objects.items():
            x_center, y_center = centroid

            print(
                f"Tracked Object ID: {object_id}, Centroid: ({x_center:.2f}, {y_center:.2f})"
                )

            # This is to update visual indicators.
            self.update_visual_indicator(
                object_id, centroid)

            # This is to perform additional actions if necessary.
            self.perform_action_based_on_location(
                object_id, centroid)