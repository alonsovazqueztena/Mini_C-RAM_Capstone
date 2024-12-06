import pygame

class ControllerInputManager:

    def __init__(self):

        # This initializes the controller input manager.
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("There is no controller connected. Please connect an Xbox controller.")

        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

        print(f"Controller connected: {self.controller.get_name()}")

    # This reads inputs from the Xbox controller.
    # This returns a dictionary of commands.
    def get_input(self):

        commands = {
            # This is to move in the x-axis and y-axis.
            "move_camera": (0, 0),

            # Zooming in would be positive, zooming out would be negative.
            "zoom": 0,

            # This must be true to toggle the laser.
            "toggle_laser": False,

            # This goes "on" or "off" for system power control.
            "system_power": None
        }

        # This is to process events.
        pygame.event.pump()

        # This is to read joystick axes for camera movement.
        x_axis = self.controller.get_axis(0) # Left stick horizontal
        y_axis = self.controller.get_axis(1) # Left stick vertical
        commands["move_camera"] = (x_axis, y_axis)

        # This reads triggers for zoom control.
        right_trigger = self.controller.get_axis(5) # Right triger (zoom in)
        left_trigger = self.controller.get_axis(4) # Left trigger (zoom out)

        if right_trigger > 0.1:
            commands["zoom"] = right_trigger
        elif left_trigger > 0.1:
            commands["zoom"] = -left_trigger

        # This is to read buttons for laser toggle and system power control.
        if self.controller.get_button(0):   # "A" button to toggle laser
            commands["toggle_laser"] = True

        if self.controller.get_button(7): # "Start" button to turn on the system
            commands["system_power"] = "on"

        if self.controller.get_button(6): # "Back" button to turn off the system
            commands["system_power"] = "off"

        return commands
    
    # This is to shut down the controller input manager and pygame.
    def shutdown(self):
        pygame.quit()
        print("The controller input manager has shut down.")
                  