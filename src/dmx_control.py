"""
Daniel Saravia
STG-452: Capstone Project II
February 2, 2025
Citations:
- `pynput`: https://pypi.org/project/pynput/
- `websocket-client`: https://pypi.org/project/websocket-client/
- Python `socket` library documentation: https://docs.python.org/3/library/socket.html

This script connects to a QLC+ WebSocket server and allows the user to control pan and tilt DMX values using arrow keys.

website to complete this assignment:
https://chatgpt.com/share/67bb92eb-d380-8012-8681-535bc6395a02
(used as starter code for basic functionality).
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import websocket
import socket
import time

def get_host_ip():
    """Retrieve the local host computer's IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:
        print(f"Error determining host IP: {e}")
        return "127.0.0.1"

# Establish QLC+ WebSocket connection
QLC_IP = get_host_ip()
QLC_WS_URL = f"ws://{QLC_IP}:9999/qlcplusWS"
ws = websocket.WebSocket()
ws.connect(QLC_WS_URL)
print(f"Connected to QLC+ WebSocket at {QLC_WS_URL}.")

def send_dmx_value(channel, value):
    """Send a DMX value to the specified channel through the WebSocket connection."""
    ws.send(f"CH|{channel}|{int(value)}")
    print(f"Sent: CH|{channel}|{int(value)}")

def dmx_to_xyz(dmx_pan, dmx_tilt):
    """
    Converts DMX channel values to a 3D unit vector representing the beam direction.
    Assumes:
      - DMX channel 1 (pan): 0-255 maps to 0° to 540° rotation about the Z-axis.
      - DMX channel 3 (tilt): 0-255 maps to 0° to 205° rotation about the X-axis.
      - The initial beam vector (before rotations) is [0, 1, 0] (pointing forward).
      - Tilt is applied first, then pan.
    """
    # Map DMX values to angles in degrees
    pan_deg = (dmx_pan / 255.0) * 540.0
    tilt_deg = (dmx_tilt / 255.0) * 205.0

    # Convert degrees to radians
    pan_rad = np.deg2rad(pan_deg)
    tilt_rad = np.deg2rad(tilt_deg)
    
    # The fixture's local beam vector (pointing forward)
    beam_local = np.array([0, 1, 0])
    
    # Rotation for tilt (about the X-axis)
    Rx = np.array([[1,              0,               0],
                   [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
                   [0, np.sin(tilt_rad),  np.cos(tilt_rad)]])
    
    # Rotation for pan (about the Z-axis)
    Rz = np.array([[ np.cos(pan_rad), -np.sin(pan_rad), 0],
                   [ np.sin(pan_rad),  np.cos(pan_rad), 0],
                   [              0,               0, 1]])
    
    # Apply tilt first, then pan
    beam_global = Rz @ (Rx @ beam_local)
    return beam_global

# --- Visualization Setup ---
# Create a grid of DMX values for channels 1 and 3 (for the background field)
dmx_values = np.linspace(0, 255, num=50)
pan_grid, tilt_grid = np.meshgrid(dmx_values, dmx_values)
x_grid = np.empty_like(pan_grid)
y_grid = np.empty_like(pan_grid)
z_grid = np.empty_like(pan_grid)

for i in range(pan_grid.shape[0]):
    for j in range(pan_grid.shape[1]):
        vec = dmx_to_xyz(pan_grid[i, j], tilt_grid[i, j])
        x_grid[i, j] = vec[0]
        y_grid[i, j] = vec[1]
        z_grid[i, j] = vec[2]

# Compute the reference great circle path (with tilt = 0)
t_vals = np.linspace(0, 2*np.pi, 200)
x_path = -np.sin(t_vals)
y_path = np.cos(t_vals)
z_path = np.zeros_like(t_vals)

# Set up the interactive 3D plot
plt.ion()  # Interactive mode on
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x_grid, y_grid, z_grid, c=tilt_grid, cmap='viridis', s=10, alpha=0.3)
ax.plot(x_path, y_path, z_path, c='red', lw=2, label="Great Circle Path")

# Animated marker for current beam direction (dot)
beam_marker, = ax.plot([], [], [], 'ko', markersize=8, label="Current Beam")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("LED Moving Head Light DMX Mapping\nwith Great Circle Scan & Light Cone")
ax.legend()
plt.draw()

# Initialize cone surface handle
cone_surface = None

# --- Automatic Movement along the Great Circle ---
# We let tilt remain fixed at 0, and let pan vary (oscillate) between 0° and 360°.
# DMX mapping: pan_DMx = (desired_pan_deg/540)*255.
t_deg = 0
direction = 1   # 1 for increasing, -1 for decreasing
step = 2        # degrees per iteration

# Light cone parameters
cone_length = 1.2           # Length of the cone (from the origin)
cone_half_angle = np.deg2rad(15)  # Half-angle of the cone (in radians)

try:
    while True:
        # Update pan angle in degrees (oscillatory motion)
        t_deg += direction * step
        if t_deg >= 360:
            t_deg = 360
            direction = -1
        elif t_deg <= 0:
            t_deg = 0
            direction = 1

        # Compute DMX values: using tilt=0 for the horizontal great circle
        dmx_pan = (t_deg / 540) * 255   # Map desired pan to DMX value
        dmx_tilt = 0

        # Send DMX commands to QLC+ (channel 1: pan, channel 3: tilt)
        send_dmx_value(1, dmx_pan)
        send_dmx_value(3, dmx_tilt)
        
        # Compute current beam direction for visualization
        beam_dir = dmx_to_xyz(dmx_pan, dmx_tilt)
        beam_marker.set_data([beam_dir[0]], [beam_dir[1]])
        beam_marker.set_3d_properties([beam_dir[2]])

        # --- Compute and plot the light cone ---
        # Remove previous cone if it exists
        if cone_surface is not None:
            cone_surface.remove()
        
        # To create a cone from the origin along beam_dir, first build an orthonormal basis.
        axis = beam_dir
        # Choose an arbitrary vector not collinear with axis
        if abs(axis[2]) < 0.99:
            ref_vec = np.array([0, 0, 1])
        else:
            ref_vec = np.array([0, 1, 0])
        u = np.cross(axis, ref_vec)
        u = u / np.linalg.norm(u)
        v = np.cross(axis, u)
        
        # Create cone mesh in (z,phi) coordinates
        z_vals = np.linspace(0, cone_length, 20)  # distance along the cone
        phi_vals = np.linspace(0, 2*np.pi, 30)
        Z_mesh, Phi_mesh = np.meshgrid(z_vals, phi_vals)
        R_mesh = Z_mesh * np.tan(cone_half_angle)
        
        # Map to 3D coordinates: point = (z * beam_dir) + R*(cos(phi)*u + sin(phi)*v)
        X_cone = Z_mesh * axis[0] + R_mesh * (np.cos(Phi_mesh)*u[0] + np.sin(Phi_mesh)*v[0])
        Y_cone = Z_mesh * axis[1] + R_mesh * (np.cos(Phi_mesh)*u[1] + np.sin(Phi_mesh)*v[1])
        Z_cone = Z_mesh * axis[2] + R_mesh * (np.cos(Phi_mesh)*u[2] + np.sin(Phi_mesh)*v[2])
        
        cone_surface = ax.plot_surface(X_cone, Y_cone, Z_cone, alpha=0.3, color='yellow', rstride=1, cstride=1)
        
        plt.draw()
        plt.pause(0.01)
        time.sleep(0.05)  # adjust for update rate

except KeyboardInterrupt:
    print("Movement interrupted by user.")

finally:
    ws.close()
    print("WebSocket closed.")
