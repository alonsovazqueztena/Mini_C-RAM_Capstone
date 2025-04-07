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
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
    # Uncomment the next line if you want console logging:
    # print(f"Sent: CH|{channel}|{int(value)}")

def dmx_to_xyz(dmx_pan, dmx_tilt):
    """
    Converts DMX channel values to a 3D unit vector representing the beam direction.
    Assumes:
      - DMX channel 1 (pan): 0-255 maps to 0° to 540° rotation about the Z-axis.
      - DMX channel 3 (tilt): 0-255 maps to 0° to 205° rotation about the X-axis.
      - The initial beam vector (before rotations) is [0, 1, 0] (pointing forward).
      - Tilt is applied first, then pan.
    """
    pan_deg = (dmx_pan / 255.0) * 540.0
    tilt_deg = (dmx_tilt / 255.0) * 205.0
    pan_rad = np.deg2rad(pan_deg)
    tilt_rad = np.deg2rad(tilt_deg)
    beam_local = np.array([0, 1, 0])
    # Rotation about X (tilt)
    Rx = np.array([[1,              0,               0],
                   [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
                   [0, np.sin(tilt_rad),  np.cos(tilt_rad)]])
    # Rotation about Z (pan)
    Rz = np.array([[ np.cos(pan_rad), -np.sin(pan_rad), 0],
                   [ np.sin(pan_rad),  np.cos(pan_rad), 0],
                   [              0,               0, 1]])
    beam_global = Rz @ (Rx @ beam_local)
    return beam_global

def compute_cone_verts(axis, cone_length, cone_half_angle, num_z=20, num_phi=30):
    """
    Compute vertices for a cone originating at (0,0,0) along the given axis.
    Returns a list of quads (each a list of 4 (x,y,z) tuples) suitable for updating a Poly3DCollection.
    """
    z_vals = np.linspace(0, cone_length, num_z)
    phi_vals = np.linspace(0, 2*np.pi, num_phi)
    Z_mesh, Phi_mesh = np.meshgrid(z_vals, phi_vals)
    R_mesh = Z_mesh * np.tan(cone_half_angle)
    
    # Build an orthonormal basis for the beam direction (axis)
    if abs(axis[2]) < 0.99:
        ref_vec = np.array([0, 0, 1])
    else:
        ref_vec = np.array([0, 1, 0])
    u = np.cross(axis, ref_vec)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    
    # Map to 3D coordinates for the cone surface
    X_cone = Z_mesh * axis[0] + R_mesh * (np.cos(Phi_mesh)*u[0] + np.sin(Phi_mesh)*v[0])
    Y_cone = Z_mesh * axis[1] + R_mesh * (np.cos(Phi_mesh)*u[1] + np.sin(Phi_mesh)*v[1])
    Z_cone = Z_mesh * axis[2] + R_mesh * (np.cos(Phi_mesh)*u[2] + np.sin(Phi_mesh)*v[2])
    
    # Create quads for the Poly3DCollection
    verts = []
    nrows, ncols = X_cone.shape
    for i in range(nrows - 1):
        for j in range(ncols - 1):
            quad = [(X_cone[i, j], Y_cone[i, j], Z_cone[i, j]),
                    (X_cone[i+1, j], Y_cone[i+1, j], Z_cone[i+1, j]),
                    (X_cone[i+1, j+1], Y_cone[i+1, j+1], Z_cone[i+1, j+1]),
                    (X_cone[i, j+1], Y_cone[i, j+1], Z_cone[i, j+1])]
            verts.append(quad)
    return verts

# --- Static Visualization Setup ---
# Background DMX grid (remains unchanged)
dmx_values = np.linspace(0, 255, num=50)
pan_grid, tilt_grid = np.meshgrid(dmx_values, dmx_values)
x_grid = np.empty_like(pan_grid)
y_grid = np.empty_like(pan_grid)
z_grid = np.empty_like(pan_grid)
for i in range(pan_grid.shape[0]):
    for j in range(pan_grid.shape[1]):
        vec = dmx_to_xyz(pan_grid[i, j], tilt_grid[i, j])
        x_grid[i, j], y_grid[i, j], z_grid[i, j] = vec

# Set up the interactive 3D plot
plt.ion()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_grid, y_grid, z_grid, c=tilt_grid, cmap='viridis', s=10, alpha=0.3)

# --- Compute and Plot a Reference Lissajous Path ---
# Pan: 0-540° centered at 270°; Tilt is limited so that DMX (channel 3) never exceeds 128.
T_pan = 10    # seconds period for pan
T_tilt = 15   # seconds period for tilt
max_dmx_tilt = 128
max_tilt_deg = (max_dmx_tilt / 255.0) * 205.0  # ≈102.5°
t_ref = np.linspace(0, 30, 300)
lissajous_path = []
for t in t_ref:
    pan_deg = 270 + 270 * np.sin(2*np.pi*t / T_pan)
    tilt_deg = (max_tilt_deg/2) + (max_tilt_deg/2) * np.sin(2*np.pi*t / T_tilt + np.pi/2)
    dmx_pan = (pan_deg / 540) * 255
    dmx_tilt = (tilt_deg / 205) * 255
    vec = dmx_to_xyz(dmx_pan, dmx_tilt)
    lissajous_path.append(vec)
lissajous_path = np.array(lissajous_path)
ax.plot(lissajous_path[:,0], lissajous_path[:,1], lissajous_path[:,2],
        c='blue', lw=2, label="Lissajous Path (Tilt ≤128)")

# Animated marker for current beam direction
beam_marker, = ax.plot([], [], [], 'ko', markersize=8, label="Current Beam")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("LED Moving Head Light DMX Mapping\nwith Efficient Lissajous Scan & Light Cone")
ax.legend()
plt.draw()

# --- Precreate the Light Cone Collection ---
cone_length = 1.2
cone_half_angle = np.deg2rad(15)
# Start with an initial beam direction (using DMX values for pan=0 and tilt corresponding to half of max)
init_dmx_pan = (270 / 540) * 255
init_tilt_deg = max_tilt_deg / 2
init_dmx_tilt = (init_tilt_deg / 205) * 255
init_beam = dmx_to_xyz(init_dmx_pan, init_dmx_tilt)
cone_verts = compute_cone_verts(init_beam, cone_length, cone_half_angle)
cone_surface = Poly3DCollection(cone_verts, facecolor='yellow', alpha=0.3)
ax.add_collection3d(cone_surface)

# --- Automatic Movement using Lissajous Pattern with Tilt Limited ---
start_time = time.time()
try:
    while True:
        t = time.time() - start_time
        # Lissajous equations (angles in degrees)
        pan_deg = 270 + 270 * np.sin(2 * np.pi * t / T_pan)
        tilt_deg = (max_tilt_deg/2) + (max_tilt_deg/2) * np.sin(2 * np.pi * t / T_tilt + np.pi/2)
        dmx_pan = (pan_deg / 540) * 255
        dmx_tilt = (tilt_deg / 205) * 255  # will not exceed 128
        
        # Send DMX commands
        send_dmx_value(1, dmx_pan)
        send_dmx_value(3, dmx_tilt)
        
        # Compute current beam direction and update marker
        beam_dir = dmx_to_xyz(dmx_pan, dmx_tilt)
        beam_marker.set_data([beam_dir[0]], [beam_dir[1]])
        beam_marker.set_3d_properties([beam_dir[2]])
        
        # Update the light cone by computing new vertices and updating the existing collection
        new_verts = compute_cone_verts(beam_dir, cone_length, cone_half_angle)
        cone_surface.set_verts(new_verts)
        
        # Instead of full redraw, we just pause briefly to let the GUI update
        plt.pause(0.005)
        
except KeyboardInterrupt:
    print("Movement interrupted by user.")
    
finally:
    ws.close()
    print("WebSocket closed.")
