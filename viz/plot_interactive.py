import pyvista as pv
import numpy as np
import glob
import sys

from fields import load, speed

def load_data(filename):
    N, u, v, w = load(filename)
    return speed(u, v, w)

files = sorted(glob.glob('output/vel_*.bin'))
if not files:
    print("No output files found in output/")
    sys.exit(1)

# Load the initial frame
data = load_data(files[0])
grid = pv.ImageData()
grid.dimensions = data.shape
grid.spacing = (1.0, 1.0, 1.0)
grid.point_data["Velocity Magnitude"] = data.flatten(order="F")

# Do NOT use off_screen=True here so the interactive window pops up!
plotter = pv.Plotter()

vol = plotter.add_volume(grid, scalars="Velocity Magnitude", cmap="inferno", opacity="sigmoid", show_scalar_bar=True)
plotter.camera_position = 'iso'
plotter.add_axes()
plotter.add_text("Interactive 3D Navier-Stokes\nUse the slider to scrub through time!", font_size=12, position="upper_left")

def update_time(value):
    idx = int(round(value))
    if idx < 0 or idx >= len(files):
        return
    
    # Load the specific frame chosen by the slider
    new_data = load_data(files[idx])
    grid.point_data["Velocity Magnitude"] = new_data.flatten(order="F")
    
    # Force VTK to update the volume rendering by replacing the actor
    global vol
    plotter.remove_actor(vol)
    vol = plotter.add_volume(grid, scalars="Velocity Magnitude", cmap="inferno", opacity="sigmoid", show_scalar_bar=False)

# Add a scrub slider for the time frames
plotter.add_slider_widget(
    update_time, 
    [0, len(files) - 1], 
    value=0, 
    title="Time Frame", 
    pointa=(0.25, 0.1), 
    pointb=(0.75, 0.1), 
    style='modern'
)

print("Opening interactive 3D viewer...")
print("You can rotate the volume with left-click, pan with middle-click, and zoom with the scroll wheel.")
print("Use the slider at the bottom to animate through the time steps!")

# This opens the native desktop window and blocks until you close it
plotter.show()
