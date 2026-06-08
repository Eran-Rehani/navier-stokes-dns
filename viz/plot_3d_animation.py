import pyvista as pv
import numpy as np
import glob
import sys
import os

from fields import load, speed

def load_data(filename):
    N, u, v, w = load(filename)
    return speed(u, v, w)

files = sorted(glob.glob('output/vel_*.bin'))
if not files:
    print("No output files found in output/")
    sys.exit(1)

# Read first frame to initialize grid
data = load_data(files[0])
grid = pv.ImageData()
grid.dimensions = data.shape
grid.spacing = (1.0, 1.0, 1.0)
# Flatten with Fortran order to match PyVista's internal indexing
grid.point_data["Velocity Magnitude"] = data.flatten(order="F")

plotter = pv.Plotter(off_screen=True)

# Add solid cylinder visual to represent the Immersed Boundary Volume Penalty
N = grid.dimensions[0]
r_idx = 0.5 * N / (2 * np.pi)
cyl = pv.Cylinder(center=(N/2, N/2, N/2), direction=(0, 0, 1), radius=r_idx, height=N)
plotter.add_mesh(cyl, color='silver', opacity=0.3)

# Volume rendering maps the 3D scalar field to color and opacity
# 'sigmoid' makes lower values transparent and higher values opaque
vol = plotter.add_volume(grid, scalars="Velocity Magnitude", cmap="inferno", 
                         opacity="sigmoid", show_scalar_bar=True)

plotter.camera_position = 'iso'
plotter.add_axes()
plotter.add_text("3D Navier-Stokes DNS\nTaylor-Green Vortex", font_size=12, position="upper_left")

out_file = "output/fluid_animation.gif"
plotter.open_gif(out_file)

print(f"Generating 3D volume rendering animation with {len(files)} frames...")

for i, f in enumerate(files):
    data = load_data(f)
    grid.point_data["Velocity Magnitude"] = data.flatten(order="F")
    
    plotter.remove_actor(vol)
    vol = plotter.add_volume(grid, scalars="Velocity Magnitude", cmap="inferno", opacity="sigmoid", show_scalar_bar=False)
    
    plotter.write_frame()
    print(f"  Processed frame {i+1}/{len(files)}: {f}")

plotter.close()
print(f"\nSaved 3D animation to {out_file}")
