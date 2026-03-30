import os
import numpy as np
import pyvista as pv
import vtk

import matplotlib.pyplot as plt

# Path to your Exodus file
file_path = "THeat_Flow_TV_Vmat_HTC_ex.e"  # Replace with your actual file path

# reader = pv.get_reader(file_path)
# reader.set_active_time_value(reader.time_values[-1])
# Dataset = reader.read()

# # Show all available block names
# print("Available block names:")
# for name in Dataset.keys():
#     print(f" - {name}")

# mesh = Dataset["Node Sets"]
# mesh.plot(show_edges=True)
# print(mesh)
# print("Number of points:", mesh.n_points)
# print("Number of cells:", mesh.n_cells)

ExoRead = pv.get_reader(file_path)
ExoRead.set_active_time_point(ExoRead.number_time_points-1)

TargetBlock = ExoRead.read()["Element Blocks"]["monoblock"]

T_target = TargetBlock.point_data["T"]

Points = [[0.0116, -0.0245, 0.0194],
    [0.0138, -0.0245, 0.0013],
    [0.0067, -0.0245, 0.0124],
    [0.0110, 0.0245, 0.0031],
    [-0.0105, 0.0245, -0.0050],
    [-0.0058, 0.0245, 0.0171],
    [-0.0180, -0.0006, 0.0164],
    [-0.0180, -0.0040, -0.0085],
    [-0.0180, -0.0047, 0.0073],
    [-0.0180, 0.0124, -0.0032]]

modified_points = [TargetBlock.points[TargetBlock.find_closest_point(p)] for p in Points]
pset = pv.PointSet(modified_points)

pset = pset.sample(TargetBlock, snap_to_closest_point=True)

print(pset['T'])

fig, ax = plt.subplots()
index = np.arange(len(pset['T']))

exp_data = [675.14,	489.81,	582.48,	480.46,	448.44,	608.38,	655.11,	461.47,	544.66,	470.33]
TC = [f"TC{i}" for i in range(10)]
bar_width = 0.35

ax.bar(index, exp_data, bar_width, label="Experiment")
ax.bar(index+bar_width, pset['T'], bar_width, label="MFEM")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(TC)
ax.set_ylabel("Temperature (K)")
ax.set_xlabel("Thermocouples")
ax.set_ylim(bottom=425)
ax.legend()

fig.savefig("Results.png")
plt.show()

plotter = pv.Plotter(off_screen=True)

plotter.add_mesh(
    TargetBlock,
    scalars="T",
    show_edges=False,
    cmap="jet"
)

plotter.add_mesh(
    pset,
    color="red",
    point_size=12,
    render_points_as_spheres=True
)

plotter.add_axes()

plotter.view_yz(negative=True)
plotter.camera.azimuth = -25
plotter.camera.elevation = 20

fname = os.path.splitext(os.path.basename(file_path))[0] + ".png"
plotter.screenshot(fname)

