import os
import numpy as np
import pyvista as pv
import vtk
import csv

# Path to your Exodus file
file_path = []
file_path.append(r"\\wsl.localhost\Ubuntu-22.04\home\kfletch123\GeneralFolder\HIVEsim\HIVE\HIVE_MFEM_Repo\Baseline_HTC\THeat_Flow_TV_HTC_ex.e")  # Replace with your actual file path
file_path.append(r"\\wsl.localhost\Ubuntu-22.04\home\kfletch123\GeneralFolder\HIVEsim\HIVE\HIVE_MFEM_Repo\HTC_VmatT\THeat_Flow_TV_Vmat_HTC_ex.e")
file_path.append(r"\\wsl.localhost\Ubuntu-22.04\home\kfletch123\GeneralFolder\HIVEsim\HIVE\HIVE_MFEM_Repo\HTC_VmatTE\THeat_Flow_TV_Vmat_HTC_Emod_ex.e")

all_thermocouples = []

for file in file_path:
    ExoRead = pv.get_reader(file)
    ExoRead.set_active_time_point(ExoRead.number_time_points-1)
    
    TargetBlock = ExoRead.read()["Element Blocks"]["target"]
    
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
    
    plotter = pv.Plotter(off_screen=True)
    
    modified_points = [TargetBlock.points[TargetBlock.find_closest_point(p)] for p in Points]
    pset = pv.PointSet(modified_points)
    
    pset = pset.sample(TargetBlock, snap_to_closest_point=True)

    csv_path = os.path.splitext(os.path.basename(file))[0] + ".csv"
    
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "T"])  # header
    
        T = pset["T"]  # extract the array once for clarity
    
        for p, t in zip(pset.points, T):
            writer.writerow([p[0], p[1], p[2], t])
    
    print(pset["T"])
    
    plotter = pv.Plotter(off_screen=True)
    
    plotter.add_mesh(
        TargetBlock,
        scalars="T",
        show_edges=False,
        cmap="viridis"
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
    
    fname = os.path.splitext(os.path.basename(file))[0] + ".png"
    plotter.screenshot(fname)
    
#Write all points to CSV
csv_path = "points.csv"

# Write points to CSV

