# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:39:15 2026

@author: dr19382
"""

#Functions to read and write the hippo and foam files to be imported and kept common between programs
import os
import numpy as np
import pyvista as pv
import vtk
import re
import itertools
import torch

#Read/Write Hippo:
def HippoWrite(FileLocation, Variable, Fluid_Temp_Sample):
    prelude = "     = "

    with open(FileLocation, "r") as f:
        lines = f.readlines()
    
    line_written = False
    updated_lines = []
    for line in lines:
        # Check if line starts with the key (no spaces)
        if line.startswith(Variable[0]):
            if line_written==False:
                # Split into two parts: key and existing value
                parts = line.strip().split()
                new_value = prelude + ' '.join([str(Fluid_Temp_Sample)])
                # Replace value (second item)
                #if len(parts) > 1:
                parts[2:] = []
                parts[1] = str(new_value)
                #else:
                    # If value is missing, add one
                    #parts.append(str(new_value))
                
                # Rebuild the line
                new_line = " ".join(parts) + "\n"
                updated_lines.append(new_line)
                line_written=True
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    #Write updated files
    with open(FileLocation, "w") as f:
        f.writelines(updated_lines)
    return()

#Read/Write Foam:
def FoamWrite(FileLocation, Variables, Fluid_Flow_Sample):
    Seperation = 3
    with open(FileLocation, "r") as f:
        lines = f.readlines()
    InternalField = 'internalField'
    updated_lines = []
    Counter = 0
    StartCount=False
    for line in lines:
        # Check if line starts with the key (no spaces)
        if Variables[0] in line:
            StartCount = True
            updated_lines.append(line)
            print('detector triggered at:', line)
            Counter = Counter + 1
        elif Counter == Seperation:
            # Split into parts: key and existing value
            parts = line.strip().split()
            new_value = ' '.join([str(Fluid_Flow_Sample)])
            # Replace value
            parts[3] = str(new_value)

            # Rebuild the line
            new_line = " ".join(parts) + "\n"
            updated_lines.append(new_line)
            Counter=Counter+1
        elif StartCount == True:
            Counter = Counter + 1
            updated_lines.append(line)
        elif InternalField in line:
            parts = line.strip().split()
            new_value = ' '.join([str(Fluid_Flow_Sample)])
            # Replace value
            parts[3] = str(new_value)
            # Rebuild the line
            new_line = " ".join(parts) + "\n"
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)

    with open(FileLocation, "w") as f:
        f.writelines(updated_lines)
    return()

# def HippoExodusReader(FileName, output=False):
#    #Read exodus file and set last timestep
#    MainPath = FileName #os.path.join(foldername, FileName)
#    MainExo = pv.read(MainPath)
#    MainExo.set_active_time_step(MainExo.n_timesteps - 1)
   
#    if output==True:
#        #Check type is multiblock 
#        print(MainPath)
#        print(type(MainExo))
#        print("Blocks:", list(MainExo.keys()))
#        print("N blocks:", len(MainExo))
    
#        #Check data
#        print("Point arrays:", list(MainExo["Element Blocks"]))
#        print("Nodes:", list(MainExo["Node Sets"]))
       
#    #Check individual meshes
#    elem_sets = MainExo["Element Blocks"]
   
#    if output==True:
#        print(type(elem_sets))
#        print(len(elem_sets))

#    #Try to get the block (not coil or vacuum)
#    mesh_block = elem_sets[1]

#    #Extract Data
#    block_coords = mesh_block.points
#    block_T = mesh_block.point_data['T']  
    
#    return(block_coords, block_T)

def HippoExodusReader(FileName, output=False):
    ExoRead = pv.get_reader(FileName)
    ExoRead.set_active_time_point(ExoRead.number_time_points-1)
        
    if output:
            print("Time points:", ExoRead.number_time_points)
            print("Using:", ExoRead.active_time_point)
    
    #Read data for this timestep ONLY
    data = ExoRead.read()

    #Access 'Element Blocks' → 'target'
    elem_blocks = data["Element Blocks"]
    TargetBlock = elem_blocks["target"]

    #Extract coords and temperature
    block_coords = TargetBlock.points
    block_T = TargetBlock.point_data["T"]

    if output:
        print("Coords shape:", block_coords.shape)
        print("T shape:", block_T.shape)

    return block_coords, block_T

def HippoExodusReader_Mesh(FileName, output=False):
    # Load Exodus
    ExoRead = pv.get_reader(FileName)
    ExoRead.set_active_time_point(ExoRead.number_time_points - 1)

    data = ExoRead.read()

    # Access the block
    elem_blocks = data["Element Blocks"]
    TargetBlock = elem_blocks["target"]

    # Node coordinates + temperature
    coords = TargetBlock.points
    T = TargetBlock.point_data["T"]

    # Element connectivity (PyVista stores in a flattened cell array)
    cells = TargetBlock.cells
    celltypes = TargetBlock.celltypes  # useful if you have mixed elements

    # Convert cells → list of element node lists
    # Each hex begins with an 8
    connectivity = []
    i = 0
    while i < len(cells):
        N = cells[i]      # number of nodes in this element
        nodes = cells[i+1:i+1+N]
        connectivity.append(nodes)
        i += 1 + N

    # Build edges: all unique node pairs from each element
    edges = set()
    for elem_nodes in connectivity:
        for u, v in itertools.combinations(elem_nodes, 2):
            edges.add((u, v))
            edges.add((v, u))     # bidirectional

    # Convert to torch edge_index format (2, E)
    edge_index = torch.tensor(list(edges), dtype=torch.long).T

    if output:
        print("Coords:", coords.shape)
        print("T:", T.shape)
        print("Num elements:", len(connectivity))
        print("Num edges:", edge_index.shape[1])

    return coords, T, edge_index

def FoamExodusReader(Foam_Path, FoamFile, output=False):
    FoamPath = os.path.join(Foam_Path, FoamFile)
    FoamOut_1 = pv.OpenFOAMReader(FoamPath)

    #Check the time values and grab the last one
    if output==True:
        print("Available time steps:", FoamOut_1.time_values)
    last_time = FoamOut_1.time_values[-1]
    FoamOut_1.set_active_time_value(last_time)
    mesh_last = FoamOut_1.read()

    #find internal_mesh and read the points and T's
    internal_foam = mesh_last[0]
    foam_points = internal_foam.points
    if output==True:
        print(foam_points.shape)
        print(foam_points)
    foam_T = internal_foam.point_data["T"]
    if output==True:
        print(foam_T.shape)
        print(foam_T)
    return(foam_points, foam_T)