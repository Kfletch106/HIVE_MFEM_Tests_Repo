#!/bin/bash

#Load OpenFOAM environment
source /home/kfletch123/SOFTWARE/hippo/external/openfoam/OpenFOAM-12/etc/bashrc
unset FOAM_SIGFPE

#Correct WSL path (Linux format, NOT Windows UNC)
cd /home/kfletch123/GeneralFolder/HIVEsim/HIVE/HIVE_MFEM_Repo/Baseline_HTC

set -v -e

#Run hippo
mpirun -n 8 ~/SOFTWARE/hippo/hippo-opt -i THeat_Flow_TV_HTC.i -w

