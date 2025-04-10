#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=96:00:00
#SBATCH --job-name=cylinder3D

module load release/24.04 GCC/12.3.0
module load OpenMPI/4.1.5
module load OpenFOAM/v2312 
source $FOAM_BASH

# define mesh level
mesh="0"

# path to the test case (adjust if necessary)
cd "cylinder_3D_Re3900_mesh${mesh}/"


# execute simulation
./Allrun &> "log.main"
cd ..

# create archive for download
tar zcf "cylinder_3D_Re3900_mesh${mesh}.tar.gz" "cylinder_3D_Re3900_mesh${mesh}"


# mpirun -np 64 pimpleFoam -postProcess -parallel &> log.post_process
