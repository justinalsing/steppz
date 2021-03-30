#!/bin/bash -l

#SBATCH -J model_A_training_set
#SBATCH -t 12:00:00
#SBATCH -p cops
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64

ml unload openmpi3
ml conda mpich

mpiexec -np 64 python3 generate_spectra_mpi.py '/cfs/home/alju5794/steppz/sps_models/model_A/training_data'
mpiexec -np 64 python3 add_nebular_emission_mpi.py '/cfs/home/alju5794/steppz/sps_models/model_A/training_data'
