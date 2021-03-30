#/bin/bash -l
#SBATCH -J model_A_training_set
#SBATCH -t 12:00:00
#SBATCH -p cops
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64

ml openmpi3

mpirun -np 64 python3 generate_spectra_mpi.py
mpirun -np 64 python3 add_nebular_emission_mpi.py
