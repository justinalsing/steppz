#!/bin/bash -l

#SBATCH -J train_B_KV
#SBATCH -t 12:00:00
#SBATCH -p cops
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64

ml unload conda/02
source ~/.local-co/bin/setup-environment.sh

ulimit -s unlimited  # Unlimited stack
ulimit -u 16000      # Increase max number of tasks
ulimit -n 65536      # Increase max number of open files

python3 train_emulator_KV.py
