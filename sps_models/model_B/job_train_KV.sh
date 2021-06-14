#!/bin/bash -l

#SBATCH -J train_B_KV
#SBATCH -t 24:00:00
#SBATCH -A cops
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1

ml unload conda/02
ml cuda
source ~/.local-co/bin/setup-environment.sh

ulimit -s unlimited  # Unlimited stack
ulimit -u 16000      # Increase max number of tasks
ulimit -n 65536      # Increase max number of open files

python3 train_emulator_KV.py '/cfs/home/alju5794/steppz/sps_models/model_B/'
