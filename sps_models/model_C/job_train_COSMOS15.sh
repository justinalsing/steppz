#!/bin/bash -l

#SBATCH -J train_C_COSMOS15
#SBATCH -t 80:00:00
#SBATCH -A cops
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sinan.deger@fysik.su.se

ml unload conda/02
ml cuda
source ~/.local-co/bin/setup-environment.sh
conda activate steppz_fsps

ulimit -s unlimited  # Unlimited stack
ulimit -u 16000      # Increase max number of tasks
ulimit -n 65536      # Increase max number of open files

python3 train_emulator_COSMOS15.py '/cfs/home/side0330/projects/steppz_flexible_sfh/' 0 6
python3 train_emulator_COSMOS15.py '/cfs/home/side0330/projects/steppz_flexible_sfh/' 6 12
python3 train_emulator_COSMOS15.py '/cfs/home/side0330/projects/steppz_flexible_sfh/' 12 18
python3 train_emulator_COSMOS15.py '/cfs/home/side0330/projects/steppz_flexible_sfh/' 18 24
python3 train_emulator_COSMOS15.py '/cfs/home/side0330/projects/steppz_flexible_sfh/' 24 26
