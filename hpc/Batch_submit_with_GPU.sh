#!/bin/bash -l
#SBATCH -J SLG
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1   # Cores assigned to each tasks
#SBATCH --time=0-20:00:00

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"
# Python 3.X by default (also on system)
module load lang/Python
source slg_env/bin/activate
module load  vis/FFmpeg
pip install pydub

python /home/users/gmenon/workspace/songsLyricsGenerator/src/data_preparation.py


wait $pid