#!/bin/bash -l
#SBATCH -J SLG
#SBATCH -N 1
#SBATCH -G 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 2   # Cores assigned to each tasks
#SBATCH --time=0-10:00:00
#SBATCH -p gpu

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"
# Python 3.X by default (also on system)

nvidia-smi

module load lang/Python
source slg_env/bin/activate
module load  vis/FFmpeg
pip install --upgrade pip wheel
pip install pydub
pip install lightning-flash
pip install 'lightning-flash[audio,text]'
pip install --force-reinstall soundfile


python /home/users/gmenon/workspace/songsLyricsGenerator/src/torch_lightning_dali.py


wait $pid