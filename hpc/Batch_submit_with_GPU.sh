#!/bin/bash -l
#SBATCH -J SLG 
#SBATCH -N 2
#SBATCH -G 8
#SBATCH --ntasks-per-node=2
#SBATCH -c 2   # Cores assigned to each tasks
#SBATCH -A Christoph.Schommer
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=all
#SBATCH --mail-user=greeshmaseetharaman@gmail.com
#SBATCH --qos=normal


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


srun python /home/users/gmenon/workspace/songsLyricsGenerator/src/torch_lightning_dali.py