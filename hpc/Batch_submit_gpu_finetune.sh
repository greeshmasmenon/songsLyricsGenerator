#!/bin/bash -l
#SBATCH -J SLG 
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1   # Cores assigned to each tasks
#SBATCH --qos normal
#SBATCH --time=0-6:00:00
#SBATCH --partition=gpu
#SBATCH -A Christoph.Schommer
#SBATCH --mail-type=all
#SBATCH --mail-user=greeshmaseetharaman@gmail.com



print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"
# Python 3.X by default (also on system)

nvidia-smi

module load lang/Python
module load  vis/FFmpeg
python3 -m venv slg_finetun
source slg_finetun/bin/activate
pip install --upgrade pip wheel
pip install pydub
pip install lightning-flash
pip install 'lightning-flash[audio,text]'
pip install --force-reinstall soundfile
pip install datasets, transformers, argparse
pip install dali-dataset
pip install wandb
pip install jiwer

cd /home/users/gmenon/workspace/songsLyricsGenerator/src

#srun python /home/users/gmenon/workspace/songsLyricsGenerator/src/wav2vec2_training.py
srun python3 /home/users/gmenon/workspace/songsLyricsGenerator/src/training/lyrics_finetune.py