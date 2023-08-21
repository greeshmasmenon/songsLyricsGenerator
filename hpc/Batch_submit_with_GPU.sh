#!/bin/bash -l
#SBATCH -J SLG 
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1   # Cores assigned to each tasks
#SBATCH --qos normal
#SBATCH --time=0-5:00:00
#SBATCH --partition=gpu
#SBATCH -A Christoph.Schommer
#SBATCH --mail-type=all
#SBATCH --mail-user=greeshmaseetharaman@gmail.com



print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"
# Python 3.X by default (also on system)

nvidia-smi

module load lang/Python
#source slg_env/bin/activate
module load  vis/FFmpeg
python3 -m venv slg_finetuned
source slg_finetuned/bin/activate
pip install --upgrade pip wheel
pip install pydub
pip install lightning-flash
pip install 'lightning-flash[audio,text]'
pip install --force-reinstall soundfile
pip install datasets, transformers, argparse
pip install dali-dataset
pip install wandb
pip install jiwer
pip install pyctcdecode
pip install https://github.com/kpu/kenlm/archive/master.zip


cd home/users/gmenon/workspace/songsLyricsGenerator/src
srun python3 /home/users/gmenon/workspace/songsLyricsGenerator/src/wav2vec2_training.py
#srun python /home/users/gmenon/workspace/songsLyricsGenerator/src/training/lyrics_finetune.py