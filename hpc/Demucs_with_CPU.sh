#!/bin/bash -l
#SBATCH -J SLG 
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 2   # Cores assigned to each tasks
#SBATCH --qos normal
#SBATCH --time=2-0:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=all
#SBATCH --mail-user=greeshmaseetharaman@gmail.com


# Python 3.X by default (also on system)
module load lang/Python
# source slg_env/bin/activate
module load  vis/FFmpeg
python3 -m venv slg_demucs
pip3 install -U demucs

# cd /home/users/gmenon/workspace/songsLyricsGenerator/data/DALI_v1.0/audio/wav_clips/

cd /scratch/users/gmenon/wav_clips/

for i in *.wav;
do
 demucs -d cpu --two-stems=vocals "$i" ;
  echo "$i" > separated.txt;
done