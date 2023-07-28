#!/bin/bash -l
#SBATCH -J SLG
#SBATCH -N 2
#SBATCH -G 6
#SBATCH --ntasks-per-node=2
#SBATCH -c 2   # Cores assigned to each tasks
#SBATCH --time=0-6:00:00
#SBATCH -p gpu
#SBATCH -A Christoph.Schommer
#SBATCH --qos normal
#SBATCH --mail-type=all
#SBATCH --mail-user=greeshmaseetharaman@gmail.com


print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"
# Python 3.X by default (also on system)
module load lang/Python
source slg_env/bin/activate
module load  vis/FFmpeg


jupyter notebook --ip $(hostname -i) --no-browser  & pid=$!
sleep 5s
jupyter notebook list
jupyter --paths
jupyter kernelspec list
echo "Enter this command on your laptop: ssh -p 8022 -NL 8888:$(hostname -i):8888 ${USER}@access-iris.uni.lu " > notebook.log
wait $pid