#!/bin/bash -l
#SBATCH -J SLG
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -C volta32
#SBATCH --ntasks-per-node=2
#SBATCH -c 4   # Cores assigned to each tasks
#SBATCH --time=0-8:00:00
#SBATCH -p gpu
#SBATCH -A Christoph.Schommer
#SBATCH --qos normal
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
pip install git+https://github.com/huggingface/huggingface_hub
pip install 'huggingface_hub[cli,torch]'
pip install git+https://github.com/huggingface/transformers


cd /scratch/users/gmenon/hf_seq2seq/hubert-xlarge-bart-base

#srun python main.py # Use this if you have python in your environment
srun python run_speech_recognition_seq2seq.py \
	--dataset_name="gmenon/slt-lyrics-audio" \
	--model_name_or_path="/scratch/users/gmenon/hf_seq2seq/hubert-xlarge-bart-base/" \
	--train_split_name="train" \
	--eval_split_name="eval" \
	--output_dir="/scratch/users/gmenon/hf_seq2seq/hubert-xlarge-bart-base/outputs/" \
	--preprocessing_num_workers="4" \
	--length_column_name="input_length" \
	--overwrite_output_dir \
	--num_train_epochs="5" \
	--per_device_train_batch_size="1" \
	--per_device_eval_batch_size="1" \
	--gradient_accumulation_steps="1" \
	--learning_rate="3e-4" \
	--warmup_steps="400" \
	--evaluation_strategy="steps" \
	--save_steps="400" \
	--eval_steps="400" \
	--logging_steps="10" \
	--save_total_limit="1" \
	--gradient_checkpointing \
	--fp16 \
	--group_by_length \
	--predict_with_generate \
	--generation_max_length="40" \
	--generation_num_beams="1" \
	--do_train --do_eval \
	--do_lower_case \
  --text_column_name "transcription"





echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"


