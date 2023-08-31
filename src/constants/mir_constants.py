from dataclasses import dataclass
import torch
from torch.optim import lr_scheduler

@dataclass(frozen=True)
class HPCDataConstants:
    """constants for the use within the Songs to Lyrics generation"""
    NAME: str = "Dali Dataset"
    DATASET_PATH: str = "/home/users/gmenon/workspace/songsLyricsGenerator/data/DALI_v1.0/"
    DATASET_INFO_GZ: str = "/home/users/gmenon/workspace/songsLyricsGenerator/data/DALI_v1.0/info/DALI_DATA_INFO.gz"
    DATASET_INFO_CSV: str = "/home/users/gmenon/workspace/songsLyricsGenerator/data/DALI_v1.0/info/dali_info.csv"
    AUDIO_FILE_PATH: str = "/home/users/gmenon/workspace/songsLyricsGenerator/data/DALI_v1.0/audio/"
    DATASET_METADATA: str = ""
    DATASET_CLEANED_METADATA: str = ""


@dataclass(frozen=True)
class MacDataConstants:
    """constants for the use within the Songs to Lyrics generation"""
    NAME: str = "Dali Dataset"
    DATASET_PATH: str = "/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/"
    DATASET_INFO_GZ: str = "/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/info/DALI_DATA_INFO.gz"
    DATASET_INFO_CSV: str = "/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/info/dali_info.csv"
    AUDIO_FILE_PATH: str = "/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/audiodownload/"
    DATASET_METADATA: str = ""
    DATASET_CLEANED_METADATA: str = ""


@dataclass
class TrainingArgs:
    TRAIN_FILE_PATH: str = "/home/users/gmenon/notebooks/home/users/gmenon/notebooks/train_metadata_cleaned.csv"
    TEST_FILE_PATH: str = "/home/users/gmenon/notebooks/home/users/gmenon/notebooks/validation_metadata_cleaned.csv"
    MODEL_BACKBONE: str = "facebook/wav2vec2-base-960h"
    BATCH_SIZE: int = 1
    NUM_EPOCHS: int = 15
    NUM_GPUS = torch.cuda.device_count()
    MODEL_SAVE_PATH: str = "model_artefacts/finetuned_wav2vec2_large_xlsr_53_english_model.pt"
    FINETUNE_STRATEGY: str = "freeze"
    ACCUMULATE_GRAD_BATCHES = 8
    PRECISION = 16
    MAX_EPOCHS = 20
    NUM_NODES = 1
    CHARS_TO_REMOVE_FROM_TRANSCRIPTS = '[\,\?\.\!\-\;\:\"\%\$\&\^\*\@\#\<\>\/\+\\=\_\\}\{\)\(\]\[\`1234567890]'
    LR_SCHEDULER : str = 'reduce_on_plateau_schedule'


# WAV2VEC2_ARGS = TrainingArgs(MODEL_BACKBONE='jonatasgrosman/wav2vec2-large-xlsr-53-english',#"facebook/wav2vec2-large-960h-lv60-self", #"facebook/wav2vec2-large-robust-ft-libri-960h"
#                              TRAIN_FILE_PATH = "/home/users/gmenon/notebooks/home/users/gmenon/notebooks/train_song_metadata_en_demucs_cleaned.csv",
#                              TEST_FILE_PATH = "/home/users/gmenon/notebooks/home/users/gmenon/notebooks/validation_song_metadata_en_demucs_cleaned.csv",
#                               MODEL_SAVE_PATH="/home/users/gmenon/workspace/songsLyricsGenerator/src/model_artefacts/wav2vec2_demucs_en_finetuned_model.pt",
#                               BATCH_SIZE = 2,
#                               NUM_EPOCHS = 2,
#                               FINETUNE_STRATEGY = "no_freeze_deepspeed"
#                               ) # type: ignore

WAV2VEC2_ARGS = TrainingArgs(MODEL_BACKBONE='facebook/wav2vec2-large-960h-lv60-self',#"facebook/wav2vec2-large-960h-lv60-self", #"facebook/wav2vec2-large-robust-ft-libri-960h"
                             TRAIN_FILE_PATH = "/scratch/users/gmenon/train_song_metadata_en_demucs_cleaned_filtered_095.csv",
                             TEST_FILE_PATH = "/scratch/users/gmenon/validation_song_metadata_en_demucs_cleaned_filtered_005.csv",
                              MODEL_SAVE_PATH="/scratch/users/gmenon//model_artefacts/wav2vec2_demucs_en_large-960h-lv60-self_freeze_unfreeze_15epochs_adamw.pt",
                              BATCH_SIZE = 1,
                              NUM_EPOCHS = 15,
                              FINETUNE_STRATEGY = ('freeze_unfreeze', 5)
                              ) # type: ignore

# WAV2VEC2_ARGS = TrainingArgs(MODEL_BACKBONE='jonatasgrosman/wav2vec2-large-xlsr-53-english',#"facebook/wav2vec2-large-960h-lv60-self", #"facebook/wav2vec2-large-robust-ft-libri-960h"
#                              TRAIN_FILE_PATH = "/scratch/users/gmenon/train_song_metadata_en_demucs_cleaned_095.csv",
#                              TEST_FILE_PATH = "/scratch/users/gmenon/validation_song_metadata_en_demucs_cleaned_005.csv",
#                               MODEL_SAVE_PATH="/scratch/users/gmenon//model_artefacts/wav2vec2_demucs_en_wav2vec2-large-xlsr-53-english_zero_shot.pt",
#                               BATCH_SIZE = 2,
#                               NUM_EPOCHS = 2,
#                               FINETUNE_STRATEGY = "no_freeze_deepspeed"
#                               ) # type: ignore

