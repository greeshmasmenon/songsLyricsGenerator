from dataclasses import dataclass
import torch


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
    BATCH_SIZE: int = 2
    NUM_EPOCHS: int = 20
    NUM_GPUS = torch.cuda.device_count()
    MODEL_SAVE_PATH: str = "model_artefacts/finetuned_ALT_model.pt"
    FINETUNE_STRATEGY: str = "freeze"
    ACCUMULATE_GRAD_BATCHES = 2
    PRECISION = 16
    MAX_EPOCHS = 10


WAV2VEC2_ARGS = TrainingArgs(MODEL_BACKBONE="facebook/wav2vec2-large-960h-lv60-self",
                              MODEL_SAVE_PATH="/home/users/gmenon/workspace/songsLyricsGenerator/src/model_artefacts/wav2vec2_finetuned_model.pt"
                              )
