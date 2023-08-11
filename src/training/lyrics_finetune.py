import pytorch_lightning as pl
import wandb
import os 
from pytorch_lightning.loggers import WandbLogger
from constants.mir_constants import WAV2VEC2_ARGS,TrainingArgs
import pandas as pd

import torch
from torch import nn
from torch.utils.data import random_split,DataLoader
from datasets import Dataset
from datasets import load_dataset, Dataset, Audio, load_metric, load_dataset, Metric
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import TrainingArguments, Trainer, Wav2Vec2ForCTC
from typing import Optional
import re

# @dataclass
# class TrainingArgs:
#     TRAIN_FILE_PATH: str = "/home/users/gmenon/notebooks/home/users/gmenon/notebooks/train_metadata_cleaned.csv"
#     TEST_FILE_PATH: str = "/home/users/gmenon/notebooks/home/users/gmenon/notebooks/validation_metadata_cleaned.csv"
#     MODEL_BACKBONE: str = "facebook/wav2vec2-base-960h"
#     BATCH_SIZE: int = 2
#     NUM_EPOCHS: int = 4
#     NUM_GPUS = torch.cuda.device_count()
#     MODEL_SAVE_PATH: str = "model_artefacts/finetuned_ALT_model.pt"
#     FINETUNE_STRATEGY: str = "freeze"
#     ACCUMULATE_GRAD_BATCHES = 16
#     PRECISION = 16
#     MAX_EPOCHS = 10
#     NUM_NODES = 1


# WAV2VEC2_ARGS = TrainingArgs(MODEL_BACKBONE="facebook/wav2vec2-large-960h-lv60-self", #"facebook/wav2vec2-large-robust-ft-libri-960h"
#                              TRAIN_FILE_PATH = "/home/users/gmenon/notebooks/home/users/gmenon/notebooks/train_song_metadata_en_demucs_cleaned.csv",
#                              TEST_FILE_PATH = "/home/users/gmenon/notebooks/home/users/gmenon/notebooks/validation_song_metadata_en_demucs_cleaned.csv",
#                               MODEL_SAVE_PATH="/home/users/gmenon/workspace/songsLyricsGenerator/src/model_artefacts/wav2vec2_demucs_en_finetuned_model.pt",
#                               BATCH_SIZE = 1,
#                               NUM_EPOCHS = 10,
#                               FINETUNE_STRATEGY = "no_freeze_deepspeed"
#                               )


class DALIDataset(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4,
                  train_path :Optional[str] = None,
                    validation_path: Optional[str] = None,
                      model_backbone: pl.LightningModule = None,
                      args: TrainingArgs = WAV2VEC2_ARGS
                      ):
        
        super().__init__()
        self.train_path = train_path if validation_path is not None else args.TRAIN_FILE_PATH
        self.validation_path = validation_path if validation_path is not None else args.TEST_FILE_PATH
        self.model_backbone = model_backbone if model_backbone is not None else args.MODEL_BACKBONE

        def prepare_data(self):
            pass
        
        def setup(self):
            train_df = pd.read_csv(WAV2VEC2_ARGS.TRAIN_FILE_PATH) 
            validation_df = pd.read_csv(WAV2VEC2_ARGS.TEST_FILE_PATH)
            songs_metadata = pd.concat([train_df,validation_df], ignore_index = True)
            audio_dataset = Dataset.from_dict(
                {"audio": list(songs_metadata["file_name"]),
                 "transcription": list(songs_metadata["transcription"])}).cast_column("audio", Audio(sampling_rate=16_000))
            audio_dataset["transcription"] = audio_dataset["transcription"] = re.sub(WAV2VEC2_ARGS.CHARS_TO_REMOVE_FROM_TRANSCRIPTS, '', audio_dataset["transcription"]).upper()
            audio_dataset = audio_dataset.train_test_split(test_size=0.2, shuffle=True)

