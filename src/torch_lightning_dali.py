import flash
import torch
from flash.audio import SpeechRecognition, SpeechRecognitionData
from flash.core.data.utils import download_data
import pandas as pd
from dataclasses import dataclass


@dataclass
class training_args():
    TRAIN_FILE_PATH: str = "/home/users/gmenon/notebooks/home/users/gmenon/notebooks/train_metadata_cleaned.csv"
    TEST_FILE_PATH: str = "/home/users/gmenon/notebooks/home/users/gmenon/notebooks/validation_metadata_cleaned.csv"
    MODEL_BACKBONE: str = "facebook/wav2vec2-base-960h"
    # MODEL_BACKBONE =  "openai/whisper-medium", #English only
    # MODEL_BACKBONE =  "openai/whisper-large-v2" #All languages
    BATCH_SIZE: int = 8
    NUM_EPOCHS: int = 2
    NUM_GPUS = torch.cuda.device_count()
    MODEL_SAVE_PATH: str = "model_artefacts/finetuned_ALT_model.pt"
    FINETUNE_STRATEGY: str = "freeze"


whisper_args = training_args(MODEL_BACKBONE = "openai/whisper-medium",
                             MODEL_SAVE_PATH = "model_artefacts/whisper_finetuned_model.pt")

wav2vec2_args = training_args(MODEL_SAVE_PATH = "model_artefacts/wav2vec2_finetuned_model.pt")


# Dataset

print(f"Shape of the training file is {str(pd.read_csv(training_args.TRAIN_FILE_PATH).shape)}")
print(f"Shape of the validation files is {str(pd.read_csv(training_args.TEST_FILE_PATH).shape)}")


datamodule = SpeechRecognitionData.from_csv(
    "file_name",
    "transcription",
    train_file = training_args.TRAIN_FILE_PATH,
    test_file = training_args.TEST_FILE_PATH,
    batch_size = training_args.BATCH_SIZE
)

# Wav2Vec2.0
wav2vec2_model = SpeechRecognition(backbone=wav2vec2_args.MODEL_BACKBONE, MODEL_SAVE_PATH = wav2vec2_args.MODEL_SAVE_PATH)
# Create the trainer and finetune the model
wav2vec2_trainer = flash.Trainer(accumulate_grad_batches=10,
                        precision=16,
                        max_epochs=wav2vec2_args.NUM_EPOCHS, 
                        gpus=wav2vec2_args.NUM_GPUS)

wav2vec2_trainer.finetune(wav2vec2_model,
                 datamodule=datamodule, 
                 strategy=wav2vec2_args.FINETUNE_STRATEGY)

# 5. Save the model!
wav2vec2_trainer.save_checkpoint(wav2vec2_args.MODEL_SAVE_PATH)


# 2. Build the Whisper Task
whisper_model = SpeechRecognition(backbone=whisper_args.MODEL_BACKBONE, 
                          MODEL_SAVE_PATH = whisper_args.MODEL_SAVE_PATH)

# 3. Create the trainer and finetune the model
whisper_trainer = flash.Trainer(accumulate_grad_batches=10, precision=16, max_epochs=whisper_args.NUM_EPOCHS,
                        gpus=whisper_args.NUM_GPUS)

whisper_trainer.finetune(whisper_model, datamodule=datamodule, strategy=whisper_args.FINETUNE_STRATEGY)

# 5. Save the model!
whisper_trainer.save_checkpoint(whisper_args.MODEL_SAVE_PATH)


# 4. Predict on audio files!
test_datamodule = SpeechRecognitionData.from_files(predict_files=["/home/users/gmenon/workspace/songsLyricsGenerator/test_clip.wav"], 
                                              batch_size=training_args.BATCH_SIZE)

wav2vec2_predictions = wav2vec2_trainer.predict(wav2vec2_model, datamodule=test_datamodule)
print(wav2vec2_predictions)


whisper_predictions = whisper_trainer.predict(whisper_model, datamodule=test_datamodule)
print(whisper_predictions)
