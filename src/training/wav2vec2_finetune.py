import flash
from flash.audio import SpeechRecognition, SpeechRecognitionData
import pandas as pd
from training.basetrainer import SpeechModel
from constants.mir_constants import TrainingArgs
from typing import List
from typing import Dict

__all__ = ["Wav2Vec2SpeechRecognition"]


class Wav2Vec2SpeechRecognition(SpeechModel):
    def __init__(self, wav2vec2_args: TrainingArgs) -> None:
        super().__init__(wav2vec2_args)
        self.WAV2VEC2_ARGS = self.training_args
        self.wav2vec2_model = SpeechRecognition(backbone=self.WAV2VEC2_ARGS.MODEL_BACKBONE)
        self.wav2vec2_trainer = flash.Trainer(accumulate_grad_batches=self.WAV2VEC2_ARGS.ACCUMULATE_GRAD_BATCHES,
                                              precision=self.WAV2VEC2_ARGS.PRECISION,
                                              max_epochs=self.WAV2VEC2_ARGS.MAX_EPOCHS,
                                              gpus=self.WAV2VEC2_ARGS.NUM_GPUS
                                              )
        self.datamodule = SpeechRecognitionData.from_csv("file_name",
                                                         "transcription",
                                                         train_file=self.WAV2VEC2_ARGS.TRAIN_FILE_PATH,
                                                         test_file=self.WAV2VEC2_ARGS.TEST_FILE_PATH,
                                                         batch_size=self.WAV2VEC2_ARGS.BATCH_SIZE
                                                         )

    @property
    def trainer(self) -> flash.Trainer:
        return self.wav2vec2_trainer

    @trainer.setter
    def trainer(self,
                accumulate_grad_batches: int,
                precision: int,
                max_epochs: int,
                gpus: int):
        self.wav2vec2_trainer = flash.Trainer(accumulate_grad_batches=accumulate_grad_batches,
                                              precision=precision,
                                              max_epochs=max_epochs,
                                              gpus=gpus
                                              )

    @property
    def model(self) -> SpeechRecognition:
        return self.wav2vec2_model

    @model.setter
    def model(self, model_path:str) -> None:
        self.wav2vec2_model = SpeechRecognition.load_from_checkpoint(model_path)

    def shape(self) -> Dict:
        return {"training": pd.read_csv(self.WAV2VEC2_ARGS.TRAIN_FILE_PATH).shape,
                "validation": pd.read_csv(self.WAV2VEC2_ARGS.TEST_FILE_PATH).shape}

    def finetune(self):
        self.wav2vec2_trainer.finetune(self.wav2vec2_model,
                                       datamodule=self.datamodule,
                                       strategy=self.WAV2VEC2_ARGS.FINETUNE_STRATEGY)
        self.wav2vec2_trainer.save_checkpoint(self.WAV2VEC2_ARGS.MODEL_SAVE_PATH)
    @staticmethod
    def inference(inference_files: List[str], batch_size: int, model_path: str, wav2vec2_trainer):
        model = SpeechRecognition.load_from_checkpoint(model_path)
        inference_data = SpeechRecognitionData.from_files(
            predict_files=inference_files,
            batch_size=batch_size)
        wav2vec2_predictions = wav2vec2_trainer.predict(model, datamodule=inference_data)
        return wav2vec2_predictions