import os
os.chdir('/home/users/gmenon/workspace/songsLyricsGenerator/src')

from constants.mir_constants import WAV2VEC2_ARGS
from training.wav2vec2_finetune import Wav2Vec2SpeechRecognition, SpeechRecognitionData
import json
from dataclasses import dataclass, asdict
import pandas as pd
from jiwer import wer



#print hyperparameters used for training this model into wandb
print(json.dumps(asdict(WAV2VEC2_ARGS), indent = 4))

speech_recognition_task = Wav2Vec2SpeechRecognition(wav2vec2_args=WAV2VEC2_ARGS)

#print shape of data in console. Will be visible in wandb logger
print(json.dumps(speech_recognition_task.shape(), indent = 4))

speech_recognition_task.finetune()

inference_files_lst = ["/home/users/gmenon/workspace/songsLyricsGenerator/src/notebooks/separated/mdx_extra/ad887cbfa84749e5a9789f303e2f5c30/vocals.wav",
                  "/home/users/gmenon/dali/DALI_v1.0/audio/wav_clips/ad887cbfa84749e5a9789f303e2f5c30.wav"]

inference_predictions = speech_recognition_task.inference(inference_files=inference_files_lst,
                   batch_size=4,
                   model_path=WAV2VEC2_ARGS.MODEL_SAVE_PATH,
                  wav2vec2_trainer = speech_recognition_task.wav2vec2_trainer)


print(f"Inference Predictions = {inference_predictions}")

print("Starting Validation dataset Word Error Rate Metric Calculation")

test_data = pd.read_csv(WAV2VEC2_ARGS.TEST_FILE_PATH)

test_datamodule = SpeechRecognitionData.from_files(
    predict_files=list(test_data["consolidated_file_path"]), 
    batch_size=1)

finetuned_predictions = speech_recognition_task.wav2vec2_trainer.predict(speech_recognition_task.wav2vec2_model, 
                                                datamodule=test_datamodule)

test_data["finetuned_predictions"] = finetuned_predictions
print(test_data.head(10))
finetuned_pred_transformed = []
for predictions in finetuned_predictions:
    finetuned_pred_transformed.append(predictions[0])
reference = list(test_data["transcription_capitalized"])
hypothesis = finetuned_pred_transformed
error = wer(reference, hypothesis)
print(f"Word Error Rate = {error}")

