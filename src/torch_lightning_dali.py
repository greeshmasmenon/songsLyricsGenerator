import flash
import torch
from flash.audio import SpeechRecognition, SpeechRecognitionData
from flash.core.data.utils import download_data

# 1. Create the DataModule
# download_data("https://pl-flash-data.s3.amazonaws.com/timit_data.zip", "./data")


datamodule = SpeechRecognitionData.from_csv(
    "file_name",
    "transcription",
    train_file = "/home/users/gmenon/notebooks/home/users/gmenon/notebooks/train_metadata_cleaned.csv",
    test_file = "/home/users/gmenon/notebooks/home/users/gmenon/notebooks/validation_metadata_cleaned.csv",
    batch_size = 4

)

# datamodule = SpeechRecognitionData.from_json(
#     "file",
#     "text",
#     train_file="data/timit/train.json",
#     test_file="data/timit/test.json",
#     batch_size=4,
# )


# 2. Build the task
#model = SpeechRecognition(backbone="facebook/wav2vec2-base-960h")
model = SpeechRecognition(backbone="facebook/wav2vec2-large-960h-lv60-self")

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=20, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Predict on audio files!
datamodule = SpeechRecognitionData.from_files(predict_files=["/home/users/gmenon/dali/DALI_v1.0/audio/wav_clips/48f4a0fa25b84e01ac6ff451d493bf74.wav"], batch_size=4)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("model_artefacts/speech_recognition_model.pt")