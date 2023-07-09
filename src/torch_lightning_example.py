import flash
import torch
from flash.audio import SpeechRecognition, SpeechRecognitionData
from flash.core.data.utils import download_data

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/timit_data.zip", "./data")

datamodule = SpeechRecognitionData.from_json(
    "file",
    "text",
    train_file="data/timit/train.json",
    test_file="data/timit/test.json",
    batch_size=4,
)



# 2. Build the task
model = SpeechRecognition(backbone="facebook/wav2vec2-base-960h")

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Predict on audio files!
datamodule = SpeechRecognitionData.from_files(predict_files=["data/timit/example.wav"], batch_size=4)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("model_artefacts/speech_recognition_model.pt")