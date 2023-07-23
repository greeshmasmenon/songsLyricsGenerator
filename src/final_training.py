from constants.mir_constants import WAV2VEC2_ARGS
from training.wav2vec2_finetune import Wav2Vec2SpeechRecognition
import json

wav2vec2 = Wav2Vec2SpeechRecognition(wav2vec2_args=WAV2VEC2_ARGS)
shape_dict = wav2vec2.shape()
print(json.dumps(shape_dict, indent = 4))

wav2vec2.finetune()


inference_files_lst = ["/home/users/gmenon/workspace/songsLyricsGenerator/src/notebooks/separated/mdx_extra/ad887cbfa84749e5a9789f303e2f5c30/vocals.wav",
                  "/home/users/gmenon/dali/DALI_v1.0/audio/wav_clips/ad887cbfa84749e5a9789f303e2f5c30.wav"]

inference_predictions = wav2vec2.inference(inference_files=inference_files_lst,
                   batch_size=4,
                   model_path="/home/users/gmenon/workspace/songsLyricsGenerator/model_artefacts/wav2vec2_finetuned_model.pt",
                  wav2vec2_trainer = wav2vec2.trainer)


print(inference_predictions)

