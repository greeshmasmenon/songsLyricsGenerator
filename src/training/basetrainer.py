from flash.audio import SpeechRecognition
from constants.mir_constants import TrainingArgs
from dataclasses import dataclass
from abc import ABC


class SpeechModel(ABC):
    def __init__(self, args: TrainingArgs) -> None:
        self.training_args = args

    def load_data(self):
        raise NotImplementedError        
    
    def finetune(self):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
