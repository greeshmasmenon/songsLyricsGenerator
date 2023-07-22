from abc import ABC
from constants.mir_constants import training_args
from dataclasses import dataclass

class SpeechModel(ABC):
    def __init__(self, args: training_args) -> None:
        self.training_args = args

    def load_data(self):
        raise NotImplementedError        
    
    def finetune(self):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
