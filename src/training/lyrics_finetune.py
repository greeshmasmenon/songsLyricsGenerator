import os
import re
import torch
import json
import pandas as pd
import argparse
from torch import nn
from torchmetrics.text import WordErrorRate
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from constants.mir_constants import TrainingArgs, WAV2VEC2_ARGS
from dataclasses import dataclass, asdict
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForCTC, AutoModelForSeq2SeqLM
from datasets import load_dataset,Dataset,Audio


wandb_logger = WandbLogger(project="SLG - wav2vec2 transfer learning",log_model=True,)

print(json.dumps(asdict(WAV2VEC2_ARGS), indent = 4))


class SpeechRecognitionDataModule(LightningDataModule):
    def __init__(self, WAV2VEC2_ARGS: TrainingArgs, num_workers: int=None):
        super().__init__()
        self.batch_size = WAV2VEC2_ARGS.BATCH_SIZE
        self.num_workers = num_workers if num_workers is not None else os.getenv('NUMBER_OF_CPUS')
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(WAV2VEC2_ARGS.MODEL_BACKBONE)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_ARGS.MODEL_BACKBONE)
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.auto_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.cpu_count = 4
        print(f"CPU Count = {self.cpu_count}")
    
    def setup(self, stage=None):
        train_df = pd.read_csv(WAV2VEC2_ARGS.TRAIN_FILE_PATH)
        validation_df = pd.read_csv(WAV2VEC2_ARGS.TEST_FILE_PATH)
        #songs_metadata = pd.concat([train_df,validation_df], ignore_index = True)
        if stage == 'fit' or stage is None:
            print("In Stage = Fit")
            train_dataset = Dataset.from_dict(
                    {"audio": list(train_df["consolidated_file_path"]),
                    "transcription": list(train_df["transcription_capitalized"])}).cast_column("audio", Audio(sampling_rate=16_000))
            self.train_dataset = train_dataset.map(self.prepare_dataset,remove_columns = train_dataset.column_names)
            
            val_dataset = Dataset.from_dict(
                    {"audio": list(validation_df["consolidated_file_path"]),
                    "transcription": list(validation_df["transcription_capitalized"])}).cast_column("audio", Audio(sampling_rate=16_000))
            self.val_dataset = val_dataset.map(self.prepare_dataset,remove_columns = val_dataset.column_names)
            #common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)

        if stage == 'test' or stage is None:
            print("In Stage = Test")
            val_dataset = Dataset.from_dict(
                    {"audio": list(validation_df["consolidated_file_path"]),
                    "transcription": list(validation_df["transcription_capitalized"])}).cast_column("audio", Audio(sampling_rate=16_000))
            self.test_dataset = val_dataset
    
    def train_dataloader(self):
        print("entering train data loader")
        return DataLoader(
            self.train_dataset.with_format("torch"), 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        print("entering val data loader")
        return DataLoader(
            self.val_dataset.with_format("torch"), 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        print("entering test data loader")
        return DataLoader(
            self.test_dataset.with_format("torch"), 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )
        
    def prepare_dataset(self, batch):
        audio = batch["audio"]
        transcription = batch["transcription"]
        batch["input_values"] = audio["array"]
        batch["labels"] = self.auto_tokenizer(transcription, padding=True, truncation=True).input_ids
        # batch["input_length"] = len(batch["input_values"])
        # batch["target_length"] = len(batch["labels"])
        return batch


class Wav2SeqModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = hparams.batch_size
        self.learning_rate = hparams.learning_rate
        self.wav2vec2 = AutoModelForCTC.from_pretrained(hparams.wav2vec2_model,ctc_zero_infinity=True,ctc_loss_reduction="mean")
        self.wav2vec2.freeze_feature_encoder()
        self.seq2seq = AutoModelForCausalLM.from_pretrained(hparams.lm_model)
        self.enc_to_dec_proj = nn.Linear(self.wav2vec2.config.hidden_size, self.seq2seq.config.hidden_size) #TODO: Can be improved.
        print(self.seq2seq.config)

    def forward(self, audio):
        x = self.wav2vec2(audio[0],output_hidden_states=True,output_attentions=True)
        predicted_ids = torch.argmax(x.logits, dim=-1)
        predicted_ids = predicted_ids[:,:512]
        print(predicted_ids.shape)
        logits = self.seq2seq(input_ids = predicted_ids).logits
        return logits

    def training_step(self, batch, batch_idx):
        #print("entering training step")
        audio = batch["input_values"].unsqueeze(0)
        labels = batch["labels"]
        labels = labels.reshape(-1)
        logits = self(audio)
        logits = logits.reshape(-1, self.seq2seq.config.vocab_size)
        input_lengths = torch.full(size=(self.batch_size,), fill_value=logits.shape[0], dtype=torch.long)
        target_lengths = torch.full(size=(self.batch_size,), fill_value=labels.shape[0], dtype=torch.long)
        ctc_loss =  nn.CTCLoss(blank=0)
        loss = ctc_loss(logits,labels,input_lengths,target_lengths)
        wer = WordErrorRate()
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch,batch_idx):
        #print("entering validation step")
        audio = batch["input_values"].unsqueeze(0)
        labels = batch["labels"]
        labels = labels.reshape(-1)
        logits = self(audio)
        logits = logits.reshape(-1, self.seq2seq.config.vocab_size)
        input_lengths = torch.full(size=(self.batch_size,), fill_value=logits.shape[0], dtype=torch.long)
        target_lengths = torch.full(size=(self.batch_size,), fill_value=labels.shape[0], dtype=torch.long)
        print(logits.shape,labels.shape,input_lengths.shape,target_lengths.shape)
        #loss = nn.CTCLoss(blank=0).forward(logits, labels,input_length,target_length)
        ctc_loss =  nn.CTCLoss(blank=0)
        loss = ctc_loss(logits,labels,input_lengths,target_lengths)
        #loss_fct = nn.CrossEntropyLoss()
        #loss = loss_fct(logits.reshape(-1, self.seq2seq.config.vocab_size), labels.reshape(-1))
        self.log('val_loss', loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    

def run(hparams):
    print(hparams)
    model = Wav2SeqModel(hparams)
    trainer = Trainer(max_epochs=4,devices=1, accelerator="gpu", logger=wandb_logger)
    trainer.fit(model,SpeechRecognitionDataModule(WAV2VEC2_ARGS,num_workers=4))
    return 'completed'

    # {'audio': [{'path': None, 'array': array([-0.05114746, -0.11273193, -0.09152222, ...,  0.2796936 ,
    #         0.29998779,  0.20550537]), 'sampling_rate': 16000}, {'path': None, 'array': array([ 0.14523315,  0.05508423, -0.13534546, ...,  0.01959229,
    #         0.00973511, -0.00811768]), 'sampling_rate': 16000}], 'transcription': ["just a tryin' to survive", "don't go on me"], 'input_values':