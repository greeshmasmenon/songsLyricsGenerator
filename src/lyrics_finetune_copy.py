import os
os.chdir('/home/users/gmenon/workspace/songsLyricsGenerator/src')

import re
import torch
import json
import pandas as pd
import argparse
import numpy as np
import random
from torch import nn
from torchmetrics.text import WordErrorRate
from typing import Optional
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from constants.mir_constants import TrainingArgs, WAV2VEC2_ARGS
from dataclasses import dataclass, asdict, field # type: ignore
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperTokenizer, WhisperFeatureExtractor, Wav2Vec2Processor, BertTokenizer,WhisperForConditionalGeneration,BartForConditionalGeneration
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForCTC, AutoModelForSeq2SeqLM,AutoFeatureExtractor
from datasets import load_dataset, Dataset, Audio
from typing import Any, Dict, List, Optional, Union
from flash.audio import SpeechRecognition, SpeechRecognitionData
from training.wav2vec2_finetune import Wav2Vec2SpeechRecognition, SpeechRecognitionData
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


wandb_logger = WandbLogger(project="SLT Encoder Decoder",log_model=False,)

print(json.dumps(asdict(WAV2VEC2_ARGS), indent = 4))


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    feature_extractor: AutoFeatureExtractor
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")
        label_attention_features =[{"input_ids": feature["label_attention_mask"]} for feature in features]

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways

        label_attention_batch = self.tokenizer.pad(
            label_attention_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_attention = label_attention_batch["input_ids"]
        #.masked_fill(labels_batch.input_ids.eq(101), 0).masked_fill(labels_batch.input_ids.eq(102), 0)
        labels = labels[:, 1:]
        labels_attention = labels_attention[:,1:]

        batch["labels"] = labels
        batch["label_attention_mask"]  = labels_attention

        return batch


class SpeechRecognitionDataModule(LightningDataModule):
    def __init__(self, WAV2VEC2_ARGS: WAV2VEC2_ARGS, num_workers,hparams):
        super().__init__()
        self.batch_size = WAV2VEC2_ARGS.BATCH_SIZE
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.lm_model)
        #self.feature_extractor = WhisperFeatureExtractor.from_pretrained(hparams.whisper_model)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(hparams.wav2vec2_model,do_normalize=True, return_attention_mask=True)
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer, padding=True)
    
    def setup(self, stage=None):
        train_df = pd.read_csv(WAV2VEC2_ARGS.TRAIN_FILE_PATH)
        validation_df = pd.read_csv(WAV2VEC2_ARGS.TEST_FILE_PATH)
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

        
        if stage == 'test' or stage is None:
            print("In Stage = Test")
            test_dataset = Dataset.from_dict(
                    {"audio": list(validation_df["consolidated_file_path"]),
                    "transcription": list(validation_df["transcription"])}).cast_column("audio", Audio(sampling_rate=16_000))
            test_dataset = val_dataset.map(self.prepare_dataset,remove_columns = val_dataset.column_names)
            self.test_dataset = test_dataset
    
    def train_dataloader(self):
        print("entering train data loader")
        return DataLoader(
            self.train_dataset.with_format("torch"), 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            collate_fn = self.data_collator
        )
    
    def val_dataloader(self):
        print("entering val data loader")
        return DataLoader(
            self.val_dataset.with_format("torch"), 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            collate_fn = self.data_collator
        )
    
    def test_dataloader(self):
        print("entering test data loader")
        return DataLoader(
            self.test_dataset.with_format("torch"), 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            collate_fn = self.data_collator
        )

    def prepare_dataset(self,batch):
        audio = batch["audio"]
        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = self.tokenizer(batch["transcription"]).input_ids
        batch["label_attention_mask"] = self.tokenizer(batch["transcription"]).attention_mask
        return batch


class Wav2SeqModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.batch_size = hparams.batch_size
        self.learning_rate=hparams.learning_rate
        self.unfreeze_after_epoch = hparams.unfreeze_after_epoch
        self.unfreeze_after_epoch_more = hparams.unfreeze_after_epoch_more
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.lm_model)
        #self.whisper = WhisperForConditionalGeneration.from_pretrained(hparams.whisper_model).model.encoder
        #self.wav2vec2 = AutoModelForCTC.from_pretrained(hparams.wav2vec2_model)
        self.wav2vec2 = SpeechRecognition.load_from_checkpoint("/scratch/users/gmenon/model_artefacts/wav2vec2_demucs_en_large-960h-lv60-self_freeze_unfreeze_15epochs_adam.pt").model
        self.seq2seq = AutoModelForCausalLM.from_pretrained(hparams.lm_model)
        self.seq2seq.config.is_decoder = True
        self.seq2seq.add_cross_attention = True
        self.seq2seq.hidden_dropout_prob = 0.3
        self.seq2seq.attention_probs_dropout_prob = 0.25
        #self.bridging_layer = nn.Linear(self.wav2vec2.config.hidden_size, self.seq2seq.config.hidden_size)
        self.predicted_ids = None
        self.wav2vec2.eval()
        self.seq2seq.train()
        self.wer=WordErrorRate()
        self.predicted_ids=None


    def forward(self, audio, labels, label_attention_mask,teacher_force_ratio=0.75):
        #print("entering forward step")
        #Create necessary variables for the forward step
        input_lengths = labels.shape[1]
        label_tensor=torch.zeros(1,1)

        #Encoding
        encoder_outputs = self.wav2vec2(audio,
                                        output_hidden_states=True,
                                        output_attentions=True)
        encoder_hidden_states = encoder_outputs[0]  
        encoder_attention_mask = self.wav2vec2._get_feature_vector_attention_mask(encoder_hidden_states.shape[1], label_attention_mask)
        #encoder_hidden_states = self.bridging_layer(encoder_hidden_states)

        # Decoding in a Seq2Seq fashion
        for length in range(1,input_lengths+1):
            teacher_force_flag= random.random() < teacher_force_ratio
            if length ==1:
                teacher_force_flag=False
                label_tensor = labels[:,:length]
            else:
                if teacher_force_flag :
                    label_tensor = torch.cat(
                        (label_tensor,
                         labels[:,length-1].unsqueeze(0))
                         , dim =-1)
                else:
                    label_tensor = torch.cat(
                        (label_tensor, 
                         self.predicted_ids.unsqueeze(0))
                         , dim =-1)
            label_attention_mask_unit = label_attention_mask[:,:length]
            decoder_attention_masks = self.shift_tokens_right_mask(label_attention_mask_unit)
            decoder_input_ids = self.shift_tokens_right(input_ids=label_tensor, 
                                                        pad_token_id=0, # type: ignore
                                                        decoder_start_token_id=101)  # type: ignore
            decoder_outputs = self.seq2seq(input_ids=decoder_input_ids, # type: ignore
                                           #attention_mask=decoder_attention_masks,
                                           encoder_attention_mask=encoder_attention_mask,
                                           encoder_hidden_states=encoder_hidden_states)
            self.predicted_ids = torch.argmax(decoder_outputs.logits, dim=-1)[:,length-1]
            #print(f"teacher_force_flag={teacher_force_flag},label_tensor={label_tensor},decoder_input_ids={decoder_input_ids},predicted_ids={self.predicted_ids}")
        return decoder_outputs # type: ignore

    def training_step(self, batch, batch_idx):
        #print("entering training step")
        if self.current_epoch== self.unfreeze_after_epoch:
            self.seq2seq.train()
        if self.current_epoch==self.unfreeze_after_epoch_more:
            self.wav2vec2.train()
        audio = batch["input_values"]
        label_attention_mask = batch["label_attention_mask"]
        labels = batch["labels"]
        logits = self(audio,labels,label_attention_mask).logits
        ce_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = ce_loss(logits.squeeze(),labels.squeeze())
        predicted_ids = torch.argmax(logits, dim=-1)
        original_text = self.tokenizer.decode(labels[0],skip_special_tokens=False)
        predicted_text = self.tokenizer.decode(predicted_ids[0],skip_special_tokens=False)
        print(f"Predicted text = {predicted_text}, original_text = {original_text}")
        self.log("train_loss",loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_wer',self.wer(predicted_text,original_text),on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch,batch_idx):
        #print("validation_step")
        audio = batch["input_values"]
        labels = batch["labels"]
        label_attention_mask = batch["label_attention_mask"]
        logits = self(audio,labels,label_attention_mask).logits
        ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        loss = ce_loss(logits.squeeze(),labels.squeeze())
        predicted_ids = torch.argmax(logits, dim=-1)
        original_text = self.tokenizer.decode(labels[0],skip_special_tokens=False)
        predicted_text = self.tokenizer.decode(predicted_ids[0],skip_special_tokens=False)
        #print(f"original text = {original_text}, labels = {labels[0]}")
        #print(f"Predicted text = {predicted_text}, predicted ids = {predicted_ids[0]}")
        #self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_wer',self.wer(predicted_text,original_text),on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss


    def configure_optimizers(self):
        print("Entering Optimization Step")
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int=0, decoder_start_token_id: int=101):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        if decoder_start_token_id is None:
            raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
        shifted_input_ids[:, 0] = decoder_start_token_id
    
        if pad_token_id is None:
            raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    @staticmethod
    def shift_tokens_right_mask(input_ids: torch.Tensor):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = 0
        return shifted_input_ids


def run(hparams):
    print(hparams)
    model = Wav2SeqModel(hparams)
    trainer = Trainer(max_epochs=hparams.max_epochs,devices=1, accelerator="gpu", logger=wandb_logger, enable_progress_bar=True,auto_lr_find=True)
    trainer.fit(model,SpeechRecognitionDataModule(WAV2VEC2_ARGS,num_workers=4,hparams=hparams))
    return model, trainer


hparams = argparse.Namespace()
hparams.wav2vec2_model = 'facebook/wav2vec2-large-960h-lv60-self'
hparams.whisper_model = 'openai/whisper-large-v2'#'openai/whisper-large'
#hparams.lm_model = 'bert-base-uncased' #'bert-base-uncased' #
hparams.lm_model = 'bert-base-uncased'
hparams.vocab_size = 40_000
hparams.learning_rate = 1e-5
hparams.max_epochs = 30
hparams.batch_size = 1
hparams.unfreeze_after_epoch = 1
hparams.unfreeze_after_epoch_more = 5


model,trainer = run(hparams=hparams)