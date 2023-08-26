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


wandb_logger = WandbLogger(project="SLG - Whisper transfer learning",log_model=False,)

print(json.dumps(asdict(WAV2VEC2_ARGS), indent = 4))


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    feature_extractor: AutoFeatureExtractor
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
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
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(hparams.whisper_model)
        #self.feature_extractor = AutoFeatureExtractor(do_normalize=True, return_attention_mask=True)
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer, padding=True)
    
    def setup(self, stage=None):
        train_df = pd.read_csv(WAV2VEC2_ARGS.TRAIN_FILE_PATH)
        validation_df = pd.read_csv(WAV2VEC2_ARGS.TEST_FILE_PATH)
        if stage == 'fit' or stage is None:
            print("In Stage = Fit")
            train_dataset = Dataset.from_dict(
                    {"audio": list(train_df["consolidated_file_path"]),
                    "transcription": list(train_df["transcription"])}).cast_column("audio", Audio(sampling_rate=16_000))
            self.train_dataset = train_dataset.map(self.prepare_dataset,remove_columns = train_dataset.column_names)
            
            val_dataset = Dataset.from_dict(
                    {"audio": list(validation_df["consolidated_file_path"]),
                    "transcription": list(validation_df["transcription"])}).cast_column("audio", Audio(sampling_rate=16_000))
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
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(batch["input_features"])
        batch["labels"] = self.tokenizer(batch["transcription"]).input_ids
        batch["label_attention_mask"] = self.tokenizer(batch["transcription"]).attention_mask
        return batch


class Wav2SeqModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.batch_size = hparams.batch_size
        self.learning_rate=hparams.learning_rate
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.lm_model)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(hparams.whisper_model).model.encoder
        #self.seq2seq = AutoModelForSeq2SeqLM.from_pretrained(hparams.lm_model)
        self.seq2seq = BartForConditionalGeneration.from_pretrained(hparams.lm_model, forced_bos_token_id=0)

        #self.seq2seq = AutoModelForSeq2SeqLM.from_pretrained(hparams.lm_model)
        #self.seq2seq = BartForConditionalGeneration.from_pretrained(hparams.lm_model,
        #                                            forced_bos_token_id=0) #https://github.com/huggingface/transformers/issues/15559
        self.seq2seq.config.is_decoder = True
        self.seq2seq.add_cross_attention = True
        self.bridging_layer = nn.Linear(self.whisper.config.hidden_size, self.seq2seq.config.hidden_size)
        self.predicted_ids = None
        self.wer=WordErrorRate()


    def forward(self, audio, labels, label_attention_mask):
        #print("entering forward step")
        encoder_outputs = self.whisper(audio,
                                        output_hidden_states=True,
                                        output_attentions=True)
        encoder_hidden_states = encoder_outputs[0]  
        encoder_hidden_states = self.bridging_layer(encoder_hidden_states)
        decoder_input_ids = self.shift_tokens_right(labels,pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_id = self.tokenizer.bos_token_id) 
        decoder_attention_masks = self.shift_tokens_right_mask(label_attention_mask)
        
        #print(f"labels={labels},label_attention_mask = {label_attention_mask}, 
        # decoder_input_ids={decoder_input_ids},
        # decoder_attention_masks={decoder_attention_masks}")

        decoder_outputs = self.seq2seq(input_ids=decoder_input_ids,
                                       #decoder_attention_masks=decoder_attention_masks,
                                       encoder_hidden_states=encoder_hidden_states,)
        return decoder_outputs

    def training_step(self, batch, batch_idx):
        #print("entering training step")
        self.whisper.eval()
        self.seq2seq.train()
        audio = batch["input_features"]
        label_attention_mask = batch["label_attention_mask"]
        labels = batch["labels"]
        input_lengths = labels.shape[1]
        #print(f"input_lengths = {input_lengths}")
        loss_acc = 0
        label_tensor=torch.zeros(1,1)
        #Implementing Teacher Forcing Algorithm
        for length in range(1,input_lengths+1):
            teacher_force_flag= random.random() < 0.75
            if length ==1:
                teacher_force_flag=False
                label_tensor = labels[:,:length]
            else:
                if teacher_force_flag :
                    label_tensor = torch.cat(
                        (label_tensor,
                         labels[:,length-1].unsqueeze(0)
                         ), dim =-1)
                else:
                    label_tensor = torch.cat(
                        (self.predicted_ids.unsqueeze(0), 
                         labels[:,length-1].unsqueeze(0)
                         ), dim =-1)

            label_attention_mask_unit = label_attention_mask[:,:length]
            
            #print(f"teacher_force_flag={teacher_force_flag},label_tensor={label_tensor},predicted_ids={self.predicted_ids},newly_added_label = {labels[:,length-1]}")
                # teacher_force_flag=True,
                # label_tensor=tensor([[ 1005,  1041,  1998,  1999,  1998, 10222,  1996,  2466,  2003,  2204,102]], device='cuda:0'),
                # predicted_ids=tensor([2000, 1998, 2000, 1998, 1999, 1998, 1998, 1996, 1996, 2003],device='cuda:0'),
                # newly_added_label = tensor([102], device='cuda:0')

            logits = self(audio,label_tensor,label_attention_mask_unit).logits
            ce_loss = nn.CrossEntropyLoss(ignore_index=0)
            loss = ce_loss(logits.squeeze(),label_tensor.squeeze())
            self.predicted_ids = torch.argmax(logits, dim=-1)[0]
            loss_acc+=loss
        self.log("train_loss",loss_acc/input_lengths, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_acc/input_lengths

    def validation_step(self, batch,batch_idx):
        #print("validation_step")
        audio = batch["input_features"]
        labels = batch["labels"]
        label_attention_mask = batch["label_attention_mask"]
        logits = self(audio,labels,label_attention_mask).logits
        ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        loss = ce_loss(logits.squeeze(),labels.squeeze())
        predicted_ids = torch.argmax(logits, dim=-1)
        original_text = self.tokenizer.decode(labels[0],skip_special_tokens=False)
        predicted_text = self.tokenizer.decode(predicted_ids[0],skip_special_tokens=False)
        print(f"original text = {original_text}, labels = {labels[0]}")
        print(f"Predicted text = {predicted_text}, predicted ids = {predicted_ids[0]}")
        #self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_wer',self.wer(predicted_text,original_text),on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss


    def configure_optimizers(self):
        print("Entering Optimization Step")
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

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
    trainer = Trainer(max_epochs=hparams.max_epochs,devices=1, accelerator="gpu", logger=wandb_logger, enable_progress_bar=True)
    trainer.fit(model,SpeechRecognitionDataModule(WAV2VEC2_ARGS,num_workers=4,hparams=hparams))
    return model, trainer


hparams = argparse.Namespace()
hparams.wav2vec2_model = 'facebook/wav2vec2-large-960h-lv60-self'
hparams.whisper_model = 'openai/whisper-large-v2'#'openai/whisper-large'
hparams.lm_model = 'bert-base-uncased' #'bert-base-uncased' #
hparams.vocab_size = 20_000
hparams.learning_rate = 1e-5
hparams.max_epochs = 3
hparams.batch_size = 1

model,trainer = run(hparams=hparams)