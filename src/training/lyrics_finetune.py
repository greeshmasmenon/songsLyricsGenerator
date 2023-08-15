import os
import re
import torch
import json
import pandas as pd
import argparse
from torch import nn
from torchmetrics.text import WordErrorRate
from typing import Optional
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from constants.mir_constants import TrainingArgs, WAV2VEC2_ARGS
from dataclasses import dataclass, asdict # type: ignore
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForCTC, AutoModelForSeq2SeqLM
from datasets import load_dataset,Dataset,Audio


wandb_logger = WandbLogger(project="SLG - wav2vec2 transfer learning",log_model=True,)

print(json.dumps(asdict(WAV2VEC2_ARGS), indent = 4))

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor,BertTokenizer
from datasets import load_dataset,Dataset,Audio
import pandas as pd
import re
import os

class SpeechRecognitionDataModule(pl.LightningDataModule):
    def __init__(self, WAV2VEC2_ARGS: WAV2VEC2_ARGS, num_workers):
        super().__init__()
        self.batch_size = WAV2VEC2_ARGS.BATCH_SIZE
        self.num_workers = num_workers
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(WAV2VEC2_ARGS.MODEL_BACKBONE)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_ARGS.MODEL_BACKBONE)
        #self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.bert_tokenizer)
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)
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
            test_dataset = Dataset.from_dict(
                    {"audio": list(validation_df["consolidated_file_path"]),
                    "transcription": list(validation_df["transcription_capitalized"])}).cast_column("audio", Audio(sampling_rate=16_000))
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
        
    # def prepare_dataset(self, batch):
    #     audio = batch["audio"]
    #     transcription = batch["transcription"]
    #     #batch["input_values"] = audio["array"]
    #     batch["input_values"] = self.feature_extractor(audio["array"],sampling_rate=16_000)
    #     #batch["labels"] = self.bert_tokenizer(transcription, padding=True, truncation=True).input_ids
    #     batch["labels"]=transcription
    #     #batch["audios"] = audio["array"]
    #     # batch["input_length"] = len(batch["input_values"])
    #     # batch["target_length"] = len(batch["labels"])
    #     return batch
    # #tokenizer(batch["target_text"], **additional_kwargs).input_ids

    def prepare_dataset(self,batch):
        audio = batch["audio"]
        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["transcription"]).input_ids
        return batch

import torch
from torch import nn
from pytorch_lightning import LightningModule, Trainer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC

class Wav2SeqModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.batch_size = hparams.batch_size
        self.learning_rate = hparams.learning_rate
        self.save_hyperparameters()
        # self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(hparams.wav2vec2_model)
        # self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hparams.wav2vec2_model)
        # self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.wav2vec2 = AutoModelForCTC.from_pretrained(hparams.wav2vec2_model,ctc_zero_infinity=True,ctc_loss_reduction="mean")
        self.wav2vec2.freeze_feature_encoder()
        print(self.wav2vec2.config)
        self.seq2seq = AutoModelForCausalLM.from_pretrained(hparams.lm_model)
        self.seq2seq.config.is_decoder = True
        self.enc_to_dec_proj = nn.Linear(self.wav2vec2.config.hidden_size, self.seq2seq.config.hidden_size)
        print(self.seq2seq.config)
        #self.data_collator = DataCollatorCTCWithPadding()


    def forward(self, audio):
        #print("entering forward step")
        encoder_outputs = self.wav2vec2(audio[0],output_hidden_states=True,output_attentions=True)
        # print(f"encoder_outputs = {encoder_outputs[0].shape},{encoder_outputs[0]},logits = {encoder_outputs.logits},hidden_states = {encoder_outputs.hidden_states}")
        # encoder_attention_mask = self.wav2vec2._get_feature_vector_attention_mask(
        #         encoder_outputs[0].shape[1], encoder_outputs.attentions)
        #print(f"encoder_attention_mask = {encoder_outputs.attentions}")
        predicted_ids = torch.argmax(encoder_outputs.logits, dim=-1)
        predicted_ids = predicted_ids[:,:512]
        # print(predicted_ids.shape)
        encoder_hidden_states = encoder_outputs[0]
        # print(encoder_hidden_states.shape,type(encoder_hidden_states),encoder_hidden_states)
        # print(f"pad_token_id = {self.seq2seq.config.pad_token_id}")
        
        #x_hidden_states = self.enc_to_dec_proj(x.hidden_states[0])
        #print(f"hidden states = {x.hidden_states[0].shape},{x.hidden_states[0]}")
        # print(f"attentions = {len(x.attentions[0])}, {x.attentions}")
        # print(f"logits = {len(x.logits[0])},{x.logits}")
        #decoder_input_ids = self.shift_tokens_right(encoder_hidden_states, 0, 0)
        decoder_outputs = self.seq2seq(input_ids=predicted_ids,encoder_hidden_states=encoder_hidden_states) #,attention_mask = x.attentions, decoder_input_ids=predicted_ids

            #     decoder_outputs = self.decoder(
            # input_ids=decoder_input_ids,
            # attention_mask=decoder_attention_mask,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_attention_mask,
            # inputs_embeds=decoder_inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # use_cache=use_cache,
            # past_key_values=past_key_values,
            # return_dict=return_dict,
        return decoder_outputs

    def training_step(self, batch, batch_idx):
        #print("entering training step")
        # audio = batch["input_values"].unsqueeze(0)
        # labels = batch["labels"]
        # labels = labels.reshape(-1)
        # # input_length = batch["input_length"]
        # # target_length = batch["input_length"]
        # #audio,labels,attention_mask = batch
        # # audio = batch["input_values"]
        # # labels = batch["labels"]
        # # print(audio)
        # # print(labels)
        # logits = self(audio)
        # logits = logits.reshape(-1, self.seq2seq.config.vocab_size)
        # input_lengths = torch.full(size=(self.batch_size,), fill_value=logits.shape[0], dtype=torch.long)
        # target_lengths = torch.full(size=(self.batch_size,), fill_value=labels.shape[0], dtype=torch.long)
        # ctc_loss =  nn.CTCLoss(blank=0)
        # loss = ctc_loss(logits,labels,input_lengths,target_lengths)
        # self.log('train_loss', loss, on_step=True, on_epoch=True)
        audio = batch["input_values"].unsqueeze(0)
        labels = batch["labels"]
        labels = labels.reshape(self.batch_size,-1)
        # input_length = batch["input_length"]
        # target_length = batch["input_length"]
        #audio,labels,attention_mask = batch
        # audio = batch["input_values"]
        # labels = batch["labels"]
        # print(audio)
        # print(labels)
        logits = self(audio).logits
        logits = logits.reshape(-1, self.batch_size, self.seq2seq.config.vocab_size)
        input_lengths = torch.full(size=(self.batch_size,), fill_value=logits.shape[0], dtype=torch.long)
        target_lengths = torch.full(size=(self.batch_size,), fill_value=labels.shape[0], dtype=torch.long)
        #print(logits.shape,labels.shape,input_lengths.shape,target_lengths.shape)
        #print(f"input_lengths = {input_lengths}")
        #print(f"target_lengths = {target_lengths}")
        #loss = nn.CTCLoss(blank=0).forward(logits, labels,input_length,target_length)
        ctc_loss =  nn.CTCLoss(blank=0)
        loss = ctc_loss(logits,labels,input_lengths,target_lengths)
        #loss_fct = nn.CrossEntropyLoss()
        #loss = loss_fct(logits.reshape(-1, self.seq2seq.config.vocab_size), labels.reshape(-1))
        return loss

    def validation_step(self, batch,batch_idx):
        #print("entering validation step")
       # print(batch)
        audio = batch["input_values"].unsqueeze(0)
        labels = batch["labels"]
        labels = labels.reshape(self.batch_size,-1)
        # input_length = batch["input_length"]
        # target_length = batch["input_length"]
        #audio,labels,attention_mask = batch
        # audio = batch["input_values"]
        # labels = batch["labels"]
        # print(audio)
        # print(labels)
        logits = self(audio).logits
        logits = logits.reshape(-1, self.batch_size, self.seq2seq.config.vocab_size)
        input_lengths = torch.full(size=(self.batch_size,), fill_value=logits.shape[0], dtype=torch.long)
        target_lengths = torch.full(size=(self.batch_size,), fill_value=labels.shape[0], dtype=torch.long)
        # print(logits.shape,labels.shape,input_lengths.shape,target_lengths.shape)
        # print(f"input_lengths = {input_lengths}")
        # print(f"target_lengths = {target_lengths}")
        #loss = nn.CTCLoss(blank=0).forward(logits, labels,input_length,target_length)
        ctc_loss =  nn.CTCLoss(blank=0)
        loss = ctc_loss(logits,labels,input_lengths,target_lengths)
        #loss_fct = nn.CrossEntropyLoss()
        #loss = loss_fct(logits.reshape(-1, self.seq2seq.config.vocab_size), labels.reshape(-1))
        self.log('val_loss', loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    @staticmethod
    def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
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



def run(hparams):
    print(hparams)
    model = Wav2SeqModel(hparams)
    trainer = Trainer(max_epochs=4,devices=1, accelerator="gpu", logger=wandb_logger)
    trainer.fit(model,SpeechRecognitionDataModule(WAV2VEC2_ARGS,num_workers=4))
    return 'completed'

    # {'audio': [{'path': None, 'array': array([-0.05114746, -0.11273193, -0.09152222, ...,  0.2796936 ,
    #         0.29998779,  0.20550537]), 'sampling_rate': 16000}, {'path': None, 'array': array([ 0.14523315,  0.05508423, -0.13534546, ...,  0.01959229,
    #         0.00973511, -0.00811768]), 'sampling_rate': 16000}], 'transcription': ["just a tryin' to survive", "don't go on me"], 'input_values':