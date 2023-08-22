import re
import torch
import json
import pandas as pd
import argparse
import numpy as np
from torch import nn
from torchmetrics.text import WordErrorRate
from typing import Optional
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from constants.mir_constants import TrainingArgs, WAV2VEC2_ARGS
from dataclasses import dataclass, asdict, field # type: ignore
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, BertTokenizer
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForCTC, AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset, Audio
from typing import Any, Dict, List, Optional, Union
from flash.audio import SpeechRecognition, SpeechRecognitionData
from training.wav2vec2_finetune import Wav2Vec2SpeechRecognition, SpeechRecognitionData


wandb_logger = WandbLogger(project="SLG - wav2vec2 transfer learning",log_model=True,)

print(json.dumps(asdict(WAV2VEC2_ARGS), indent = 4))


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
# {
#     'input_values': tensor([[ 0.1252,  0.1147, -0.0070,  ...,  0.0000,  0.0000,  0.0000], [ 0.0015, -0.0029, -0.0065,  ..., -0.0397, -0.0444,  0.0110]]), 
#     'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],[1, 1, 1,  ..., 1, 1, 1]]), 
#     'labels': tensor([[  101,  9294,  1185,  1119,   112,  1325,  1294,  2059,   102], [  101,  1105,   178,  1169,  1437,  1128,  1103, 11711,   102]]), 
#     'label_attention_masks': tensor([[0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0]])
#  }

#{
#        'input_values': tensor([[-0.1088,  0.7008,  0.0430,  ...,  0.0174, -0.6597, -1.5689],
#                               [ 0.0170,  0.0307,  0.0260,  ...,  0.0000,  0.0000,  0.0000]]),
#        'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1], [1, 1, 1,  ..., 0, 0, 0]]),
#        'labels': tensor([[  101, 15969, 19141,   146,   157, 10719, 12880, 27157,  2137,   102,  -100,  -100,  -100],
#                          [  101, 18732,  2591, 20521, 23676,  1592,  3663,   138,   160,  3048, 17656,  2036,   102]]), 
#        'label_attention_masks': tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
# }


    feature_extractor: Wav2Vec2FeatureExtractor
    bert_tokenizer: BertTokenizer
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        label_attention_features =[{"input_ids": feature["label_attention_mask"]} for feature in features]
        attention_features =[{"input_values": feature["attention_mask"]} for feature in features]
        
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt")

        batch_attention = self.feature_extractor.pad(
            attention_features,
            padding=self.padding,
            return_tensors="pt")
    
        labels_batch = self.bert_tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
            )

        label_attention_batch = self.bert_tokenizer.pad(
            label_attention_features,
            padding=self.padding,
            return_tensors="pt",
            )
                
        # replace padding with -100 to ignore loss correctly
        labels_attention = label_attention_batch["input_ids"].masked_fill(labels_batch.input_ids.eq(101), 0).masked_fill(labels_batch.input_ids.eq(102), 0)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), 0) #Changed from -100 as this is not Wav2Vec2 but it is BERT. The latter's pad token is 0.
        batch["labels"] = labels
        batch["label_attention_masks"] = labels_attention
        batch["attention_mask"] = batch_attention["input_values"]
        #print(batch)

        return batch


class SpeechRecognitionDataModule(LightningDataModule):
    def __init__(self, WAV2VEC2_ARGS: WAV2VEC2_ARGS, num_workers,hparams):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = WAV2VEC2_ARGS.BATCH_SIZE
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.wav2vec2_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained(hparams.lm_model)
        #self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_ARGS.MODEL_BACKBONE)
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        #self.feature_extractor = AutoFeatureExtractor.from_pretrained(hparams.wav2vec2_model)
        #self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.bert_tokenizer)
        #self.processor = AutoProcessor(feature_extractor=self.feature_extractor, tokenizer=self.bert_tokenizer)
        #self.processor = WhisperProcessor(feature_extractor=self.feature_extractor, tokenizer=self.bert_tokenizer)
        self.data_collator = DataCollatorCTCWithPadding(feature_extractor=self.feature_extractor, bert_tokenizer=self.bert_tokenizer, padding=True,)
        self.cpu_count = 4
        print(f"CPU Count = {self.cpu_count}")
    
    def setup(self, stage=None):
        train_df = pd.read_csv(WAV2VEC2_ARGS.TRAIN_FILE_PATH).head(1000)
        validation_df = pd.read_csv(WAV2VEC2_ARGS.TEST_FILE_PATH).head(10)
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
        batch["input_values"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["attention_mask"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).attention_mask[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = self.bert_tokenizer(batch["transcription"]).input_ids
        batch["label_attention_mask"] = self.bert_tokenizer(batch["transcription"]).attention_mask
        #print(batch)
        # with self.processor.as_target_processor():
        #     batch["labels"] = self.processor(batch["transcription"]).input_ids
        #     batch["label_attention_mask"] = self.processor(batch["transcription"]).attention_mask
        return batch

class Wav2SeqModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.batch_size = hparams.batch_size
        self.bert_tokenizer = BertTokenizer.from_pretrained(hparams.lm_model)
        speech_recognition_task = Wav2Vec2SpeechRecognition(wav2vec2_args=WAV2VEC2_ARGS)
        #self.wav2vec2 = AutoModelForCTC.from_pretrained(hparams.wav2vec2_model,ctc_zero_infinity=False,ctc_loss_reduction="sum")
        self.wav2vec2 = SpeechRecognition.load_from_checkpoint("/scratch/users/gmenon/model_artefacts/wav2vec2_demucs_en_large-960h-lv60-self_freeze_unfreeze_15epochs_adam.pt").model
        #self.wav2vec2.freeze_feature_encoder()
        #self.wav2vec2 = AutoModel.from_pretrained(hparams.wav2vec2_model)
        self.wav2vec2.eval()
        print(self.wav2vec2.config)
        self.seq2seq = AutoModelForCausalLM.from_pretrained(hparams.lm_model)
        self.seq2seq.config.is_decoder = True
        self.seq2seq.add_cross_attention = True
        self.seq2seq.train()
        bert_unwanted = np.zeros((1996,), dtype=int)
        bert_wanted = np.ones((len(self.bert_tokenizer.get_vocab())-1996,), dtype=int)
        self.weights = torch.Tensor(np.concatenate((bert_unwanted, bert_wanted), axis=0)).cuda()
        print(self.seq2seq.config)

    def forward(self, audio, attention_mask, labels, label_attention_mask):
        #print("entering forward step")
        encoder_outputs = self.wav2vec2(audio.squeeze(),
                                        attention_mask=attention_mask,
                                        output_hidden_states=True,
                                        output_attentions=True)
        encoder_hidden_states = encoder_outputs[0]  
        #encoder_attention_mask = self.wav2vec2._get_feature_vector_attention_mask(
        #        encoder_hidden_states.shape[1], 
        #        attention_mask
        #        )
        #print(f"[PRIOR]decoder_input_ids={labels}")
        #print(f"encoder_hidden_states = {encoder_hidden_states}")
        #print(f"[PRIOR]decoder_attention_masks={label_attention_mask}")
        #decoder_input_ids = self.shift_tokens_right(labels, 0, 101)
        #decoder_attention_masks = self.shift_tokens_right_mask(label_attention_mask)
        #print(f"Shape of labels ={labels.shape}, label_attention_mask = {label_attention_mask.shape}, decoder_input_ids ={decoder_input_ids.shape}, decoder_attention_mask = {decoder_attention_masks.shape}")
        #print(f"[AFTER]decoder_input_ids={decoder_input_ids}")
        #print(f"[AFTER]decoder_attention_masks={decoder_attention_masks}")
        decoder_outputs = self.seq2seq(input_ids=labels,
                                       attention_mask = label_attention_mask.squeeze(),
                                       encoder_hidden_states=encoder_hidden_states)
                                       #,encoder_attention_mask=encoder_attention_mask) 
        # decoder_outputs = self.seq2seq(input_ids=labels, attention_mask = label_attention_mask, encoder_attention_mask=encoder_attention_mask)
        return decoder_outputs

    def generate(self, audio):
        encoder_outputs = self.wav2vec2(audio[0],output_hidden_states=True,output_attentions=True)
        encoder_hidden_states = encoder_outputs[0]
        bos_ids = (
            torch.ones(
                (encoder_hidden_states.size()[0], 1),
                dtype=torch.long,
                device=self.seq2seq.device,
            )
            * self.seq2seq.config.pad_token_id
        )
        return self.seq2seq.generate(
            input_ids=bos_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

    def training_step(self, batch, batch_idx):
        #print("entering training step")
        audio = batch["input_values"].unsqueeze(0)
        attention_mask = batch["attention_mask"]
        label_attention_mask = batch["label_attention_masks"].unsqueeze(0)
        #print(f"audio={audio}, attention_mask = {attention_mask}, label_attention_mask = {label_attention_mask}")
        labels = batch["labels"]
        labels = labels.reshape(self.batch_size,-1)
        #labels = labels.reshape(-1,self.batch_size) 
        logits = self(audio,attention_mask,labels,label_attention_mask).logits
        logits = logits.reshape(-1, self.batch_size, self.seq2seq.config.vocab_size)
        input_lengths = torch.full(size=(self.batch_size,), fill_value=logits.shape[0], dtype=torch.long)
        target_lengths = torch.full(size=(self.batch_size,), fill_value=labels.shape[0], dtype=torch.long)
        loss = None
        ctc_loss =  nn.CTCLoss(blank=0)
        loss = ctc_loss(logits,labels,input_lengths,target_lengths)
        #print("Shape of Logits and Labels")
        #print(logits.shape,labels.shape)
        #ce_loss = nn.CrossEntropyLoss(weight=self.weights)
        #loss = ce_loss(logits.permute(0,2,1), labels.permute(1,0))
        return loss

    def validation_step(self, batch,batch_idx):
        #print("entering validation step")
        audio = batch["input_values"].unsqueeze(0)
        labels = batch["labels"]
        
        attention_mask = batch["attention_mask"]
        label_attention_mask = batch["label_attention_masks"].unsqueeze(0)
        #print(f"audio={audio}, attention_mask = {attention_mask}, label_attention_mask = {label_attention_mask}")
        #labels = labels.reshape(-1,self.batch_size) 
        labels = labels.reshape(self.batch_size,-1)
        logits = self(audio,attention_mask,labels,label_attention_mask).logits
        logits = logits.reshape(-1, self.batch_size, self.seq2seq.config.vocab_size)
        input_lengths = torch.full(size=(self.batch_size,), fill_value=logits.shape[0], dtype=torch.long)
        target_lengths = torch.full(size=(self.batch_size,), fill_value=labels.shape[0], dtype=torch.long)
        loss=None
        ctc_loss =  nn.CTCLoss(blank=0)
        loss = ctc_loss(logits,labels,input_lengths,target_lengths)
        #print("Shape of Logits and Labels")
        #print(logits.shape,labels.shape)
        #ce_loss = nn.CrossEntropyLoss(weight=self.weights)
        #loss = ce_loss(logits.permute(0,2,1), labels.permute(1,0))
        predicted_ids = torch.argmax(logits, dim=-1)
        #label_decoded = labels.type(torch.int32).tolist()
        #print(label_decoded)
        #print(f"labels = {labels}")
        #print(f"Shape/predicted ids = {predicted_ids.shape},{predicted_ids}")
        #print(f"original transcript = {batch['transcription']}")
        print(f"original text = {self.bert_tokenizer.decode(labels[0],skip_special_tokens=False)},{self.bert_tokenizer.decode(labels[1],skip_special_tokens=False)}")
        #predicted_text = predicted_ids.type(torch.int32)
        print(f"Predicted text = {self.bert_tokenizer.decode(predicted_ids.T[0,:],skip_special_tokens=False)},{self.bert_tokenizer.decode(predicted_ids.T[1,:],skip_special_tokens=False)}")
        self.log('val_loss', loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        print("Entering Optimization Step")
        return torch.optim.AdamW(self.parameters(), lr=hparams.learning_rate)

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

    @staticmethod
    def shift_tokens_right_mask(input_ids: torch.Tensor):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:,:, 1:] = input_ids[:,:, :-1].clone()
        shifted_input_ids[:,:, 0] = 0
        return shifted_input_ids.squeeze()

def run(hparams):
    print(hparams)
    model = Wav2SeqModel(hparams)
    trainer = Trainer(max_epochs=10,devices=1, accelerator="gpu", logger=wandb_logger)
    trainer.fit(model,SpeechRecognitionDataModule(WAV2VEC2_ARGS,num_workers=4,hparams=hparams))
    return model, trainer


hparams = argparse.Namespace()
hparams.wav2vec2_model = 'facebook/wav2vec2-large-960h-lv60-self'
hparams.lm_model = 'bert-base-uncased' #'facebook/bart-large'
hparams.vocab_size = 40000
hparams.learning_rate = 1e-3
hparams.batch_size = 2

model,trainer = run(hparams=hparams)