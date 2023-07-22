import re
import data
from ftlangdetect import detect
import pandas as pd
import numpy as np
from data import load_dataset, Dataset, Audio, load_metric, load_dataset, Metric
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import TrainingArguments, Trainer, Wav2Vec2ForCTC
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

CHARS_TO_REMOVE_FROM_TRANSCRIPTS = '[\,\?\.\!\-\;\:\"\%\$\&\^\*\@\#\<\>\/\+\\=\_\\}\{\)\(\]\[\`1234567890]'


def detect_language(text: str) -> dict:
    """Simple Function to detect whether the language of the text is english or not.
    Returns a Boolean output for the same.Input the text column"""
    return detect(text, low_memory=True)["lang"]


def songs_metadata_language_filtered(song_metadata: pd.DataFrame, language: str = "en") -> pd.DataFrame:
    song_metadata["transcription"] = song_metadata["transcription"].replace(CHARS_TO_REMOVE_FROM_TRANSCRIPTS, '',
                                                                            regex=True)
    song_metadata = song_metadata[song_metadata.transcription.str.len() > 8]
    song_metadata["language"] = song_metadata["transcription"].apply(detect_language)
    song_metadata = song_metadata.loc[song_metadata["language"] == language]
    return song_metadata


def create_huggingface_datasets(song_metadata: pd.DataFrame, sampling_rate=16000) -> Dataset:
    audio_dataset = Dataset.from_dict(
        {"audio": list(song_metadata["file_name"]),
         "transcription": list(song_metadata["transcription"])}).cast_column("audio",
                                                                             Audio(sampling_rate=sampling_rate))
    return audio_dataset


def train_test_split_audio(audio_dataset: Dataset, test_size=0.2, shuffle=True) -> Dataset:
    return audio_dataset.train_test_split(test_size=test_size, shuffle=shuffle)


def audio_transcripts_remove_special_char(batch: Dataset) -> Dataset:
    # batch["transcription"] = re.sub(chars_to_ignore_regex, '', batch["transcription"]).lower() + ' '
    batch["transcription"] = re.sub(CHARS_TO_REMOVE_FROM_TRANSCRIPTS, '', batch["text"]).upper()
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["transcription"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def extract_tokens(audio_dataset: Dataset, download_json_file_path="vocab.json"):
    vocabs = audio_dataset.map(extract_all_chars,
                               batched=True,
                               batch_size=-1,
                               keep_in_memory=True,
                               remove_columns=audio_dataset.column_names["train"])
    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open(download_json_file_path, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)


def create_tokenizer(download_json_file_path="vocab.json") -> Wav2Vec2CTCTokenizer:
    return Wav2Vec2CTCTokenizer(download_json_file_path,
                                unk_token="[UNK]",
                                pad_token="[PAD]",
                                word_delimiter_token="|")


def create_huggingface_feature_extractor(download_json_file_path="vocab.json") -> Wav2Vec2FeatureExtractor:
    return Wav2Vec2FeatureExtractor(feature_size=1,
                                    sampling_rate=16000,
                                    padding_value=0.0,
                                    do_normalize=True,
                                    return_attention_mask=False)


def create_huggingface_processor(feature_extractor: Wav2Vec2FeatureExtractor,
                                 tokenizer: Wav2Vec2CTCTokenizer) -> Wav2Vec2Processor:
    return Wav2Vec2Processor(feature_extractor=feature_extractor,
                             tokenizer=tokenizer)


def transform_huggingface_dataset(batch: Dataset, processor: Wav2Vec2Processor) -> Dataset:
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch


def train_dataset_transform(audio_dataset: Dataset, num_processes=1, ) -> Dataset:
    return audio_dataset.map(transform_huggingface_dataset, \
                             remove_columns=audio_dataset.column_names["train"], \
                             num_proc=num_processes)


def filter_huggingface_dataset_by_time_duration(audio_dataset: Dataset, processor: Wav2Vec2Processor, time_duration=6):
    audio_dataset["train"] = audio_dataset["train"]. \
        filter(lambda x: x < time_duration * processor.feature_extractor.sampling_rate,
               input_columns=["input_length"])


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for processing the data.
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


def data_collator(custom_processor: Wav2Vec2Processor, padding: bool) -> DataCollatorCTCWithPadding:
    return DataCollatorCTCWithPadding(processor=custom_processor, padding=padding)


def wer_metric() -> Metric:
    return load_metric("wer")


def compute_metrics(pred: Dataset, processor: Wav2Vec2Processor) -> Dict:
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def huggingface_train_asr(audio_dataset:Dataset,
                          model:Wav2Vec2ForCTC,
                          processor: Wav2Vec2Processor,
                          save_location: str):

    training_args = TrainingArguments(
        output_dir="songstolyrics_wav2vec",
        group_by_length=True,
        per_device_train_batch_size=8,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=audio_dataset["train"],
        eval_dataset=audio_dataset["test"],
        tokenizer=processor.feature_extractor,
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_state()
    trainer.save_model(save_location)
    return "Success"


def load_model(model_name: str, processor: Wav2Vec2Processor) -> Wav2Vec2ForCTC:

    return Wav2Vec2ForCTC.from_pretrained(
        model_name,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )