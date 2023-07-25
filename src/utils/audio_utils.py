from ftlangdetect import detect
import pandas as pd
import re
from constants.mir_constants import CHARS_TO_REMOVE_FROM_TRANSCRIPTS
from data import load_dataset, Dataset, Audio, load_metric, load_dataset, Metric


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
         "transcription": list(song_metadata["transcription"])
         }).cast_column("audio",Audio(sampling_rate=sampling_rate))
    return audio_dataset


def train_test_split_audio(audio_dataset: Dataset, test_size=0.2, shuffle=True) -> Dataset:
    return audio_dataset.train_test_split(test_size=test_size, shuffle=shuffle)


def audio_transcripts_remove_special_char(batch: Dataset) -> Dataset:
    batch["transcription"] = re.sub(CHARS_TO_REMOVE_FROM_TRANSCRIPTS, '', batch["text"]).upper()
    return batch

