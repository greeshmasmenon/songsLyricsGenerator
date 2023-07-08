from dataclasses import dataclass

@dataclass(frozen = True)
class HPCDataConstants:
    """constants for the use within the Songs to Lyrics generation"""
    NAME: str = "Dali Dataset"
    DATASET_PATH: str = "/home/users/gmenon/workspace/songsLyricsGenerator/data/DALI_v1.0/"
    DATASET_INFO_PATH: str = "/home/users/gmenon/workspace/songsLyricsGenerator/data/DALI_v1.0/info/DALI_DATA_INFO.gz"
    AUDIO_FILE_PATH: str = "/home/users/gmenon/workspace/songsLyricsGenerator/data/DALI_v1.0/audio/"
    DATASET_METADATA: str = ""
    DATASET_CLEANED_METADATA: str = ""


@dataclass(frozen = True)
class MacDataConstants:
    """constants for the use within the Songs to Lyrics generation"""
    NAME: str = "Dali Dataset"
    DATASET_PATH: str = "/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/"
    DATASET_INFO_PATH: str = "/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/info/DALI_DATA_INFO.gz"
    AUDIO_FILE_PATH: str = "/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/audiodownload/"
    DATASET_METADATA: str = ""
    DATASET_CLEANED_METADATA: str = ""