from dataclasses import dataclass

@dataclass(frozen = True)
class DataConstants:
    """constants for the use within the Songs to Lyrics generation"""
    name: str
    dataset_path: str
    dataset_info_path: str
    dataset_metadata: str
    dataset_cleaned_metadata: str



