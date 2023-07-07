from abc import ABC
import DALI as dali_code
import logging
import pandas as pd
import uuid
import csv
import os
import numpy as np
from pydub import AudioSegment
from typing import Dict, Optional, List

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', level=logging.INFO)
__all__ = ["DALIDataset"]


class DALIDataset:

    def __init__(self, data_path: str, file_path: Optional[str] = None):
        self._data_path = data_path
        if file_path is None:
            self._file_path = self._data_path + 'audio/'
        else:
            self._file_path = file_path

    @property
    def data_path(self):
        logging.info("Setting the data_path")
        return self._data_path

    @data_path.setter
    def data_path(self, data_path: str):
        logging.info("Setting the data_path")
        self._data_path = data_path

    @property
    def file_path(self):
        logging.info("Setting the data_path")
        return self._file_path

    @file_path.setter
    def file_path(self, file_path: str):
        logging.info("Setting the data_path")
        self._file_path = file_path

    def get_data(self) -> Dict:
        logging.info("Getting the data_path")
        if self._data_path is not None:
            dali_dataset = dali_code.get_the_DALI_dataset(self._data_path, skip=[], keep=[])
            logging.info(f"The DALI dataset has been downloaded")
            return dali_dataset
        else:
            raise TypeError(f"Set the data_path for the location of the DALI datasets; data_path = {self._data_path}")

    def download_data(self) -> NotImplementedError:
        # dali_data = self.get_data()
        # logging.info(f"Downloading the data into the file path = {self._data_path}data/")
        raise NotImplementedError

    def get_info(self) -> pd.DataFrame:
        logging.info(f"Getting the info related to the data from the data_path = {self._data_path}")
        if self._data_path is not None:
            dali_info = dali_code.get_info(self._data_path + 'info/DALI_DATA_INFO.gz')
            dali_df = pd.DataFrame(dali_info)[1:]
            dali_df.columns = dali_info[0]
            logging.info(f"The DALI dataset has {len(dali_info)} rows in it")
            return dali_df
        else:
            raise TypeError(f"Set the data_path for the location of the DALI datasets; data_path = {self._data_path}")

    def download_info(self) -> None:
        dali_df = self.get_info()
        logging.info(f"Downloading to the file path = {self._data_path}info/ ")
        dali_df.to_csv(self._data_path + 'info/dali_info.csv')
        logging.info(f"Download complete in the file path = {self._data_path}info/ ")

    def download_audio(self) -> List:
        logging.info("Downloading audio from youtube URLs associated with the info file")
        if self._data_path is not None or self._file_path is not None:
            dali_dataset_info = self.get_info()
            logging.info(f"The DALI Audio download has {len(dali_dataset_info)} errors in it")
            return dali_code.get_audio(dali_dataset_info, self._file_path, skip=[], keep=[])
        else:
            raise TypeError(f"Set the data_path & file_path for the location of the DALI datasets; "
                            f"data_path = {self._data_path}, file_path = {self._file_path}")

    def extract_dali_id_from_directory(self, path: str, extension: str) -> List[str]:
        wav_files = os.listdir(path)
        extract_dali_id = lambda x : x.split('.')[0]
        extract_file_extension = lambda x : x.split('.')[1]
        dali_ids = [extract_dali_id(file_name) for file_name \
                    in wav_files if extract_file_extension(file_name) == extension]
        logging.info(f"dali ids extracted from the file system directory = {path}")
        return dali_ids

    def split_align_wav_transcripts(self, source_audio_path: str, destination_audio_path: str, extension: str = '.wav'):
        header = ["file_name", "transcription"]
        metadata_csv_save_path = destination_audio_path + "metadata.csv"
        dali_ids = self.extract_dali_id_from_directory(source_audio_path, extension)
        dali_dataset = self.get_data()
        with open(metadata_csv_save_path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
            for dali_id in dali_ids:
                logging.info(f"Extracting the data for the dali_id = {dali_id}")
                for segment in dali_dataset[dali_id].annotations["annot"]["lines"]:
                    logging.info(f"frequency = {segment['freq']}, time = {segment['time']}, text = {segment['text']}")
                    segment_start, segment_end = segment['time']
                    transcript = segment['text']
                    source_audio_file = AudioSegment.from_wav(source_audio_path + dali_id + extension)
                    extracted_audio_segment = source_audio_file[segment_start * 1000:segment_end * 1000]
                    extracted_audio_filename = uuid.uuid4().hex + '.wav'
                    extracted_audio_segment.export(destination_audio_path + extracted_audio_filename)
                    writer.writerow([dali_id, segment_start, segment_end, extracted_audio_filename, transcript])
                    logging.info(f"wav file saved at {destination_audio_path + extracted_audio_filename} and has transcription = {_transcript}")

