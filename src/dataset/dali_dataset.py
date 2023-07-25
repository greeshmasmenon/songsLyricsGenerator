import _pickle

from data.base_dataset import Dataset
import DALI as dali_code
import logging
import pandas as pd
import uuid
import csv
import os
import numpy as np
from pydub import AudioSegment
from typing import Dict, Optional, List

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
__all__ = ["DALIDataset"]


class DALIDataset(Dataset):

    def __init__(self, data_path: str, audio_file_path: Optional[str] = None, info_file_path=Optional[str]):

        if data_path is None:
            raise KeyError("Enter the value for data_path")
        else:
            self._data_path = data_path

        if audio_file_path is None:
            self._audio_file_path = self._data_path + 'audio/'
        else:
            self._audio_file_path = audio_file_path

        if info_file_path is None:
            self._info_path = self._data_path + 'info/DALI_DATA_INFO.gz'
        elif str.split(info_file_path, ".")[-1] == "gz":
            self._info_path = info_file_path
        else:
            self._info_path = info_file_path + '/DALI_DATA_INFO.gz'

    @property
    def data_path(self):
        logging.info("Returning the data_path")
        return self._data_path

    @data_path.setter
    def data_path(self, data_path: str):
        logging.info("Setting the data_path")
        self._data_path = data_path

    @property
    def audio_file_path(self):
        logging.info("Returning the audio_file_path")
        return self._audio_file_path

    @audio_file_path.setter
    def file_path(self, audio_file_path: str):
        logging.info("Setting the audio_file_path")
        self._audio_file_path = audio_file_path

    @property
    def info_path(self):
        logging.info("Returning the info_path")
        return self._info_path

    @info_path.setter
    def info_path(self, info_path: str):
        logging.info("Setting the info_path")
        self._info_path = info_path

    def get_dataset(self) -> Dict:
        logging.info(f"Getting the data_path = {self._data_path}")
        if self._data_path is not None:
            dali_dataset = dali_code.get_the_DALI_dataset(self._data_path, skip=[], keep=[])
            logging.info(f"The DALI dataset has been downloaded")
            return dali_dataset
        else:
            raise KeyError(f"Set the data_path for the location of the DALI datasets; data_path = {self._data_path}")

    def download_dataset(self) -> NotImplementedError:
        raise NotImplementedError

    def get_information(self) -> pd.DataFrame:
        logging.info(f"Getting the info related to the data from the data_path = {self._info_path}")
        if self._info_path is not None:
            dali_info = dali_code.get_info(self._info_path)
            dali_df = pd.DataFrame(dali_info)[1:]
            dali_df.columns = dali_info[0]
            logging.info(f"The DALI dataset has {len(dali_info)} rows in it")
            return dali_df
        else:
            raise TypeError(f"Set the info_path for the location of the DALI datasets; info_path = {self._info_path}")

    def download_information(self) -> None:
        dali_df = self.get_information()
        logging.info(f"Downloading to the info file path = {self._info_path}info/ ")
        dali_df.to_csv(self._info_path)
        logging.info(f"Download complete in the file path = {self._info_path}info/ ")

    def get_audio_files(self) -> NotImplementedError:
        raise NotImplementedError

    def download_audio(self) -> List[str]:
        """
        This particular function returns the errors from where the Audio
        """
        logging.info("Downloading audio from youtube URLs associated with the information file")
        if self._data_path is not None or self._audio_file_path is not None:
            try:
                dali_dataset_info = self.get_information()
            except _pickle.UnpicklingError:
                # TODO: Need to clean this code here
                logging.info("Received unpicklingerror. Taking backup option of CSV info files")
                dali_dataset_info = pd.read_csv(
                    "/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/info/dali_info.csv"
                    , index_col = 0
                                                )
            errored_dali_ids = dali_code.get_audio(dali_dataset_info, self._audio_file_path, skip=[], keep=[])
            logging.info(f"The DALI Audio download has {len(errored_dali_ids)} errors in it")
            return errored_dali_ids
        else:
            raise TypeError(f"Set the data_path & audio_file_path for the location of the DALI datasets; "
                            f"data_path = {self._data_path}, audio_file_path = {self._audio_file_pathh}")

    def extract_dali_id_from_directory(self, extension: str) -> List[str]:
        audio_files = os.listdir(self._audio_file_path)
        extract_dali_id = lambda x: x.split('.')[0]
        extract_file_extension = lambda x: x.split('.')[1]
        dali_ids = [extract_dali_id(file_name) for file_name \
                    in audio_files if extract_file_extension(file_name) == extension]
        logging.info(f"dali ids extracted from the file system directory = {self._audio_file_pat}")
        return dali_ids

    def split_align_wav_transcripts(self, source_audio_path: str, destination_audio_path: str, extension: str = '.wav'):
        header = ["dali_id", "segment_start", "segment_end", "file_name", "transcription"]
        dali_ids = self.extract_dali_id_from_directory(source_audio_path, extension)
        dali_dataset = self.get_data()
        # TODO: Make this save in chunks of 5000 rather than all in one CSV File.
        metadata_csv_save_path = destination_audio_path + "metadata.csv"
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
                    logging.info(
                        f"wav file saved at {destination_audio_path + extracted_audio_filename} and has transcription = {transcript}")
