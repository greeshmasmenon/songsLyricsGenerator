from abc import ABC
import DALI as dali_code
import logging
from typing import Dict
import pandas as pd
import numpy as np

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
__all__ = ["DALIDataset"]

class DALIDataset():
    def __init__(self, data_path: str , file_path: str):
        self._data_path = data_path
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

    def get_audio(self) -> None:
        logging.info("Downloading audio from youtube URLs associated with the info file")
        if self._data_path is not None or self._file_path is not None:
            dali_info = self.get_info()
            logging.info(f"The DALI Audio download has {len(dali_info)} errors in it")
            dali_code.get_audio(dali_info, self._file_path, skip=[], keep=[])
        else:
            raise TypeError(f"Set the data_path & file_path for the location of the DALI datasets; "
                            f"data_path = {self._data_path}, file_path = {self._file_path}")




