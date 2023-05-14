from abc import ABC
import DALI as dali_code
import logging
from typing import Dict, Optional, List
import pandas as pd
import numpy as np

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
__all__ = ["DALIDataset"]

class DALIDataset():
    def __init__(self, data_path: str , file_path: Optional[str] = None):
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
            dali_info = self.get_info()
            logging.info(f"The DALI Audio download has {len(dali_info)} errors in it")
            return dali_code.get_audio(dali_info, self._file_path, skip=[], keep=[])
        else:
            raise TypeError(f"Set the data_path & file_path for the location of the DALI datasets; "
                            f"data_path = {self._data_path}, file_path = {self._file_path}")




