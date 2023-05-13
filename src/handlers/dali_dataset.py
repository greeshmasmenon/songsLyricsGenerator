from abc import ABC
import DALI as dali_code
import logging
from typing import Dict
import pandas as pd
import numpy as np

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
__all__ = ["DALIDataset"]

class DALIDataset():
    def __init__(self):
        self._data_path = None

    @property
    def data_path(self):
        logging.info("Setting the data_path")
        return self._data_path

    @property.setter
    def data_path(self, data_path: str):
        logging.info("Setting the data_path")
        self._data_path = data_path

    def get_data(self) -> Dict:
        logging.info("Getting the data_path")
        return dali_code.get_the_DALI_dataset(self._data_path, skip=[], keep=[])

    def get_info(self) -> pd.DataFrame:
        logging.info("Extracting the info related to the data from the data_path")
        dali_info = dali_code.get_info(self._data_path + 'info/DALI_DATA_INFO.gz')
        dali_df = pd.DataFrame(dali_info)[1:]
        dali_df.columns = dali_info[0]
        return dali_df


