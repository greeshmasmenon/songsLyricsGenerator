import csv
import os
from typing import Union, List, Dict, Literal
import logging

logging.basicConfig(filename='app.log', filemode='w',
                    format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', level=logging.INFO)


class CsvWriter:
    def __init__(self, file_name:str, save_path = os.getcwd()):
        self._file_name = file_name
        self._save_path = save_path

    @property
    def file_name(self):
        return self._file_name

    @file_name.setter
    def file_name(self, file_name: str):
        logging.info("Setting the file_name")
        self._file_name = file_name

    @property
    def save_path(self):
        return self._save_path

    @save_path.setter
    def save_path(self, save_path: str):
        logging.info("Setting the save_path")
        self._save_path = save_path

    def csv_write_from_list(self, header: List, data: List) -> None:
        """
        function to write csv files from lists

        :param header: Header to be saved within the file
        :param data: data to write from a list
        :return: CSV File
        """
        with open(f"{self._save_path}/{self._file_name}", 'w', encoding='UTF8', newline='') as f:
            logging.info(f"CSV Writer start. Data sample = {data[0]}")
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
            writer.writerows(data)

    def csv_write_from_dict(self, header: List, data: Dict) -> None:
        """
        function to write csv files from dictionary
        :param header: header to write from a dictionary
        :param data: data to write from a dictionary
        :return: CSV file
        """
        with open(f"{self._save_path}/{self._file_name}", 'w', encoding='UTF8', newline='') as f:
            logging.info(f"CSV Writer start. Dictionary Key sample = {list(data.keys())[0]}")
            writer = csv.DictWriter(f, fieldnames = header)
            writer.writerows(data)











