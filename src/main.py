from handlers.dali_dataset import  DALIDataset
from mir_constants import DataConstants
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', level=logging.INFO)

if __name__ == '__main__':
    DALI_CONSTANTS = DataConstants(name = "Dali dataset",
                               dataset_path = "/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/",
                               dataset_info_path = "/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/info/dali_info.csv",
                               dataset_metadata = "",
                               dataset_cleaned_metadata = "")

    dali = DALIDataset(data_path=DALI_CONSTANTS.dataset_path)
    dali_dataset = dali.get_data()
    print(dali_dataset)
    dali_info = dali.get_info()
    dali_info.to_csv(DALI_CONSTANTS.dataset_info_path)
    # print(dali_info)
    # dali_audio_errors = dali.download_audio()

