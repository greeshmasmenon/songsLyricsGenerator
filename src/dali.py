from handlers.dali_dataset import  DALIDataset
from mir_constants import DataConstants
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', level=logging.INFO)

if __name__ == '__main__':
    # DALI_CONSTANTS = DataConstants(name = "Dali dataset",
    #                            dataset_path = "/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/",
    #                            dataset_info_path = "/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/info/dali_info.csv",
    #                            dataset_metadata = "",
    #                            dataset_cleaned_metadata = "")

    dali = DALIDataset(data_path= DataConstants)
    dali_dataset = dali.get_data()
    logging.info(f"dali dataset has been downloaded")
    dali_info = dali.get_info()
    logging.info(f"dali_info has been obtained")
    dali_info.to_csv(DataConstants.DATASET_INFO_PATH)
    logging.info(f"dali_info has been saved in csv at location = {DataConstants.DATASET_INFO_PATH}")
    dali_audio_errors = dali.download_audio()
    logging.info(f"# Errors in the download of DALI dataset = {len(dali_audio_errors)}")


