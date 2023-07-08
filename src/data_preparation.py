from datasets.dali_dataset import  DALIDataset
from constants.mir_constants import HPCDataConstants, MacDataConstants
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', level=logging.INFO)



def data_preparation(dali:DALIDataset) -> str:
    """
    Data Preparation Step for the dali dataset.
    """

    dali.get_dataset()
    logging.info(f"dali dataset has been downloaded")
    dali.download_information()
    logging.info(f"dali_info has been downloaded")
    dali_audio_errors = dali.download_audio()
    logging.info(f"# Errors in the download of DALI dataset = {len(dali_audio_errors)}")
    return "completed"


if __name__ == "__main__":
    dali = DALIDataset(data_path= HPCDataConstants.DATASET_PATH,
                        audio_file_path = HPCDataConstants.AUDIO_FILE_PATH,
                        info_path = HPCDataConstants.DATASET_INFO_PATH)
    data_preparation(dali)