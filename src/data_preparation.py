from datasets.dali_dataset import  DALIDataset
from mir_constants import DataConstants, MacDataConstants
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', level=logging.INFO)



def data_preparation(DALIDataset) -> str:
    """
    Data Preparation Step for the dali dataset.
    """

    dali = DALIDataset(data_path= MacDataConstants)
    dali.get_data()
    logging.info(f"dali dataset has been downloaded")
    dali.download_info()
    logging.info(f"dali_info has been downloaded")
    dali_audio_errors = dali.download_audio()
    logging.info(f"# Errors in the download of DALI dataset = {len(dali_audio_errors)}")


