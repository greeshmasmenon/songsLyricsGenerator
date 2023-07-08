from datasets.dali_dataset import DALIDataset
from constants.mir_constants import HPCDataConstants, MacDataConstants
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


def data_preparation(dali_dataset: DALIDataset) -> str:
    """
    Data Preparation Step for the dali dataset.
    """

    dali_dataset.get_dataset()
    dali_dataset.download_information()
    dali_audio_errors = dali_dataset.download_audio()
    return f"completed with {len(dali_audio_errors)} errors}"


if __name__ == "__main__":
    dali = DALIDataset(data_path=HPCDataConstants.DATASET_PATH,
                       audio_file_path=HPCDataConstants.AUDIO_FILE_PATH,
                       info_path=HPCDataConstants.DATASET_INFO_PATH)
    data_preparation(dali)
