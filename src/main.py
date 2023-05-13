from handlers.dali_dataset import  DALIDataset
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dali = DALIDataset(data_path="/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/", file_path="/Users/macbook/PycharmProjects/songsLyricsGenerator/data/")
    logging.info(f"Starting DALI Dataset download")
    dali_dataset = dali.get_data()
    logging.info(f"The DALI dataset has been downloaded")
    dali_info = dali.get_info()
    logging.info(f"The DALI dataset has {len(dali_info)} rows in it")
    dali_audio_errors = dali.get_audio()
    logging.info(f"The DALI Audio download has {len(dali_info)} errors in it")

