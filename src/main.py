from handlers.dali_dataset import  DALIDataset
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

if __name__ == '__main__':
    dali = DALIDataset(data_path="/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/", file_path="/Users/macbook/PycharmProjects/songsLyricsGenerator/data/")
    dali_dataset = dali.get_data()
    print(dali_dataset)
    dali_info = dali.get_info()
    print(dali_info)
    dali_audio_errors = dali.get_audio()


