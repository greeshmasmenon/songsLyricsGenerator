from handlers.dali_dataset import  DALIDataset
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

if __name__ == '__main__':
    dali = DALIDataset(data_path="/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/")
    dali_dataset = dali.get_data()
    print(dali_dataset)
    dali_info = dali.get_info()
    dali_info.to_csv("/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/info/dali_info.csv")
    # print(dali_info)
    # dali_audio_errors = dali.download_audio()
