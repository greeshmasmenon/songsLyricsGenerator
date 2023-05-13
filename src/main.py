# This is a sample Python script.
import io.dali_dataset


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dali = dali_dataset.DALIDataset()
    dali.data_path = "/Users/macbook/PycharmProjects/songsLyricsGenerator/data/DALI_v1.0/"
    dali_dataset = dali.get_data()
    dali_info = dali.get_info()
    print(dali_info)