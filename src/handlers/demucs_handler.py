from demucs import separate
import pandas as pd
from utils.csv_utils import CsvWriter

song_metadata = pd.read_csv("/home/users/gmenon/notebooks/home/users/gmenon/notebooks/song_metadata_cleaned.csv")
csvwriter = CsvWriter(file_name="separated_audio.csv")
completed_list =[]
for file_name in song_metadata.file_name:
    print(file_name)
    separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", str(file_name)])
    completed_list.append(file_name)
csvwriter.csv_write_from_list(header = ["completed_audios"], data = completed_list)


# for i in *.wav; do demucs --two-stems=vocals "$i" ; echo "completed $i"; done