import librosa.display
import numpy as np
import os
import tqdm.notebook
import pandas as pd

# create a csv file from data
DATASET_PATH = "test_1"
NUM_MFCC = 40

genre = os.listdir(DATASET_PATH)
data = pd.DataFrame(
    columns=["filename", "genre", "path", "tempo", "rmse", "chroma_stft", "chroma_cqt", "chroma_cens", "spec_cent", "spec_bw", "rolloff", "zcr"] + [
        "mfcc_{}".format(x) for x in range(NUM_MFCC)])

filenames = []
genres = []
paths = []

for gnr in genre:
    for file in os.listdir(os.path.join(DATASET_PATH, gnr)):
        filenames.append(file)
        genres.append(gnr)
        paths.append(os.path.join(DATASET_PATH, gnr, file))
data["filename"] = filenames
data["genre"] = genres
data["path"] = paths

data.head()

for i, row in tqdm.notebook.tqdm(data.iterrows(), total=1000):
    audio, sr = librosa.load(row.path)
    onset_env = librosa.onset.onset_strength(audio, sr=sr)
    print('Index: ', i)
    print('Audio: {}, SampleRate: {}'.format(row.path, sr))

    row["tempo"] = np.mean(librosa.beat.tempo(audio, sr=sr))
    row["rmse"] = np.mean(librosa.feature.rms(audio))
    row["chroma_stft"] = np.mean(librosa.feature.chroma_stft(audio, sr=sr))
    row["chroma_cqt"] = np.mean(librosa.feature.chroma_stft(audio, sr=sr))
    row["chroma_cens"] = np.mean(librosa.feature.chroma_stft(audio, sr=sr))
    row["spec_cent"] = np.mean(librosa.feature.spectral_centroid(audio, sr=sr))
    row["spec_bw"] = np.mean(librosa.feature.spectral_bandwidth(audio, sr=sr))
    row["rolloff"] = np.mean(librosa.feature.spectral_rolloff(audio, sr=sr))
    row["zcr"] = np.mean(librosa.feature.zero_crossing_rate(audio))

    for x, j in zip(librosa.feature.mfcc(audio, sr=sr, n_mfcc=NUM_MFCC)[:NUM_MFCC], range(NUM_MFCC)):
        row["mfcc_{}".format(j)] = np.mean(x)

data.to_csv("GenreMusicFeaturesV3.csv")
data.head()
