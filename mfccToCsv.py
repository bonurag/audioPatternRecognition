import librosa.display
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm

eps = sys.float_info.epsilon

# create a csv file from data
DATASET_LIST = ["genres_5", "genres_10"]
MFCC_LIST = [13, 20, 40]


def energy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float64(len(frame))


def energy_entropy(frame, n_short_blocks=10):
    """Computes entropy of energy"""
    # total frame energy
    frame_energy = np.sum(frame ** 2)
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]

    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy


for num_mfcc in MFCC_LIST:
    for dataset_path in DATASET_LIST:
        genre = os.listdir(dataset_path)
        data = pd.DataFrame(
            columns=["filename", "genre", "path", "tempo", "energy", "energy_entropy", "rmse", "chroma_stft", "chroma_cqt",
                     "chroma_cens", "spec_cent",
                     "spec_bw", "spec_contrast", "rolloff", "zcr"] + [
                        "mfcc_{}".format(x) for x in range(num_mfcc)])

        filenames = []
        genres = []
        paths = []

        for gnr in genre:
            for file in os.listdir(os.path.join(dataset_path, gnr)):
                filenames.append(file)
                genres.append(gnr)
                paths.append(os.path.join(dataset_path, gnr, file))
        data["filename"] = filenames
        data["genre"] = genres
        data["path"] = paths

        data.head()
        num_iterations = len(genre)*1000
        for i, row in tqdm(data.iterrows(), total=num_iterations):
            audio, sr = librosa.load(row.path)
            # print('\nTrack: {} - Audio: {}, SampleRate: {}'.format(i, row.path, sr))

            row["tempo"] = np.mean(librosa.beat.tempo(audio, sr=sr))
            row["energy"] = np.mean(energy(audio))
            row["energy_entropy"] = np.mean(energy(audio))
            row["rmse"] = np.mean(librosa.feature.rms(audio))
            row["chroma_stft"] = np.mean(librosa.feature.chroma_stft(audio, sr=sr))
            row["chroma_cqt"] = np.mean(librosa.feature.chroma_cqt(audio, sr=sr))
            row["chroma_cens"] = np.mean(librosa.feature.chroma_cens(audio, sr=sr))
            row["spec_cent"] = np.mean(librosa.feature.spectral_centroid(audio, sr=sr))
            row["spec_bw"] = np.mean(librosa.feature.spectral_bandwidth(audio, sr=sr))
            row["spec_contrast"] = np.mean(librosa.feature.spectral_contrast(audio, sr=sr))
            row["rolloff"] = np.mean(librosa.feature.spectral_rolloff(audio, sr=sr))
            row["zcr"] = np.mean(librosa.feature.zero_crossing_rate(audio))

            for x, j in zip(librosa.feature.mfcc(audio, sr=sr, n_mfcc=num_mfcc)[:num_mfcc], range(num_mfcc)):
                row["mfcc_{}".format(j)] = np.mean(x)

        num_files = "5000" if dataset_path == "genres_5" else "10000"
        num_genres = "5GEN" if dataset_path == "genres_5" else "10GEN"

        FILE_NAME = str(num_mfcc) + "MFCC_" + num_files + "_" + num_genres + "_GTZAN.csv"
        SAVE_ROOT = "results/" + str(dataset_path) + '/'
        data.to_csv(SAVE_ROOT+FILE_NAME)
        data.head()