import librosa.display
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import features_functions

# create a csv file from data
DATASET_LIST = ['genres_5', 'genres_10']
MFCC_LIST = [13, 20, 40]

for num_mfcc in MFCC_LIST:
    for dataset_path in DATASET_LIST:
        genre = os.listdir(dataset_path)
        data = pd.DataFrame(
            columns=['filename', 'genre', 'path', 'tempo', 'energy', 'energy_entropy', 'rmse',
                     'chroma_stft', 'chroma_cqt', 'chroma_cens', 'spec_cent',
                     'spec_bw', 'spec_contrast', 'rolloff', 'zcr'] + ['mfcc_{}'.format(x) for x in range(num_mfcc)])

        filenames = []
        genres = []
        paths = []

        for gnr in genre:
            for file in os.listdir(os.path.join(dataset_path, gnr)):
                filenames.append(file)
                genres.append(gnr)
                paths.append(os.path.join(dataset_path, gnr, file))
        data['filename'] = filenames
        data['genre'] = genres
        data['path'] = paths

        data.head()
        num_iterations = len(genre)*1000
        for i, row in tqdm(data.iterrows(), total=num_iterations):
            audio, sr = librosa.load(row.path)
            # print('\nTrack: {} - Audio: {}, SampleRate: {}'.format(i, row.path, sr))

            row['tempo'] = np.mean(librosa.beat.tempo(audio, sr=sr))
            row['energy'] = np.mean(features_functions.energy(audio))
            row['energy_entropy'] = np.mean(features_functions.energy_entropy(audio, n_short_blocks=5))
            row['rmse'] = np.mean(librosa.feature.rms(audio))
            row['chroma_stft'] = np.mean(librosa.feature.chroma_stft(audio, sr=sr))
            row['chroma_cqt'] = np.mean(librosa.feature.chroma_cqt(audio, sr=sr))
            row['chroma_cens'] = np.mean(librosa.feature.chroma_cens(audio, sr=sr))
            row['spec_cent'] = np.mean(librosa.feature.spectral_centroid(audio, sr=sr))
            row['spec_bw'] = np.mean(librosa.feature.spectral_bandwidth(audio, sr=sr))
            row['spec_contrast'] = np.mean(librosa.feature.spectral_contrast(audio, sr=sr))
            row['rolloff'] = np.mean(librosa.feature.spectral_rolloff(audio, sr=sr))
            row['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))

            for x, j in zip(librosa.feature.mfcc(audio, sr=sr, n_mfcc=num_mfcc)[:num_mfcc], range(num_mfcc)):
                row['mfcc_{}'.format(j)] = np.mean(x)

        num_files = '5000' if dataset_path == 'genres_5' else '10000'
        num_genres = '5GEN' if dataset_path == 'genres_5' else '10GEN'

        FILE_NAME = str(num_mfcc) + 'MFCC_' + num_files + '_' + num_genres + '_GTZAN.csv'
        SAVE_ROOT = 'test_model/results/' + str(dataset_path) + '/'

        data.head()
