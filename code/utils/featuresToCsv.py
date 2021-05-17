import time
import librosa.display
import numpy as np
import os
from os import path
import pandas as pd
from tqdm import tqdm
import features_functions

# Create a csv file from data
DATASET_PATH = 'genres_collections_new'
EXCLUDE_FOLDER = {'results'}
SAVE_ROOT = DATASET_PATH + '/results/'
MFCC_VALUE = 13
SAMPLE_DURATION = 30

if __name__ == "__main__":
    start_time = time.time()

    for dirpath, dirname, filenames in os.walk(DATASET_PATH):
        dirname[:] = [d for d in dirname if d not in EXCLUDE_FOLDER]
        if len(dirname) > 0:
            data = pd.DataFrame(columns=['path', 'genre', 'tempo', 'energy', 'energy_entropy', 'rmse',
                                         'chroma_stft', 'chroma_cqt', 'chroma_cens', 'spec_cent',
                                         'spec_bw', 'spec_contrast', 'rolloff', 'zcr'] +
                                        ['mfcc_{}'.format(x) for x in range(MFCC_VALUE)])
            genres = []
            paths = []
            for gnr in dirname:
                for file in os.listdir(os.path.join(dirpath, gnr)):
                    genres.append(gnr)
                    paths.append(os.path.join(dirpath, gnr, file))

            data['path'] = paths
            data['genre'] = genres

            num_iterations = len(filenames)
            for i, row in tqdm(data.iterrows(), total=num_iterations):
                audio, sr = librosa.load(row.path)
                duration = librosa.get_duration(y=audio, sr=sr)
                if int(duration) == SAMPLE_DURATION:
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

                    for x, j in zip(librosa.feature.mfcc(audio, sr=sr, n_mfcc=MFCC_VALUE)[:MFCC_VALUE], range(MFCC_VALUE)):
                        row['mfcc_{}'.format(j)] = np.mean(x)

            FILE_NAME = str(MFCC_VALUE) + 'MFCC_' + str(i+1) + '_' + str(len(dirname)) + '.csv'
            if not os.path.exists(SAVE_ROOT):
                os.makedirs(SAVE_ROOT)

        if not path.isfile(SAVE_ROOT + FILE_NAME):
            del data['path']
            data = data.dropna(axis=0, subset=['tempo'])
            data.to_csv(SAVE_ROOT + FILE_NAME, index=False)
            executionTime = time.time() - start_time
            print('Save CSV!')
            print('Features Extractions Completed!')
            print('Execution Time: ', executionTime)
