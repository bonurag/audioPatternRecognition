import time
import librosa.display
import numpy as np
import os
from os import path
import pandas as pd
from tqdm import tqdm
import common_functions
import apr_constants
from utils import features_functions


# Create a csv file from samples
def extract_features(dataset_path=None, exclude_folders_name={}, save_root_path=None, mfcc_value=13,
                     sample_duration=30):
    start_time = time.time()
    if dataset_path is None or len(dataset_path) == 0:
        print('Attention, insert the path of the samples!')
    if save_root_path is None or len(save_root_path) == 0:
        print('Attention, insert the path where you want to save the extraction!')
    if dataset_path is not None and save_root_path is not None and len(dataset_path) > 0 and len(save_root_path) > 0:
        for dirpath, dirname, filenames in os.walk(dataset_path):
            dirname[:] = [d for d in dirname if d not in exclude_folders_name]
            if len(dirname) > 0:
                data = pd.DataFrame(columns=['path', 'genre', 'tempo', 'energy', 'energy_entropy', 'rmse',
                                             'chroma_stft', 'chroma_cqt', 'chroma_cens', 'spec_cent',
                                             'spec_bw', 'spec_contrast', 'rolloff', 'zcr'] +
                                            ['mfcc_{}'.format(x) for x in range(mfcc_value)])
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
                    if int(duration) == sample_duration:
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

                        for x, j in zip(librosa.feature.mfcc(audio, sr=sr, n_mfcc=mfcc_value)[:mfcc_value],
                                        range(mfcc_value)):
                            row['mfcc_{}'.format(j)] = np.mean(x)

                file_name = str(mfcc_value) + 'MFCC_' + str(i + 1) + '_' + str(len(dirname)) + '.csv'
                common_functions.check_create_directory(save_root_path)

                del data['path']  # Remove path column
                data = data.dropna(axis=0, subset=['tempo'])  # Remove empty line (duration is below sample_duration)

        if not path.isfile(save_root_path + file_name):
            data.to_csv(save_root_path + file_name, index=False)
            execution_time = time.time() - start_time
            print('CSV Saved!')
            print('Features Extractions Completed!')
            print('Execution Time: ', execution_time)
        else:
            check_overwrite = input('The file that you want to, do you want to overwrite it? [Y/N]: ')
            if check_overwrite.upper() == 'Y':
                data.to_csv(save_root_path + file_name, index=False)
                execution_time = time.time() - start_time
                print('CSV Saved!')
                print('Features Extractions Completed!')
                print('Execution Time: ', execution_time)
            elif check_overwrite.upper() == 'N':
                execution_time = time.time() - start_time
                print('Execution Time: ', execution_time)
    return file_name

if __name__ == "__main__":
    extract_features(apr_constants.FEATURES_DATASET_PATH,
                     apr_constants.FEATURES_EXCLUDE_FOLDER,
                     apr_constants.FEATURES_SAVE_ROOT,
                     apr_constants.FEATURES_MFCC_VALUE,
                     apr_constants.FEATURES_SAMPLE_DURATION)
