import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf

DATASET_PATH = 'genres'
SAMPLE_RATE = 22050
STRETCH_FACTOR = 0.8
NOISE_FACTOR = 0.009
SHIFT_FACTOR = int(SAMPLE_RATE / 10)
FILE_EXTENSION = '.wav'
LOG_ENABLE = False

class AudioAugmentation:
    def read_audio_file(self, file_path):
        signal, sr = sf.read(file_path)
        return signal, sr

    def write_audio_file(self, file, signal, sample_rate=SAMPLE_RATE):
        sf.write(file, signal, sample_rate)

    def plot_time_series(self, signal):
        fig = plt.figure(figsize=(14, 8))
        plt.title('Raw wave')
        plt.ylabel('Amplitude')
        plt.plot(np.linspace(0, 1, len(signal)), signal)
        plt.show()

    def add_noise(self, signal, factor=NOISE_FACTOR):
        noise = np.random.randn(len(signal))
        data_noise = signal + factor * noise
        return data_noise

    def shift(self, signal, factor=SHIFT_FACTOR):
        return np.roll(signal, factor)

    def stretch(self, signal, factor=STRETCH_FACTOR):
        signal = librosa.effects.time_stretch(signal, factor)
        return signal


def save_augment_data(dataset_path):
    # loop inside the folder that contains all data
    # dirpath represent the directory that contains all folder with genres
    # dirname represent the single folders per genres
    # filename represent the single audio sample in dirname
    # for count a number of iteration during the for i can insert enumerate and assign to i var
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):
        # ensure that we're not at the root level
        if dirpath is not dataset_path:
            if LOG_ENABLE:
                print('dirpath: ', dirpath)
                print('dirname: ', dirname)
                print('filenames: ', filenames)

            # save the semantic label in mapping label inside JSON
            dirpath_component = dirpath.split('/')  # return a list of element
            dirpath_root = dirpath_component[-1].split('\\')[0]
            semantic_label = dirpath_component[-1].split('\\')[1]
            if LOG_ENABLE:
                print('dirpath_root: ', dirpath_root)
                print('dirpath_component: ', dirpath_component)
                print('semantic_label: ', semantic_label)

            # process files for a specific genre
            for f in filenames:  # filenames represent only the file name and not the full path
                if LOG_ENABLE:
                    print('f: ', f)

                # load audio file
                file_path = os.path.join(dirpath, f)
                if LOG_ENABLE:
                    print('file_path: ', file_path)
                    print('file_path isfile: ', os.path.isfile(file_path))

                file_component = file_path.split('/')  # return a list of element
                file_name = file_component[-1].split('\\')[2][:-4]
                if LOG_ENABLE:
                    print('file_component: ', file_component)
                    print('file_name: ', file_name)

                # Create a new instance from AudioAugmentation class
                audioaug = AudioAugmentation()
                # Read and show cat sound
                data, sr = audioaug.read_audio_file(file_path)
                if LOG_ENABLE:
                    print("Song:{}, Data:{} - DataLen:{} - Sample Rate:{}".format(file_path, data, len(data), sr))
                # audioAug.plot_time_series(data)

                # Adding noise to sound
                data_noise = audioaug.add_noise(data)
                # audioAug.plot_time_series(data_noise)

                # Shifting the sound
                data_roll = audioaug.shift(data)
                # audioAug.plot_time_series(data_roll)

                # Stretching the sound
                data_stretch = audioaug.stretch(data)
                # audioaug.plot_time_series(data_stretch)

                # Write generated cat sounds
                output_path_noise = str(dirpath_root) + '/' + str(semantic_label) + '/' + file_name + '.Noise' + FILE_EXTENSION
                if LOG_ENABLE:
                    print('output_path_noise: ', output_path_noise)
                output_path_roll = str(dirpath_root) + '/' + str(semantic_label) + '/' + file_name + '.Roll' + FILE_EXTENSION
                if LOG_ENABLE:
                    print('output_path_roll: ', output_path_roll)
                output_path_stretch = str(dirpath_root) + '/' + str(semantic_label) + '/' + file_name + '.Stretch' + FILE_EXTENSION
                if LOG_ENABLE:
                    print('output_path_stretch: ', output_path_stretch)

                audioaug.write_audio_file(output_path_noise, data_noise)
                audioaug.write_audio_file(output_path_roll, data_roll)
                audioaug.write_audio_file(output_path_stretch, data_stretch)


if __name__ == "__main__":
    save_augment_data(DATASET_PATH)
