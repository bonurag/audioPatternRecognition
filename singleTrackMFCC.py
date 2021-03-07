import os
import librosa
import math
import json

DATASET_PATH = "SingleTrack/data/"
JSON_PATH = "SingleTrack/songs_feature/MFCC64/"
SAMPLE_RATE = 22050
DURATION = 3  # measured in seconds
SAMPLE_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc_single_track(dataset_path, json_path, num_mfcc=64, num_fft=2048, hop_length=512, num_segments=5):
    num_sample_per_segment = int(SAMPLE_PER_TRACK / num_segments)
    expected_mfcc_vectors_per_segments = math.ceil(num_sample_per_segment / hop_length)  # 1.2 -> 2 i would like to have an integer

    for (dirpath, dirname, filenames) in os.walk(dataset_path):
        # process files for a specific genre
        for f in filenames:  # filenames represent only the file name and not the full path
            # load audio file
            file_path = os.path.join(dirpath, f)
            print("file_path: ", file_path)
            signal, sample_rate = librosa.load(file_path, offset=60.0, duration=3.0)

            S = librosa.stft(signal)
            duration = librosa.get_duration(S=S, sr=sample_rate)

            print('Filename: {}'.format(f))
            print('Duration: {:.2f} s'.format(duration))
            filename = f.split(".")[0] + str(".json")

            # dictionary to store data
            data = {
                "mfcc": [],  # input data extract from audio sample
            }

            # process segments extracting mfcc and storing data
            for d in range(num_segments):
                start_sample = num_sample_per_segment * d
                finish_sample = start_sample + num_sample_per_segment

                mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], sample_rate, n_mfcc=num_mfcc, n_fft=num_fft, hop_length=hop_length)
                mfcc = mfcc.T

                # store mfcc for segment if it has the expected length
                if len(mfcc) == expected_mfcc_vectors_per_segments:
                    data["mfcc"].append(mfcc.tolist())

            # save MFCCs to json file
            with open(json_path+filename, "w") as fp:
                json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc_single_track(DATASET_PATH, JSON_PATH, num_segments=10)
