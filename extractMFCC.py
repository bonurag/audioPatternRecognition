import os
import librosa
import librosa.display
import math
import json
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = "Single_Track"
NUM_MFCC = 64
FRAME_SIZE = 2048
HOP_SIZE = 512
NUM_SEGMENT = 5

SAMPLE_RATE = 22050
DURATION = 3  # measured in seconds
SAMPLE_PER_TRACK = SAMPLE_RATE * DURATION

MONO = True

JSON_PATH = "Single_Track/DataSet_" + str(NUM_MFCC) + "MFCC.json"


def save_mfcc(dataset_path, json_path, num_mfcc=NUM_MFCC, num_fft=FRAME_SIZE, hop_length=HOP_SIZE,
              num_segments=NUM_SEGMENT):
    # dictionary to store data
    data = {
        "mapping": [],  # list of musical genres
        "mfcc": [],  # input data extract from audio sample
        "labels": []  # target label
    }

    num_sample_per_segment = int(SAMPLE_PER_TRACK / num_segments)
    expected_mfcc_vectors_per_segments = math.ceil(num_sample_per_segment / hop_length)  # 1.2 -> 2 i would like to have an integer

    exclude_folder = {"exclude"}
    # loop inside the folder that contains all data
    # dirpath represent the directory that contains all folder with genres
    # dirname represent the single folders per genres
    # filename represent the single audio sample in dirname
    # for count a number of iteration during the for i can insert enumerate and assign to i var
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):
        dirname[:] = [d for d in dirname if d not in exclude_folder]

        # ensure that we're not at the root level
        if dirpath is not dataset_path:

            # save the semantic label in mapping label inside JSON
            dirpath_component = dirpath.split("/")  # return a list of element
            semantic_label = dirpath_component[-1].split("\\")[1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process files for a specific genre
            for f in filenames:  # filenames represent only the file name and not the full path

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=MONO)

                # process segments extracting mfcc and storing data
                for d in range(num_segments):
                    start_sample = num_sample_per_segment * d
                    finish_sample = start_sample + num_sample_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], sr=sample_rate, n_mfcc=num_mfcc, n_fft=num_fft, hop_length=hop_length)
                    # print("MFCCs Shape: {}".format(mfcc.shape))
                    # plt.figure(figsize=(25, 10))
                    # librosa.display.specshow(mfcc, x_axis="time", sr=sample_rate)
                    # plt.colorbar(format="%+2.f")
                    # plt.show()
                    mfcc = mfcc.T

                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_mfcc_vectors_per_segments:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, d + 1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)
