import os
import librosa
import math
import json

DATASET_PATH = "genres"
JSON_PATH = "features_json/DataSet_13MFCC.json"
SAMPLE_RATE = 22050
DURATION = 30  # measured in seconds
SAMPLE_PER_TRACK = SAMPLE_RATE * DURATION
MONO = True


def save_mfcc(dataset_path, json_path, num_mfcc=13, num_fft=2048, hop_length=512, num_segments=5):
    # dictionary to store data
    data = {
        "mapping": [],  # list of musical genrse
        "mfcc": [],  # input data extract from audio sample
        "labels": []  # target label
    }

    num_sample_per_segment = int(SAMPLE_PER_TRACK / num_segments)
    expected_mfcc_vectors_per_segments = math.ceil(num_sample_per_segment / hop_length)  # 1.2 -> 2 i would like to have an integer

    # loop inside the folder that contains all data
    # dirpath represent the directory that contains all folder with genres
    # dirname represent the single folders per genres
    # filename represent the single audio sample in dirname
    # for count a number of iteration during the for i can insert enumerate and assign to i var
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):
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

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], sample_rate, n_mfcc=num_mfcc, n_fft=num_fft, hop_length=hop_length)
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
