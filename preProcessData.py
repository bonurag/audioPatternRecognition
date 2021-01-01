import os
import scipy
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import resample
from scipy.io import wavfile
from pydub.utils import make_chunks
import shutil
import math

dataset_path = 'test'
FILE_EXTENSION = '.wav'


def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=22050):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform


exclude = {"old"}

for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path, topdown=True)):
    dirname[:] = [d for d in dirname if d not in exclude]

    # ensure that we're not at the root level
    if dirpath is not dataset_path:
        print('dirpath: ', dirpath)

        # process files for a specific genre
        dirpath_component = dirpath.split('/')  # return a list of element
        print('dirpath_component: ', dirpath_component)
        dirpath_root = dirpath_component[-1].split('\\')[0]
        print('dirpath_root: ', dirpath_root)
        semantic_label = dirpath_component[-1].split('\\')[1]
        print('semantic_label: ', semantic_label + FILE_EXTENSION)
        counter = 0

        for f in filenames:  # filenames represent only the file name and not the full path

            print("file: ", f)
            print("filenames len: ", len(filenames))

            partFileName = '000'
            if len(filenames) > 10:
                partFileName = '00'
            elif len(filenames) > 100:
                partFileName = '0'
            elif len(filenames) > 1000:
                partFileName = ''

            old_file_path = os.path.join(dirpath, f)
            if os.path.isfile(old_file_path):
                print("Check IF is File: ", os.path.isfile(old_file_path))
                new_file_name = str(semantic_label) + "." + partFileName + str(counter) + FILE_EXTENSION
                print("new_file_name: ", new_file_name)
                print("path and new name: ", dirpath + "/" + new_file_name)
                os.rename(old_file_path, dirpath + "/" + new_file_name)

                file_path = os.path.join(dirpath, new_file_name)
                processed_file = os.path.join(dirpath + "/old", new_file_name)
                file_component = file_path.split('/')  # return a list of element
                file_name = file_component[-1].split('\\')[2]
                print('file_path: ', file_path)
                print('file_component: ', file_component)
                print('file_name: ', file_name)
                print('processed_file: ', processed_file)

                counter = counter + 1

                if file_name.endswith(".wav"):
                    sound = AudioSegment.from_file(file_path, "wav")
                    print("Sound: ", len(sound))
                    chunk_length_ms = 30000  # pydub calculates in millisecond
                    chunks = make_chunks(sound, chunk_length_ms)  # Make chunks of one sec
                    print("chunks", len(chunks))
                    if math.fmod(len(sound), chunk_length_ms) != 0:
                        chunks = chunks[:len(chunks) - 1]
                    # Export all of the individual chunks as wav files

                    for j, chunk in enumerate(chunks):
                        chunk_name = file_name[:-4] + "_chunk{0}.wav".format(j)
                        print("exporting", chunk_name)
                        output_path = str(dirpath_root) + '/' + str(semantic_label) + '/' + chunk_name
                        chunk.export(output_path, format="wav")

                    print("Check IF Folder Exist: ", os.path.isdir(dirpath + "/old"))
                    if not os.path.exists(dirpath + "/old"):
                        os.makedirs(dirpath + "/old")
                        shutil.move(file_path, processed_file)
                    else:
                        shutil.move(file_path, processed_file)