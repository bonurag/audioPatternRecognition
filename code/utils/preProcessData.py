import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import shutil
import math

DATASET_PATH = 'test_model'
FILE_EXTENSION = 'wav'
CHUNK_LENGTH = 30000
EXCLUDE_FOLDER = {'source', 'source_files', 'sub_chunk_files'}

if __name__ == "__main__":
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(DATASET_PATH, topdown=True)):
        dirname[:] = [d for d in dirname if d not in EXCLUDE_FOLDER]

        # ensure that we're not at the root level
        if dirpath is not DATASET_PATH:
            print('dirpath: ', dirpath)

            # process files for a specific genre
            dirpath_component = dirpath.split('/')  # return a list of element
            print('dirpath_component: ', dirpath_component)
            dirpath_root = dirpath_component[-1].split('\\')[0]
            print('dirpath_root: ', dirpath_root)
            semantic_label = dirpath_component[-1].split('\\')[1]
            print('semantic_label: ', semantic_label + "." + FILE_EXTENSION)
            counter = 0

            for f in filenames:  # filenames represent only the file name and not the full path

                print("file: ", f)
                print("filenames len: ", len(filenames))

                partFileName = '0000'
                if len(filenames) > 10:
                    partFileName = '000'
                elif len(filenames) > 100:
                    partFileName = '00'
                elif len(filenames) > 1000:
                    partFileName = '0'
                elif len(filenames) > 10000:
                    partFileName = ''

                old_file_path = os.path.join(dirpath, f)
                if os.path.isfile(old_file_path):
                    print("Check IF is File: ", os.path.isfile(old_file_path))
                    new_file_name = str(semantic_label) + "." + partFileName + str(counter) + "." + FILE_EXTENSION
                    print("new_file_name: ", new_file_name)
                    print("path and new name: ", dirpath + "/" + new_file_name)
                    os.rename(old_file_path, dirpath + "/" + new_file_name)

                    file_path = os.path.join(dirpath, new_file_name)
                    processed_file = os.path.join(dirpath + "/source", new_file_name)
                    file_component = file_path.split('/')  # return a list of element
                    file_name = file_component[-1].split('\\')[2]
                    print('file_path: ', file_path)
                    print('file_component: ', file_component)
                    print('file_name: ', file_name)
                    print('processed_file: ', processed_file)

                    counter = counter + 1

                    if file_name.endswith("wav"):
                        sound = AudioSegment.from_file(file_path, "wav")
                    elif file_name.endswith("mp3"):
                        sound = AudioSegment.from_file(file_path, "mp3")
                    print("Sound Data Length: ", len(sound))
                    chunk_length_ms = CHUNK_LENGTH  # pydub calculates in millisecond
                    chunks = make_chunks(sound, chunk_length_ms)  # Make chunks of one sec
                    print("Calculate Chunks: ", len(chunks))
                    if math.fmod(len(sound), chunk_length_ms) != 0:
                        chunks = chunks[:len(chunks) - 1]
                        print("Real Chunks Generated: ", len(chunks))
                    # Export all of the individual chunks as wav files

                    for j, chunk in enumerate(chunks):
                        chunk_name = file_name[:-4] + "_chunk{0}.".format(j) + FILE_EXTENSION
                        print("exporting", chunk_name)
                        output_path = str(dirpath_root) + '/' + str(semantic_label) + '/' + chunk_name
                        chunk.export(output_path, format=FILE_EXTENSION)

                    print("Check IF Folder Exist: ", os.path.isdir(dirpath + "/source"))
                    if not os.path.exists(dirpath + "/source"):
                        os.makedirs(dirpath + "/source")
                        shutil.move(file_path, processed_file)
                    else:
                        shutil.move(file_path, processed_file)
