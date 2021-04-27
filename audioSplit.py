import shutil

from pydub import AudioSegment
import os

DATASET_PATH = 'test_model'
EXCLUDE_FOLDER = {"source"}

if __name__ == "__main__":
    for dirpath, dirname, filenames in os.walk(DATASET_PATH):
        dirname[:] = [d for d in dirname if d not in EXCLUDE_FOLDER]

        if dirpath is not DATASET_PATH:
            dirpath_component = dirpath.split('/')  # return a list of element
            semantic_label = dirpath_component[-1].split('\\')[1]
            for f in filenames:  # filenames represent only the file name and not the full path
                file_name = os.path.join(dirpath, f)
                current_name = file_name.split('\\')[2]
                print('current_name: ', current_name)
                new_name = file_name.split('.')[0] + '.' + file_name.split('.')[1]
                # print('current_name: ' + str(file_name))
                # print('new_name : ' + str(new_name))
                t1 = 0
                index = 0
                for t2 in range(3000, 33000, 3000):
                    # print("T1: " + str(t1) + " -- T2: " + str(t2))
                    # print("Index: " + str(index))
                    newAudio = AudioSegment.from_wav(file_name)
                    newAudio = newAudio[t1:t2]
                    t1 = t2
                    newAudio.export(str(new_name) + '.' + str(index) + '.wav', format='wav')
                    print('New File Name: ' + str(new_name) + '.' + str(index) + '.wav')
                    new_file_name = str(new_name) + '.' + str(index) + '.wav'
                    file_component = new_file_name.split('\\')
                    chunked_files = file_component[0] + '/' + file_component[1] + '/sub_chunk_files' + '/' + file_component[2]
                    print('chunked_files: ' + chunked_files)
                    source_files = file_component[0] + '/' + file_component[1] + '/source_files' + '/' + current_name
                    print('source_files: ' + source_files)

                    if not os.path.exists(dirpath + '/sub_chunk_files'):
                        os.makedirs(dirpath + '/sub_chunk_files')
                        shutil.move(new_file_name, chunked_files)
                    else:
                        shutil.move(new_file_name, chunked_files)
                    index += 1
                if not os.path.exists(dirpath + '/source_files'):
                    os.makedirs(dirpath + '/source_files')
                    shutil.move(file_name, source_files)
                else:
                    shutil.move(file_name, source_files)