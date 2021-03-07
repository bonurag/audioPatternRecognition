from pydub import AudioSegment
import os

DATASET_PATH = 'genres'

if __name__ == "__main__":
    for dirpath, dirname, filenames in os.walk(DATASET_PATH):
        if dirpath is not DATASET_PATH:
            dirpath_component = dirpath.split('/')  # return a list of element
            semantic_label = dirpath_component[-1].split('\\')[1]
            for f in filenames:  # filenames represent only the file name and not the full path
                file_name = os.path.join(dirpath, f)
                current_name = file_name.split('\\')[2]
                new_name = file_name.split('.')[0] + "." + file_name.split('.')[1]
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
                    newAudio.export(str(new_name) + '.' + str(index) + '.wav', format="wav")
                    print("New File Name: " + str(new_name) + '.' + str(index) + '.wav')
                    index += 1
