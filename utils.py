import os
import random
import shutil

import matplotlib.pyplot as plt

DATASET_PATH = 'genres_V2'
folder = []
fileSize = []


def get_file_folder(current_path):
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(current_path, topdown=True)):
        exclude_folder = {"exclude"}
        dirname[:] = [d for d in dirname if d not in exclude_folder]

        # ensure that we're not at the root level
        if dirpath is not current_path:
            # process files for a specific genre
            dirpath_component = dirpath.split('/')  # return a list of element
            # print('dirpath_component: ', dirpath_component)
            dirpath_root = dirpath_component[-1].split('\\')[0]
            # print('dirpath_root: ', dirpath_root)
            folder_name = dirpath_component[-1].split('\\')[1]
            # print('folder_name: ', folder_name)
            folder.append(folder_name)
            fileSize.append(len(filenames))

    # print("folder: ", folder)
    # print("fileSize: ", fileSize)
    return folder, fileSize


def autolabel(bars, ax):
    # attach some text labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 0.95, bar.get_y() + bar.get_height() / 2, '%d' % int(width), ha='right', va='center')


def get_horizontal_historgram(y_Value, x_Value, x_label, top_title, width_graph, height_graph):
    # Set the colors
    colors = ['royalblue']
    # Initialize the matplotlib figure
    fig, ax = plt.subplots(figsize=(width_graph, height_graph))
    a = ax.barh(y_Value, x_Value, align='center', color=colors)
    ax.set_yticks(y_Value)
    ax.set_yticklabels(y_Value)
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_title(top_title)
    autolabel(a, ax)
    plt.show()


def random_mover(current_path):
    list_files = []
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(current_path, topdown=True)):
        exclude_folder = {"exclude"}
        dirname[:] = [d for d in dirname if d not in exclude_folder]

        # ensure that we're not at the root level
        if dirpath is not current_path:

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                list_files.append(file_path)

            files_to_move = [list_files.pop(random.randrange(0, len(list_files))) for _ in range(300)]

            for j in range(len(files_to_move)):
                print("files_to_move {} - {}".format(j, files_to_move))
                move_file = files_to_move[j].split('/')
                print('move_file: ', move_file)
                dirpath_root = move_file[-1].split('\\')[0]
                print('dirpath_root: ', dirpath_root)
                sub_folder = move_file[-1].split('\\')[1]
                print('sub_folder: ', sub_folder)
                file_name = move_file[-1].split('\\')[2]
                print('file_name: ', file_name)
                print()
                current_file = str(dirpath_root) + '/' + str(sub_folder) + '/' + str(file_name)
                output_path = str(dirpath_root) + '/' + str(sub_folder) + '/source'
                print('From: ', current_file)
                print('To: ', output_path + '/' + file_name)
                print()

                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                    shutil.move(current_file, output_path + '/' + file_name)
                else:
                    shutil.move(current_file, output_path + '/' + file_name)


if __name__ == "__main__":
    X_LABEL = 'Number of Samples'
    GRAPH_TITLE = 'Genres Vs. Samples Distribution'
    numGenres, numSamples = get_file_folder(DATASET_PATH)
    get_horizontal_historgram(numGenres, numSamples, X_LABEL, GRAPH_TITLE, 10, 7)
    random_mover(DATASET_PATH)