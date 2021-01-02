import os
import matplotlib.pyplot as plt

DATASET_PATH = 'genres_V2'
excludeFolder = {"source"}
folder = []
fileSize = []


def get_file_folder(currentPath):
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(currentPath, topdown=True)):
        dirname[:] = [d for d in dirname if d not in excludeFolder]

        # ensure that we're not at the root level
        if dirpath is not currentPath:
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


if __name__ == "__main__":
    X_LABEL = 'Number of Samples'
    GRAPH_TITLE = 'Genres Vs. Samples Distribution'
    numGenres, numSamples = get_file_folder(DATASET_PATH)
    get_horizontal_historgram(numGenres, numSamples, X_LABEL, GRAPH_TITLE, 10, 7)
