import time
import json

import pandas as pd
import numpy as np

from sklearn import preprocessing

from sklearn.cluster import KMeans
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt

from itertools import cycle

import apr_constants
import common_functions


def plot_correlation_matrix(correlation_matrix, save_plot=True, file_name=apr_constants.DEFAULT_FILE_NAME,
                            save_path=apr_constants.PROJECT_ROOT):
    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1.4)
    sns.heatmap(correlation_matrix, cmap='coolwarm', square=True)
    plt.title('Correlation between different features', fontsize=apr_constants.TITLE_FONT_SIZE)
    if save_plot:
        print('Save Correlation Matrix')
        common_functions.check_create_directory(save_path)
        plt.savefig(save_path + file_name + ' - ' + 'Correlation Matrix.jpg')
    plt.show()


def plot_pca(input_pca_data, save_plot=False, target_names=None, file_name=apr_constants.DEFAULT_FILE_NAME,
             save_path=apr_constants.PROJECT_ROOT):
    if target_names is None:
        target_names = []
    plt.figure(figsize=(20, 10))
    new_data = input_pca_data.copy()

    genres = {i: target_names[i] for i in range(0, len(target_names))}
    new_data.genre = [genres[int(item)] for item in new_data.genre]

    sns.scatterplot(x='PC1', y='PC2', data=new_data, hue='genre', alpha=0.6, palette='deep')

    plt.title('PCA on Genres', fontsize=apr_constants.TITLE_FONT_SIZE)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=10)
    plt.xlabel('Principal Component 1', fontsize=22)
    plt.ylabel('Principal Component 2', fontsize=22)

    if save_plot:
        print('Save PCA Plot')
        common_functions.check_create_directory(save_path)
        plt.savefig(save_path + file_name + ' - ' + 'PCA Scatter Plot.jpg')
    plt.show()


def plot_3d_pca(input_pca_data, save_plot=True, file_name=apr_constants.DEFAULT_FILE_NAME,
                save_path=apr_constants.PROJECT_ROOT):
    # initialize figure and 3d projection for the PC3 data
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # assign x,y,z coordinates from PC1, PC2 & PC3
    xs = input_pca_data['PC1']
    ys = input_pca_data['PC2']
    zs = input_pca_data['PC3']

    # initialize scatter plot and label axes
    plot = ax.scatter(xs, ys, zs, alpha=0.6, c=input_pca_data['genre'], cmap='magma', depthshade=True)

    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='z', labelsize=15)
    ax.set_xlabel('PC1', labelpad=25)
    ax.set_ylabel('PC2', labelpad=25)
    ax.set_zlabel('PC3', labelpad=15)

    fig.colorbar(plot, shrink=0.6, aspect=9)
    if save_plot:
        print('Save 3D PCA Plot')
        common_functions.check_create_directory(save_path)
        plt.savefig(save_path + file_name + ' - ' + 'PCA 3D Scattert Plot.jpg')
    plt.show()


def plot_clusters(input_pca_data, centroids_value=None, labels=None, colors_list=None, genres_list=None, save_plot=True,
                  plot_centroids=True, file_name=apr_constants.DEFAULT_FILE_NAME, save_path=apr_constants.PROJECT_ROOT):
    if genres_list is None:
        genres_list = []
    if colors_list is None:
        colors_list = []
    if labels is None:
        labels = []
    if centroids_value is None:
        centroids_value = []
    pca_1, pca_2 = input_pca_data['PC1'], input_pca_data['PC2']
    centroids_x = centroids_value[:, 0]
    centroids_y = centroids_value[:, 1]

    colors = {v: k for v, k in enumerate(colors_list)}
    genres = {v: k for v, k in enumerate(genres_list)}

    df = pd.DataFrame({'pca_1': pca_1, 'pca_2': pca_2, 'label': labels, 'genre': input_pca_data['genre']})
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(20, 13))

    for genre, group in groups:
        if plot_centroids:
            plt.scatter(centroids_x, centroids_y, c='black', s=200, marker='x')
        ax.plot(group.pca_1, group.pca_2, marker='o', linestyle='', ms=6, color=colors[genre], label=genres[genre],
                mec='none', alpha=0.2)
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

    ax.legend()
    ax.set_title("Genres Music Clusters Results", fontsize=apr_constants.TITLE_FONT_SIZE)
    if save_plot:
        print('Save Clusters Plot')
        common_functions.check_create_directory(save_path)
        plt.savefig(save_path + file_name + ' - ' + 'Clusters Plot.jpg')
    plt.show()


def plot_confusion_matrix_k_means(input_data, save_plot=True, labels=None, target_names=None,
                                  file_name=apr_constants.DEFAULT_FILE_NAME, save_path=apr_constants.PROJECT_ROOT):
    if target_names is None:
        target_names = []
    if labels is None:
        labels = []
    input_data['predicted_label'] = labels
    data = metrics.confusion_matrix(input_data['genre'], input_data['predicted_label'])

    df_cm = pd.DataFrame(data, columns=np.unique(target_names), index=np.unique(target_names))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.4)
    heatmap = sns.heatmap(df_cm, cmap="Blues", annot=True, fmt='g', annot_kws={"size": 8}, square=True)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45)
    plt.title('CM for K-Means', fontsize=apr_constants.TITLE_FONT_SIZE)
    if save_plot:
        print('Save K-means Confusion Matrix')
        common_functions.check_create_directory(save_path)
        plt.savefig(save_path + file_name + ' - ' + 'K-means Confusion Matrix Plot.jpg')
    plt.show()


def plot_silhouette(input_data, min_clusters=2, max_clutsers=5, save_plot=False,
                    file_name=apr_constants.DEFAULT_FILE_NAME,
                    save_path=apr_constants.PROJECT_ROOT):
    eval_data = input_data.copy()
    silhouette_score_values = list()
    executiontime_values = list()

    number_of_clusters = range(min_clusters, max_clutsers + 1)
    for i in number_of_clusters:
        start_time = time.time()
        clusters = KMeans(i)
        clusters.fit(eval_data)
        cluster_labels = clusters.predict(eval_data)
        execution_time = time.time() - start_time
        executiontime_values.append(execution_time)
        silhouette_score_values.append(
            metrics.silhouette_score(eval_data, cluster_labels, metric='euclidean', sample_size=None,
                                     random_state=None))

    fig, ax1 = plt.subplots(figsize=(15, 10))
    y_ax_ticks = np.arange(0, max(silhouette_score_values) + 1, 0.1)
    x_ax_ticks = np.arange(min_clusters, max_clutsers + 1, 1)

    ax1.plot(number_of_clusters, silhouette_score_values)
    ax1.plot(number_of_clusters, silhouette_score_values, 'bo')
    ax1.set_title("Silhouette Score Values vs Numbers of Clusters", fontsize=22)
    ax1.set_yticks(y_ax_ticks)
    ax1.set_ylabel('Silhouette Score', fontsize=22)
    ax1.set_xticks(x_ax_ticks)
    ax1.set_xlabel('Number Of Clusters', fontsize=22)
    ax1.grid(False)

    ax2 = ax1.twinx()
    y_ax2_ticks = np.arange(0, max(executiontime_values) + 1, 0.3)
    ax2.plot(number_of_clusters, executiontime_values, 'orange', linestyle='dashed')
    ax2.plot(number_of_clusters, executiontime_values, 'orange', marker="o", linestyle='dashed')
    ax2.set_yticks(y_ax2_ticks)
    ax2.set_ylabel('Fit Time (sec)', fontsize=22)
    ax2.grid(False)

    optimal_number_of_components = number_of_clusters[silhouette_score_values.index(max(silhouette_score_values))]
    worst_number_of_components = number_of_clusters[silhouette_score_values.index(min(silhouette_score_values))]

    # optimal_execution_time = number_of_clusters[executiontime_values.index(max(executiontime_values))]
    # worst_execution_time = number_of_clusters[executiontime_values.index(min(executiontime_values))]

    plt.rcParams.update({'font.size': 12})
    for i in silhouette_score_values:
        clr = 'black'
        wgt = 'normal'
        value_position = silhouette_score_values.index(i)
        sil_y_offset_value = 0.006
        if max(silhouette_score_values) == i:
            wgt = 'bold'
            clr = 'green'
            ax1.annotate(str(round(i, 3)), xy=(value_position + 2, i + sil_y_offset_value), color=clr, weight=wgt)
        elif min(silhouette_score_values) == i:
            wgt = 'bold'
            clr = 'red'
            ax1.annotate(str(round(i, 3)), xy=(value_position + 2, i + sil_y_offset_value), color=clr, weight=wgt)
        else:
            ax1.annotate(str(round(i, 3)), xy=(value_position + 2, i + sil_y_offset_value), color=clr, weight=wgt)

    for j in executiontime_values:
        clr = 'black'
        wgt = 'normal'
        value_time_position = executiontime_values.index(j)
        time_y_offset_value = 0.06
        if max(executiontime_values) == j:
            wgt = 'bold'
            clr = 'red'
            ax2.annotate(str(round(j, 3)), xy=(value_time_position + 2, j - time_y_offset_value), color=clr, weight=wgt)
        elif min(executiontime_values) == j:
            wgt = 'bold'
            clr = 'green'
            ax2.annotate(str(round(j, 3)), xy=(value_time_position + 2, j - time_y_offset_value), color=clr, weight=wgt)
        else:
            ax2.annotate(str(round(j, 3)), xy=(value_time_position + 2, j - time_y_offset_value), color=clr, weight=wgt)

    ax1.vlines(x=optimal_number_of_components, ymin=0, ymax=max(silhouette_score_values), linewidth=2, color='green',
               label='Max Value', linestyle='dashdot')
    ax1.vlines(x=worst_number_of_components, ymin=0, ymax=min(silhouette_score_values), linewidth=2, color='red',
               label='min Value', linestyle='dashdot')

    # Adding legend
    ax1.legend(loc='upper center', prop={'size': apr_constants.LEGEND_SIZE})
    ax2.legend(['ExecutionTime'], loc='upper right', prop={'size': apr_constants.LEGEND_SIZE})

    if save_plot:
        print('Save Silhouette Plot')
        common_functions.check_create_directory(save_path)
        plt.savefig(save_path + file_name + ' - ' + 'Clusters Silhouette Plot.jpg')
    plt.show()
    # print("Optimal number of components is:", Optimal_NumberOf_Components)


def plot_roc(y_test, y_score, classifier_name=apr_constants.DEFAULT_CLASSIFIER_NAME, save_plot=False, target_names=None,
             file_name=apr_constants.DEFAULT_FILE_NAME, save_path=apr_constants.PROJECT_ROOT, type_learning='SL'):
    if target_names is None:
        target_names = []

    genres = target_names

    ordinal_position = []
    for index in range(0, len(target_names)):
        ordinal_position.append(index)

    test_label = preprocessing.label_binarize(y_test, classes=ordinal_position)
    if type_learning == 'SL':
        y_label = y_score
    elif type_learning == 'UL':
        y_label = preprocessing.label_binarize(y_score, classes=ordinal_position)

    n_classes = test_label.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(test_label[:, i], y_label[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    colors = cycle(apr_constants.ROC_COLOR_LIST)
    plt.figure(figsize=(15, 10))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(genres[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=24)
    plt.ylabel('True Positive Rate (TPR)', fontsize=24)
    plt.title('Receiver operating characteristic for ' + classifier_name.replace('_', ' ').upper(),
              fontsize=apr_constants.TITLE_FONT_SIZE)
    plt.legend(loc='lower right', prop={'size': apr_constants.LEGEND_SIZE})

    if save_plot:
        print('Save ROC Plot')
        common_functions.check_create_directory(save_path)
        plt.savefig(save_path + file_name + ' - ' + 'ROC Plot.jpg')
    plt.show()


def plot_classification_report(input_data, export_json=True, labels=None, target_names=None,
                               file_name=apr_constants.DEFAULT_FILE_NAME,
                               save_path=apr_constants.PROJECT_ROOT):
    if target_names is None:
        target_names = []
    if labels is None:
        labels = []
    input_data['predicted_label'] = labels
    report_dic = False
    if export_json:
        report_dic = True
    report = metrics.classification_report(input_data['genre'], input_data['predicted_label'],
                                           target_names=target_names,
                                           output_dict=report_dic)
    # print('results_reports: ', report)

    if export_json:
        common_functions.check_create_directory(save_path + apr_constants.DATA)
        with open(save_path + apr_constants.DATA + file_name + '_report_results.json', 'w') as res:
            json.dump(report, res, indent=4)
    plt.show()


def plot_bmp_bar(input_data, save_plot=False, target_names=None, file_name=apr_constants.DEFAULT_FILE_NAME,
                 save_path=apr_constants.PROJECT_ROOT):
    if target_names is None:
        target_names = []
    plt.figure(figsize=(15, 7))
    new_data = input_data[['genre', 'tempo']].copy()

    genres = {i: target_names[i] for i in range(0, len(target_names))}
    new_data.genre = [genres[item] for item in new_data.genre]

    sns.boxplot(x=new_data.genre, y=new_data.tempo, palette='husl')
    plt.title('BPM Boxplot for Genres', fontsize=apr_constants.TITLE_FONT_SIZE)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=10)
    plt.xlabel('Genre', fontsize=22)
    plt.ylabel('BPM', fontsize=22)

    if save_plot:
        print('Save BPM Bar')
        common_functions.check_create_directory(save_path)
        plt.savefig(save_path + file_name + ' - ' + 'BPM BoxPlot.jpg')
    plt.show()


def plot_confusion_matrix(clf, x_test, y_test, classes, normalize='true',
                          classifier_name=apr_constants.DEFAULT_CLASSIFIER_NAME, save_plot=False,
                          file_name=apr_constants.DEFAULT_FILE_NAME, save_path=apr_constants.PROJECT_ROOT):
    fig, ax = plt.subplots(figsize=(10, 10))
    metrics.plot_confusion_matrix(clf, x_test, y_test, normalize=normalize, cmap=plt.cm.Blues, ax=ax,
                                  display_labels=classes, values_format='.0f')
    ax.set_title('CM for ' + classifier_name.replace('_', ' ').upper(), fontsize=apr_constants.TITLE_FONT_SIZE)
    plt.xticks(rotation=45)
    plt.grid(False)
    if save_plot:
        print('Save Confusion Matrix Plot')
        common_functions.check_create_directory(save_path)
        plt.savefig(save_path + file_name + ' - ' + 'Confusion Matrix Plot.jpg')
    plt.show()


def plot_predictions_compare(normalize_cm, y_test, y_pred, target_names=None,
                             classifier_name=apr_constants.DEFAULT_CLASSIFIER_NAME,
                             save_plot=False, file_name=apr_constants.DEFAULT_FILE_NAME,
                             save_path=apr_constants.PROJECT_ROOT):
    if target_names is None:
        target_names = []
    genres = target_names
    calc_cm = metrics.confusion_matrix(y_test, y_pred, normalize=normalize_cm)
    bar = pd.DataFrame(calc_cm, columns=genres, index=genres)
    ax = bar.plot(kind='bar', figsize=(15, 15), fontsize=14, width=0.8)
    plt.title('Musical Genres BarPlot Predictions For ' + classifier_name.replace('_', ' ').upper(),
              fontsize=apr_constants.TITLE_FONT_SIZE)
    plt.xlabel('Musical Genres', fontsize=22)
    plt.xticks(rotation=45)
    plt.ylabel('Number of Occurrences', fontsize=22)

    for p in ax.patches:
        if p.get_height() > 0:
            if normalize_cm != 'true':
                ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center',
                            va='center', size=15, xytext=(0, 8), textcoords='offset points', fontsize=8)
            else:
                ax.annotate(format(p.get_height(), '.3f') + '%',
                            (p.get_x() + (p.get_width() / 2) + 0.015, p.get_height() + 0.01), ha='center', va='center',
                            size=15, xytext=(0, 8), textcoords='offset points', fontsize=8, rotation=90)

    if save_plot:
        print('Save Model Predictions Compare')
        common_functions.check_create_directory(save_path)
        plt.savefig(save_path + file_name + ' - ' + 'Predictions Compare Plot.jpg')
    plt.show()


def plot_predictions_simple_compare(input_data, target_names=None,
                                    save_plot=False, file_name=apr_constants.DEFAULT_FILE_NAME,
                                    save_path=apr_constants.PROJECT_ROOT):
    if target_names is None:
        target_names = []
    ax = input_data.plot(kind="bar", figsize=(15, 15), fontsize=14, width=0.6)
    ax.set_xticklabels(target_names)
    ax.legend(["Real Value", "Predict Value"])
    plt.title('Simple BarPlot Predictions Evaluation', fontsize=26)
    plt.xlabel('Musical Genres', fontsize=18)
    plt.ylabel('Number of Occurrences', fontsize=18)
    for p in ax.patches:
        ax.annotate(format(p.get_height()),
                    (p.get_x() + (p.get_width() / 2) + 0.015, p.get_height() + 5), ha='center', va='center',
                    size=18, xytext=(0, 8), textcoords='offset points', fontsize=14, rotation=90)
    if save_plot:
        print('Save Simple Model Predictions Compare')
        common_functions.check_create_directory(save_path)
        plt.savefig(save_path + file_name + ' - ' + 'Simple Predictions Compare Plot.jpg')
    plt.show()
