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


def plot_correlation_matrix(correlation_matrix, savePlot=True, fileName=apr_constants.DEFAULT_FILE_NAME,
                            savePath=apr_constants.PROJECT_ROOT):
    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1.4)
    sns.heatmap(correlation_matrix, cmap='coolwarm', square=True)
    plt.title('Correlation between different features', fontsize=apr_constants.TITLE_FONT_SIZE)
    if savePlot:
        print('Save Correlation Matrix')
        common_functions.check_create_directory(savePath)
        plt.savefig(savePath + fileName + ' - ' + 'Correlation Matrix.jpg')
    plt.show()


def plot_PCA(inputPCAData, savePlot=False, target_names=[], fileName=apr_constants.DEFAULT_FILE_NAME,
             savePath=apr_constants.PROJECT_ROOT):
    plt.figure(figsize=(20, 10))
    new_data = inputPCAData.copy()
    genres = target_names
    # if len(target_names) == 10:
    #     genres = {i: target_names[i] for i in range(0, len(target_names))}
    # elif len(target_names) == 5:
    #     genres = {i: target_names[i] for i in range(0, len(target_names))}
    genres = {i: target_names[i] for i in range(0, len(target_names))}
    new_data.genre = [genres[int(item)] for item in new_data.genre]

    sns.scatterplot(x='PC1', y='PC2', data=new_data, hue='genre', alpha=0.6, palette='deep')

    plt.title('PCA on Genres', fontsize=apr_constants.TITLE_FONT_SIZE)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=10)
    plt.xlabel('Principal Component 1', fontsize=22)
    plt.ylabel('Principal Component 2', fontsize=22)

    if savePlot:
        print('Save PCA Plot')
        common_functions.check_create_directory(savePath)
        plt.savefig(savePath + fileName + ' - ' + 'PCA Scatter Plot.jpg')
    plt.show()


def plot_3D_PCA(inputPCAData, savePlot=True, fileName=apr_constants.DEFAULT_FILE_NAME,
                savePath=apr_constants.PROJECT_ROOT):
    # initialize figure and 3d projection for the PC3 data
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # assign x,y,z coordinates from PC1, PC2 & PC3
    xs = inputPCAData['PC1']
    ys = inputPCAData['PC2']
    zs = inputPCAData['PC3']

    # initialize scatter plot and label axes
    plot = ax.scatter(xs, ys, zs, alpha=0.6, c=inputPCAData['genre'], cmap='magma', depthshade=True)

    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='z', labelsize=15)
    ax.set_xlabel('PC1', labelpad=25)
    ax.set_ylabel('PC2', labelpad=25)
    ax.set_zlabel('PC3', labelpad=15)

    fig.colorbar(plot, shrink=0.6, aspect=9)
    if savePlot:
        print('Save 3D PCA Plot')
        common_functions.check_create_directory(savePath)
        plt.savefig(savePath + fileName + ' - ' + 'PCA 3D Scattert Plot.jpg')
    plt.show()


def plot_Clusters(inputPCAData, centroidsValue=[], labels=[], colors_list=[], genres_list=[], savePlot=True,
                  plotCentroids=True, fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    pca_1, pca_2 = inputPCAData['PC1'], inputPCAData['PC2']
    centroids_x = centroidsValue[:, 0]
    centroids_y = centroidsValue[:, 1]

    colors = {v: k for v, k in enumerate(colors_list)}
    genres = {v: k for v, k in enumerate(genres_list)}

    df = pd.DataFrame({'pca_1': pca_1, 'pca_2': pca_2, 'label': labels, 'genre': inputPCAData['genre']})
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(20, 13))

    for genre, group in groups:
        if plotCentroids:
            plt.scatter(centroids_x, centroids_y, c='black', s=200, marker=('x'))
        ax.plot(group.pca_1, group.pca_2, marker='o', linestyle='', ms=6, color=colors[genre], label=genres[genre],
                mec='none', alpha=0.2)
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

    ax.legend()
    ax.set_title("Genres Music Clusters Results", fontsize=apr_constants.TITLE_FONT_SIZE)
    if savePlot:
        print('Save Clusters Plot')
        common_functions.check_create_directory(savePath)
        plt.savefig(savePath + fileName + ' - ' + 'Clusters Plot.jpg')
    plt.show()


def plot_confusion_matrix_kmeans(inputData, savePlot=True, labels=[], target_names=[],
                                 fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    inputData['predicted_label'] = labels
    data = metrics.confusion_matrix(inputData['genre'], inputData['predicted_label'])

    df_cm = pd.DataFrame(data, columns=np.unique(target_names), index=np.unique(target_names))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.4)
    heatmap = sns.heatmap(df_cm, cmap="Blues", annot=True, fmt='g', annot_kws={"size": 8}, square=True)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45)
    plt.title('CM for K-Means', fontsize=apr_constants.TITLE_FONT_SIZE)
    if savePlot:
        print('Save K-means Confusion Matrix')
        common_functions.check_create_directory(savePath)
        plt.savefig(savePath + fileName + ' - ' + 'K-means Confusion Matrix Plot.jpg')
    plt.show()


def plot_Silhouette(inputData, minClusters=2, maxClutsers=5, savePlot=False, fileName=apr_constants.DEFAULT_FILE_NAME,
                    savePath=apr_constants.PROJECT_ROOT):
    eval_data = inputData.copy()
    silhouette_score_values = list()
    executiontime_values = list()

    NumberOfClusters = range(minClusters, maxClutsers + 1)
    for i in NumberOfClusters:
        start_time = time.time()
        clusters = KMeans(i)
        clusters.fit(eval_data)
        cluster_labels = clusters.predict(eval_data)
        executionTime = time.time() - start_time
        executiontime_values.append(executionTime)
        silhouette_score_values.append(
            metrics.silhouette_score(eval_data, cluster_labels, metric='euclidean', sample_size=None,
                                     random_state=None))

    fig, ax1 = plt.subplots(figsize=(15, 10))
    y_ax_ticks = np.arange(0, max(silhouette_score_values) + 1, 0.1)
    x_ax_ticks = np.arange(minClusters, maxClutsers + 1, 1)

    ax1.plot(NumberOfClusters, silhouette_score_values)
    ax1.plot(NumberOfClusters, silhouette_score_values, 'bo')
    ax1.set_title("Silhouette Score Values vs Numbers of Clusters", fontsize=22)
    ax1.set_yticks(y_ax_ticks)
    ax1.set_ylabel('Silhouette Score', fontsize=22)
    ax1.set_xticks(x_ax_ticks)
    ax1.set_xlabel('Number Of Clusters', fontsize=22)
    ax1.grid(False)

    ax2 = ax1.twinx()
    y_ax2_ticks = np.arange(0, max(executiontime_values) + 1, 0.3)
    ax2.plot(NumberOfClusters, executiontime_values, 'orange', linestyle='dashed')
    ax2.plot(NumberOfClusters, executiontime_values, 'orange', marker="o", linestyle='dashed')
    ax2.set_yticks(y_ax2_ticks)
    ax2.set_ylabel('Fit Time (sec)', fontsize=22)
    ax2.grid(False)

    Optimal_NumberOf_Components = NumberOfClusters[silhouette_score_values.index(max(silhouette_score_values))]
    Worst_NumberOf_Components = NumberOfClusters[silhouette_score_values.index(min(silhouette_score_values))]

    Optimal_Execution_Time = NumberOfClusters[executiontime_values.index(max(executiontime_values))]
    Worst_Execution_Time = NumberOfClusters[executiontime_values.index(min(executiontime_values))]

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

    ax1.vlines(x=Optimal_NumberOf_Components, ymin=0, ymax=max(silhouette_score_values), linewidth=2, color='green',
               label='Max Value', linestyle='dashdot')
    ax1.vlines(x=Worst_NumberOf_Components, ymin=0, ymax=min(silhouette_score_values), linewidth=2, color='red',
               label='min Value', linestyle='dashdot')

    # Adding legend
    ax1.legend(loc='upper center', prop={'size': apr_constants.LEGEND_SIZE})
    ax2.legend(['ExecutionTime'], loc='upper right', prop={'size': apr_constants.LEGEND_SIZE})

    if savePlot:
        print('Save Silhouette Plot')
        common_functions.check_create_directory(savePath)
        plt.savefig(savePath + fileName + ' - ' + 'Clusters Silhouette Plot.jpg')
    plt.show()
    # print("Optimal number of components is:", Optimal_NumberOf_Components)


def plot_roc(y_test, y_score, classifierName=apr_constants.DEFAULT_CLASSIFIER_NAME, savePlot=False, target_names=[],
             fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT, type_learning='SL'):
    genres = target_names
    if len(target_names) == 10:
        test_label = preprocessing.label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        if type_learning == 'SL':
            y_label = y_score
        elif type_learning == 'UL':
            y_label = preprocessing.label_binarize(y_score, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    elif len(target_names) == 5:
        test_label = preprocessing.label_binarize(y_test, classes=[0, 1, 2, 3, 4])
        if type_learning == 'SL':
            y_label = y_score
        elif type_learning == 'UL':
            y_label = preprocessing.label_binarize(y_score, classes=[0, 1, 2, 3, 4])
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
    plt.title('Receiver operating characteristic for ' + classifierName.replace('_', ' ').upper(),
              fontsize=apr_constants.TITLE_FONT_SIZE)
    plt.legend(loc='lower right', prop={'size': apr_constants.LEGEND_SIZE})

    if savePlot:
        print('Save ROC Plot')
        common_functions.check_create_directory(savePath)
        plt.savefig(savePath + fileName + ' - ' + 'ROC Plot.jpg')
    plt.show()


def plot_Classification_Report(inputData, exportJSON=True, labels=[], target_names=[],
                               fileName=apr_constants.DEFAULT_FILE_NAME,
                               savePath=apr_constants.PROJECT_ROOT):
    inputData['predicted_label'] = labels
    report_dic = False
    if exportJSON:
        report_dic = True
    report = metrics.classification_report(inputData['genre'], inputData['predicted_label'], target_names=target_names,
                                           output_dict=report_dic)
    # print('results_reports: ', report)

    if exportJSON:
        common_functions.check_create_directory(savePath + apr_constants.DATA)
        with open(savePath + apr_constants.DATA + fileName + '_report_results.json', 'w') as res:
            json.dump(report, res, indent=4)
    plt.show()


def plot_BPM_Bar(inputData, savePlot=False, target_names=[], fileName=apr_constants.DEFAULT_FILE_NAME,
                 savePath=apr_constants.PROJECT_ROOT):
    plt.figure(figsize=(15, 7))
    new_data = inputData[['genre', 'tempo']].copy()
    if len(target_names) == 10:
        genres = {i: target_names[i] for i in range(0, len(target_names))}
        # genres = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop',
        #           8: 'reggae', 9: 'rock'}
    elif len(target_names) == 5:
        genres = {i: target_names[i] for i in range(0, len(target_names))}
        # genres = {0: 'classical', 1: 'disco', 2: 'jazz', 3: 'reggae', 4: 'rock'}

    new_data.genre = [genres[item] for item in new_data.genre]

    sns.boxplot(x=new_data.genre, y=new_data.tempo, palette='husl')
    plt.title('BPM Boxplot for Genres', fontsize=apr_constants.TITLE_FONT_SIZE)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=10);
    plt.xlabel('Genre', fontsize=22)
    plt.ylabel('BPM', fontsize=22)

    if savePlot:
        print('Save BPM Bar')
        common_functions.check_create_directory(savePath)
        plt.savefig(savePath + fileName + ' - ' + 'BPM BoxPlot.jpg')
    plt.show()


def plot_confusion_matrix(clf, X_test, y_test, classes, normalize='true',
                          classifierName=apr_constants.DEFAULT_CLASSIFIER_NAME, savePlot=False,
                          fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    fig, ax = plt.subplots(figsize=(10, 10))
    metrics.plot_confusion_matrix(clf, X_test, y_test, normalize=normalize, cmap=plt.cm.Blues, ax=ax,
                                  display_labels=classes, values_format='.0f')
    ax.set_title('CM for ' + classifierName.replace('_', ' ').upper(), fontsize=apr_constants.TITLE_FONT_SIZE)
    plt.xticks(rotation=45)
    plt.grid(False)
    if savePlot:
        print('Save Confusion Matrix Plot')
        common_functions.check_create_directory(savePath)
        plt.savefig(savePath + fileName + ' - ' + 'Confusion Matrix Plot.jpg')
    plt.show()


def plot_predictions_compare(normalize_cm, y_test, y_pred, target_names=[],
                             classifierName=apr_constants.DEFAULT_CLASSIFIER_NAME,
                             savePlot=False, fileName=apr_constants.DEFAULT_FILE_NAME,
                             savePath=apr_constants.PROJECT_ROOT):
    genres = target_names
    calc_cm = metrics.confusion_matrix(y_test, y_pred, normalize=normalize_cm)
    bar = pd.DataFrame(calc_cm, columns=genres, index=genres)
    ax = bar.plot(kind='bar', figsize=(15, 15), fontsize=14, width=0.8)
    plt.title('Musical Genres BarPlot Predictions For ' + classifierName.replace('_', ' ').upper(),
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

    if savePlot:
        print('Save Models Predictions Compare')
        common_functions.check_create_directory(savePath)
        plt.savefig(savePath + fileName + ' - ' + 'Predictions Compare Plot.jpg')
    plt.show()


def plot_predictions_simple_compare(inputData, target_names=[],
                                    savePlot=False, fileName=apr_constants.DEFAULT_FILE_NAME,
                                    savePath=apr_constants.PROJECT_ROOT):
    ax = inputData.plot(kind="bar", figsize=(15, 15), fontsize=14, width=0.6)
    ax.set_xticklabels(target_names)
    ax.legend(["Real Value", "Predict Value"])
    plt.title('Simple BarPlot Predictions Evaluation', fontsize=26)
    plt.xlabel('Musical Genres', fontsize=18)
    plt.ylabel('Number of Occurrences', fontsize=18)
    for p in ax.patches:
        ax.annotate(format(p.get_height()),
                    (p.get_x() + (p.get_width() / 2) + 0.015, p.get_height() + 5), ha='center', va='center',
                    size=18, xytext=(0, 8), textcoords='offset points', fontsize=14, rotation=90)
    if savePlot:
        print('Save Simple Models Predictions Compare')
        common_functions.check_create_directory(savePath)
        plt.savefig(savePath+fileName + ' - ' + 'Simple Predictions Compare Plot.jpg')
    plt.show()
