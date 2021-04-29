import numpy as np
import pandas as pd

import time

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import apr_constants
import common_functions
import plot_functions


def load_data(data_path, normalization='std', removeNullValue=True, columnsToDrop=[]):
    df = pd.read_csv(data_path)

    if columnsToDrop:
        df = df.drop(columns=columnsToDrop)

    if removeNullValue:
        for check in df.isnull().sum().sort_values(ascending=False):
            if check > 0:
                df = df.fillna(0)

    ord_enc = preprocessing.OrdinalEncoder()
    df['genre'] = ord_enc.fit_transform(df[['genre']])

    # Split df into x and Y
    target_col = 'genre'
    X = df.loc[:, df.columns != target_col]
    y = df.loc[:, target_col]

    x_cols = X.columns
    if normalization == 'std':
        # NORMALIZE X WITH STANDARD SCALER #
        resized_data = preprocessing.StandardScaler()
        np_scaled = resized_data.fit_transform(X)
    elif normalization == 'min_max':
        # NORMALIZE X WITH Min Max SCALER #
        resized_data = preprocessing.MinMaxScaler()
        np_scaled = resized_data.fit_transform(X)
    elif normalization == None:
        np_scaled = X

    X = pd.DataFrame(np_scaled, columns=x_cols)
    y = pd.DataFrame(y).fillna(0).astype(int)

    return X, y, df


def getPCA_VarRatio_Plot(inputData, savePlot=True, fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    cov_mat = np.cov(inputData.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    tot = sum(eigen_vals)

    # var_exp ratio is fraction of eigen_val to total sum
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]

    # calculate the cumulative sum of explained variances
    cum_var_exp = np.cumsum(var_exp)
    plt.figure(figsize=(15, 10))
    plt.bar(range(0, len(inputData.columns)), var_exp, alpha=0.75, align='center',
            label='individual explained variance')
    plt.xticks(np.arange(0, len(inputData.columns), 1))
    plt.step(range(0, len(inputData.columns)), cum_var_exp, where='mid', label='cumulative explained variance', c='red')
    plt.ylim(0, 1.1)
    plt.xlabel('Principal Components', fontsize=22)
    plt.ylabel('Explained variance ratio', fontsize=22)
    plt.legend(loc='best', prop={'size': apr_constants.LEGEND_SIZE})
    if savePlot:
        print('Save PCA Variance Ratio Plot')
        common_functions.check_create_directory(savePath)
        plt.savefig(savePath+fileName + ' - ' + 'PCA Variance Ratio Plot.jpg')
    plt.show()


def getPCAWithCentroids(inputData, inputColumns, numOfComponents=1, plotMatrix=True, savePlot=False, target_names=[],
                        fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT, centroidsValue=[]):
    useData = inputData.copy()
    columnData = inputColumns.copy()
    if numOfComponents > 1:
        columnsComponents = []
        for col in range(numOfComponents):
            columnsComponents.append('PC' + str(col + 1))
    elif numOfComponents == 1:
        columnsComponents = ['PC1']

    # PCA Components #
    pca = PCA(n_components=numOfComponents)
    pca_fit = pca.fit(useData)
    principalComponents = pca_fit.transform(useData)
    principalDf = pd.DataFrame(data=principalComponents, columns=columnsComponents)

    componentsValue = "Component" if numOfComponents == 1 else "Components"
    # print('PCA Variance Ratio For {} {}: {}'.format(numOfComponents, componentsValue,
    #                                                 pca.explained_variance_ratio_.sum()))

    # Transform Clusters Centroids
    c_transformed = pca_fit.transform(centroidsValue)

    # Concatenate With Target Label
    # frames = [principalDf, columnData]
    # concatData = pd.concat(frames)
    concatData = pd.concat([principalDf.reset_index(drop=True), columnData.reset_index(drop=True)], axis=1)
    if numOfComponents == 2:
        if plotMatrix:
            plot_functions.plot_PCA(concatData, savePlot, target_names, fileName, savePath)
        return concatData, c_transformed
    elif numOfComponents == 3:
        if plotMatrix:
            plot_functions.plot_3D_PCA(concatData, savePlot, fileName, savePath)
    else:
        return concatData


def runKmeans(inputData, clustersNumber=1, randomState=10, modelFileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    start_time = time.time()
    kmean = KMeans(clustersNumber, random_state=randomState)
    kmean.fit(inputData)
    common_functions.check_create_directory(savePath+apr_constants.MODEL)
    common_functions.save_model(kmean, savePath+apr_constants.MODEL+modelFileName)
    kmean.predict(inputData)
    labels = kmean.labels_
    centroids = kmean.cluster_centers_
    executionTime = time.time() - start_time
    # print('EXECUTION TIME: %s Sec' % executionTime)
    return labels, centroids


def runKmeansSplitData(inputDataTrain, inputDataTest, clustersNumber=1, randomState=10):
    kmean = KMeans(clustersNumber, random_state=randomState)
    kmean.fit(inputDataTrain)
    kmean.predict(inputDataTrain)
    labels_train = kmean.labels_
    labels_test = kmean.predict(inputDataTest)
    centroids = kmean.cluster_centers_
    return labels_train, labels_test, centroids
