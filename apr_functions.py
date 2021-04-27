import pandas as pd

import time
import json

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn import metrics

from itertools import cycle

import pickle
import apr_constants


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

    return X, y, df


def prepare_datasets(X, y, test_size=0.3):
    """Loads data and splits it into train, validation and test sets.

    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split

    :return X_train (ndarray): Input training set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_test (ndarray): Target test set
    """
    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=100)

    return X_train, X_test, y_train, y_test


def plot_correlation_matrix(correlation_matrix, savePlot=True, fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1.4)
    sns.heatmap(correlation_matrix, cmap='coolwarm', square=True)
    plt.title('Correlation between different features', fontsize=apr_constants.TITLE_FONT_SIZE)
    if savePlot:
        print('Save Correlation Matrix')
        plt.savefig(savePath+fileName + ' - ' + 'Correlation Matrix.jpg')
    plt.show()


def getCorrelatedFeatures(inputData, corrValue=0.9, dropFeatures=True, plotMatrix=True, savePlot=False,
                          fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    correlation_matrix = inputData.corr(method='pearson', min_periods=50)
    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) >= corrValue:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

    if dropFeatures == False:
        # print('Features Uneleted')
        if plotMatrix == True:
            # print('Print Correlation Matrix')
            plot_correlation_matrix(correlation_matrix, savePlot, fileName, savePath)
        return correlated_features
    elif dropFeatures == True:
        if len(correlated_features) > corrValue:
            # print('Features Deleted')
            inputData.drop(labels=correlated_features, axis=1, inplace=True)
        if plotMatrix == True:
            # print('Print Correlation Matrix')
            drop_correlation_matrix = inputData.corr(method='pearson', min_periods=50)
            plot_correlation_matrix(drop_correlation_matrix, savePlot, fileName, savePath)


def plot_PCA(inputPCAData, savePlot=False, target_names=[], fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    plt.figure(figsize=(20, 10))
    new_data = inputPCAData.copy()
    genres = target_names
    if len(target_names) == 10:
        genres = {i: target_names[i] for i in range(0, len(target_names))}
    # genres = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop',
    #           8: 'reggae', 9: 'rock'}
    elif len(target_names) == 5:
        genres = {i: target_names[i] for i in range(0, len(target_names))}
    # genres = {0: 'classical', 1: 'disco', 2: 'jazz', 3: 'reggae', 4: 'rock'}

    new_data.genre = [genres[item] for item in new_data.genre]

    sns.scatterplot(x='PC1', y='PC2', data=new_data, hue='genre', alpha=0.6, palette='deep')

    plt.title('PCA on Genres', fontsize=apr_constants.TITLE_FONT_SIZE)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=10)
    plt.xlabel('Principal Component 1', fontsize=22)
    plt.ylabel('Principal Component 2', fontsize=22)

    if savePlot:
        print('Save PCA Plot')
        plt.savefig(savePath+fileName + ' - ' + 'PCA Scatter Plot.jpg')
    plt.show()


def getPCA(inputData, inputColumns, numOfComponents=1, plotMatrix=True, savePlot=False, target_names=[],
           fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    # PCA Components #
    pca = PCA(n_components=numOfComponents)
    principalComponents = pca.fit_transform(inputData)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    # Concatenate With Target Label
    concatData = pd.concat([principalDf, inputColumns], axis=1)

    if plotMatrix:
        plot_PCA(concatData, savePlot, target_names, fileName, savePath)
    return concatData


def plot_BPM_Bar(inputData, savePlot=False, target_names=[], fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
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
        plt.savefig(savePath+fileName + ' - ' + 'BPM BoxPlot.jpg')
    plt.show()


def plot_roc(y_test, y_score, classifierName=apr_constants.DEFAULT_CLASSIFIER_NAME, savePlot=False, target_names=[],
             fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    genres = target_names
    if len(target_names) == 10:
        test_label = preprocessing.label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    elif len(target_names) == 5:
        test_label = preprocessing.label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    n_classes = test_label.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(test_label[:, i], y_score[:, i])
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
    plt.title('Receiver operating characteristic for ' + classifierName.replace('_', ' ').upper(), fontsize=apr_constants.TITLE_FONT_SIZE)
    plt.legend(loc='lower right', prop={'size': apr_constants.LEGEND_SIZE})

    if savePlot:
        print('Save ROC Plot')
        plt.savefig(savePath+fileName + ' - ' + 'ROC Plot.jpg')
    plt.show()


def plot_confusion_matrix(clf, X_test, y_test, classes, normalize='true', classifierName=apr_constants.DEFAULT_CLASSIFIER_NAME, savePlot=False,
            fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    fig, ax = plt.subplots(figsize=(10, 10))
    metrics.plot_confusion_matrix(clf, X_test, y_test, normalize=normalize, cmap=plt.cm.Blues, ax=ax,
                                  display_labels=classes, values_format='.0f')
    ax.set_title('CM for ' + classifierName.replace('_', ' ').upper(), fontsize=apr_constants.TITLE_FONT_SIZE)
    plt.xticks(rotation=45)
    plt.grid(False)
    if savePlot:
        print('Save Confusion Matrix Plot')
        plt.savefig(savePath+fileName + ' - ' + 'Confusion Matrix Plot.jpg')
    plt.show()


def plot_predictions_compare(normalize_cm, y_test, y_pred, target_names=[], classifierName=apr_constants.DEFAULT_CLASSIFIER_NAME,
                             savePlot=False, fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    genres = target_names
    calc_cm = metrics.confusion_matrix(y_test, y_pred, normalize=normalize_cm)
    bar = pd.DataFrame(calc_cm, columns=genres, index=genres)
    ax = bar.plot(kind='bar', figsize=(15, 15), fontsize=14, width=0.8)
    plt.title('Musical Genres BarPlot Predictions For ' + classifierName.replace('_', ' ').upper(), fontsize=apr_constants.TITLE_FONT_SIZE)
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
        plt.savefig(savePath+fileName + ' - ' + 'Predictions Compare Plot.jpg')
    plt.show()


def calculate_metrics(y_test, y_pred, y_proba, executionTime, classifierName, target_names):
    single_metrics = {
        'single_results': {}
    }

    accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    error_score = (1 - metrics.accuracy_score(y_test, y_pred)) * 100
    loss = metrics.log_loss(y_test, y_proba)
    kappa = metrics.cohen_kappa_score(y_test, y_pred, labels=None, weights=None)
    mse = metrics.mean_squared_error(y_test, y_pred, squared=True)
    rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
    precision_score = metrics.precision_score(y_test, y_pred, average='weighted')
    recall_score = metrics.recall_score(y_test, y_pred, average='weighted')

    single_metrics['single_results']['CLASSIFIER_NAME'] = classifierName
    single_metrics['single_results']['ACC'] = accuracy
    single_metrics['single_results']['ERR'] = error_score
    single_metrics['single_results']['LOSS'] = loss
    single_metrics['single_results']['K'] = kappa
    single_metrics['single_results']['MSE'] = mse
    single_metrics['single_results']['RMSE'] = rmse
    single_metrics['single_results']['MAE'] = mae
    single_metrics['single_results']['WEIGHTED_F1_SCORE'] = f1_score
    single_metrics['single_results']['WEIGHTED_PRECISION'] = precision_score
    single_metrics['single_results']['WEIGHTED_RECALL'] = recall_score
    single_metrics['single_results']['EXECUTION_TIME'] = executionTime

    clf_report = metrics.classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    return single_metrics, clf_report


def save_model(inputClassifier, saveModelName):
    print('Save Model {} - {} '.format(inputClassifier, saveModelName))
    filename = saveModelName + '_Model.sav'
    pickle.dump(inputClassifier, open(filename, 'wb'))


def model_assess(clf, X_train, X_test, y_train, y_test, plotRoc=True, plotConfMatrix=True, predictionsCompare=True,
                 classifierName=apr_constants.DEFAULT_CLASSIFIER_NAME, usePredictProba=True, target_names=[],
                 fileName=apr_constants.DEFAULT_FILE_NAME,
                 savePath=apr_constants.PROJECT_ROOT):
    start_time = time.time()
    genres = target_names
    if usePredictProba:
        y_score = clf.fit(X_train, y_train).decision_function(X_test)
    else:
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    save_model(clf, savePath+apr_constants.MODEL+fileName)
    print()
    if plotConfMatrix:
        plot_confusion_matrix(clf, X_test, y_test, genres, None, classifierName, True, fileName, savePath)

    print()
    if plotRoc:
        plot_roc(y_test, y_score, classifierName, True, genres, fileName, savePath)

    print()
    if predictionsCompare:
        plot_predictions_compare(None, y_test, y_pred, genres, classifierName, True, fileName, savePath)

    print()
    executionTime = time.time() - start_time
    single_metrics, report = calculate_metrics(y_test, y_pred, y_proba, executionTime, classifierName, genres)

    print(f'CLASSIFICATION REPORT:\n{report}')
    print('EXECUTION TIME: %s Sec' % executionTime)
    print()
    return single_metrics, report


def dictionaryToDataFrame(inputData, columns):
    rows = []
    for i in inputData.keys():
        for j in inputData[i].keys():
            single_row = inputData[i][j]
            rows.append(single_row)
    return rows, columns


def getModel(test_Model=False):
    classifier_models = {'SVM': [], 'RF': [], 'ANN': []}

    if test_Model:
        # Linear Support Vector Machine
        SVM_Classifier = SVC(C=10, kernel='rbf', probability=True, random_state=10)
        classifier_models.update({'SVM': SVM_Classifier})

        # Random Forest
        RF_Classifier = RandomForestClassifier(n_estimators=100, random_state=10)
        classifier_models.update({'RF': RF_Classifier})

        # Artificial Neural Network
        ANN_Classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(16, 16), random_state=1,
                                       activation='relu', learning_rate='adaptive', early_stopping=False, verbose=False)
        classifier_models.update({'ANN': ANN_Classifier})

    else:
        # Linear Support Vector Machine
        SVM_Classifier = SVC(C=100, kernel='rbf', probability=True, random_state=10)
        classifier_models.update({'SVM': SVM_Classifier})

        # Random Forest
        RF_Classifier = RandomForestClassifier(n_estimators=100, random_state=10)
        classifier_models.update({'RF': RF_Classifier})

        # Artificial Neural Network
        ANN_Classifier = MLPClassifier(solver='adam', alpha=1e-5,
                                       hidden_layer_sizes=(512, 256, 128, 128, 128, 128, 64, 64, 32, 32),
                                       random_state=1, activation='relu', learning_rate='adaptive',
                                       early_stopping=False,
                                       verbose=False, max_iter=2000)
        classifier_models.update({'ANN': ANN_Classifier})
    return classifier_models


def getResults(classifier_models, X_train, X_test, y_train, y_test, exportCSV=True,
               exportJSON=True, target_names=[],
               fileName=apr_constants.DEFAULT_FILE_NAME,
               savePath=apr_constants.PROJECT_ROOT):
    columns_DataFrame = ['CLASSIFIER_NAME', 'ACC', 'ERR', 'LOSS', 'K', 'MSE', 'RMSE', 'MAE', 'WEIGHTED_F1_SCORE',
                         'WEIGHTED_PRECISION', 'WEIGHTED_RECALL', 'EXECUTION_TIME']
    all_models_results = {}
    all_models_reports = {}
    for key in classifier_models.keys():
        model_name = key
        if model_name == 'SVM':
            usePredictProba = True
        elif model_name == 'RF' or model_name == 'ANN':
            usePredictProba = False
        image_file_name = model_name + '_' + fileName
        single_data_result, report = model_assess(classifier_models.get(model_name), X_train, X_test, y_train, y_test,
                                                  True,
                                                  True, True, model_name, usePredictProba, target_names,
                                                  image_file_name, savePath)
        all_models_results[model_name] = single_data_result
        all_models_reports[model_name] = report
    print()
    print(f'CLASSIFICATION MODELS RESULTS:')
    rows, columns = dictionaryToDataFrame(all_models_results, columns_DataFrame)

    results = pd.DataFrame(rows, columns=columns)
    results_reports = pd.DataFrame.from_dict(all_models_reports)
    print('results_reports: ', results_reports)
    if exportCSV:
        results.to_csv(savePath+apr_constants.DATA+fileName + '_results.csv', index=False, header=True, sep='\t', encoding='utf-8')
        results_reports.to_csv(savePath+apr_constants.DATA+fileName + '_reports_results.csv', index=False, header=True, sep='\t', encoding='utf-8')
    if exportJSON:
        with open(savePath+apr_constants.DATA+fileName + '_results.json', 'w') as res:
            json.dump(all_models_results, res, indent=4)
        with open(savePath+apr_constants.DATA+fileName + '_reports_results.json', 'w') as rep:
            json.dump(all_models_reports, rep, indent=4)
    return results
