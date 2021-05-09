import os

import pandas as pd

import time
import json

from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn import metrics

import apr_constants
import plot_functions
import common_functions


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


def getPCA(inputData, inputColumns, numOfComponents=1, plotMatrix=True, savePlot=False, target_names=[],
           fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    # PCA Components #
    pca = PCA(n_components=numOfComponents)
    principal_components = pca.fit_transform(inputData)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Concatenate With Target Label
    concat_data = pd.concat([principal_df, inputColumns], axis=1)

    if plotMatrix:
        plot_functions.plot_PCA(concat_data[['PC1', 'PC2', 'genre']], savePlot, target_names, fileName, savePath)
    return concat_data


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

    if not os.path.exists(savePath + apr_constants.MODEL):
        os.makedirs(savePath + apr_constants.MODEL)
    common_functions.save_model(clf, savePath + apr_constants.MODEL + fileName)
    # print()
    if plotConfMatrix:
        plot_functions.plot_confusion_matrix(clf, X_test, y_test, genres, None, classifierName, True, fileName, savePath)

    # print()
    if plotRoc:
        plot_functions.plot_roc(y_test, y_score, classifierName, True, genres, fileName, savePath)

    # print()
    if predictionsCompare:
        plot_functions.plot_predictions_compare(None, y_test, y_pred, genres, classifierName, True, fileName, savePath)

    # print()
    executionTime = time.time() - start_time
    single_metrics, report = calculate_metrics(y_test, y_pred, y_proba, executionTime, classifierName, genres)

    # print(f'CLASSIFICATION REPORT:\n{report}')
    # print('EXECUTION TIME: %s Sec' % executionTime)
    # print()
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
        RF_Classifier = RandomForestClassifier(n_estimators=10, random_state=10)
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
        RF_Classifier = RandomForestClassifier(n_estimators=1000, random_state=10)
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
    # print()
    # print(f'CLASSIFICATION MODELS RESULTS:')
    rows, columns = dictionaryToDataFrame(all_models_results, columns_DataFrame)

    results = pd.DataFrame(rows, columns=columns)
    results_reports = pd.DataFrame.from_dict(all_models_reports)
    # print('results_reports: ', results_reports)
    if exportCSV:
        if not os.path.exists(savePath + apr_constants.DATA):
            os.makedirs(savePath + apr_constants.DATA)
        results.to_csv(savePath + apr_constants.DATA + fileName + '_results.csv', index=False, header=True, sep='\t', encoding='utf-8')
        results_reports.to_csv(savePath + apr_constants.DATA + fileName + '_reports_results.csv', index=False, header=True, sep='\t', encoding='utf-8')
    if exportJSON:
        if not os.path.exists(savePath + apr_constants.DATA):
            os.makedirs(savePath + apr_constants.DATA)
        with open(savePath + apr_constants.DATA + fileName + '_results.json', 'w') as res:
            json.dump(all_models_results, res, indent=4)
        with open(savePath + apr_constants.DATA + fileName + '_reports_results.json', 'w') as rep:
            json.dump(all_models_reports, rep, indent=4)
    return results
