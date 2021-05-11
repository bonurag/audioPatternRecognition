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


def load_data(data_path, normalization='std', remove_null_value=True, columns_to_drop=None):
    if columns_to_drop is None:
        columns_to_drop = []
    df = pd.read_csv(data_path)

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    if remove_null_value:
        for check in df.isnull().sum().sort_values(ascending=False):
            if check > 0:
                df = df.fillna(0)

    ord_enc = preprocessing.OrdinalEncoder()
    df['genre'] = ord_enc.fit_transform(df[['genre']])

    # Split df into x and Y
    target_col = 'genre'
    x = df.loc[:, df.columns != target_col]
    y = df.loc[:, target_col]

    x_cols = x.columns
    if normalization == 'std':
        # NORMALIZE X WITH STANDARD SCALER #
        resized_data = preprocessing.StandardScaler()
        np_scaled = resized_data.fit_transform(x)
    elif normalization == 'min_max':
        # NORMALIZE X WITH Min Max SCALER #
        resized_data = preprocessing.MinMaxScaler()
        np_scaled = resized_data.fit_transform(x)
    elif normalization is None:
        np_scaled = x

    x = pd.DataFrame(np_scaled, columns=x_cols)

    return x, y, df


def get_pca(input_data, input_columns, num_components=1, plot_matrix=True, save_plot=False, target_names=None,
            file_name=apr_constants.DEFAULT_FILE_NAME, save_path=apr_constants.PROJECT_ROOT):
    # PCA Components #
    if target_names is None:
        target_names = []
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(input_data)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # Concatenate With Target Label
    concat_data = pd.concat([principal_df, input_columns], axis=1)

    if plot_matrix:
        plot_functions.plot_pca(concat_data[['PC1', 'PC2', 'genre']], save_plot, target_names, file_name, save_path)
    return concat_data


def calculate_metrics(y_test, y_pred, y_proba, execution_time, classifier_name, target_names):
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

    single_metrics['single_results']['CLASSIFIER_NAME'] = classifier_name
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
    single_metrics['single_results']['EXECUTION_TIME'] = execution_time

    clf_report = metrics.classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    return single_metrics, clf_report


def model_assess(clf, x_train, x_test, y_train, y_test, plot_roc=True, plot_conf_matrix=True, predictions_compare=True,
                 classifier_name=apr_constants.DEFAULT_CLASSIFIER_NAME, use_predict_proba=True, target_names=None,
                 file_name=apr_constants.DEFAULT_FILE_NAME,
                 save_path=apr_constants.PROJECT_ROOT):
    if target_names is None:
        target_names = []
    start_time = time.time()
    genres = target_names
    if use_predict_proba:
        y_score = clf.fit(x_train, y_train).decision_function(x_test)
    else:
        clf.fit(x_train, y_train)
        y_score = clf.predict_proba(x_test)

    y_pred = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)

    if not os.path.exists(save_path + apr_constants.MODEL):
        os.makedirs(save_path + apr_constants.MODEL)
    common_functions.save_model(clf, save_path + apr_constants.MODEL + file_name)
    # print()
    if plot_conf_matrix:
        plot_functions.plot_confusion_matrix(clf, x_test, y_test, genres, None, classifier_name, True, file_name,
                                             save_path)

    # print()
    if plot_roc:
        plot_functions.plot_roc(y_test, y_score, classifier_name, True, genres, file_name, save_path)

    # print()
    if predictions_compare:
        plot_functions.plot_predictions_compare(None, y_test, y_pred, genres, classifier_name, True, file_name,
                                                save_path)

    # print()
    execution_time = time.time() - start_time
    single_metrics, report = calculate_metrics(y_test, y_pred, y_proba, execution_time, classifier_name, genres)

    # print(f'CLASSIFICATION REPORT:\n{report}')
    # print('EXECUTION TIME: %s Sec' % executionTime)
    # print()
    return single_metrics, report


def dictionary_to_data_frame(input_data, columns):
    rows = []
    for i in input_data.keys():
        for j in input_data[i].keys():
            single_row = input_data[i][j]
            rows.append(single_row)
    return rows, columns


def get_model(test_model=False):
    classifier_models = {'SVM': [], 'RF': [], 'ANN': []}

    if test_model:
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


def get_results(classifier_models, x_train, x_test, y_train, y_test, export_csv=True,
                export_json=True, target_names=None,
                file_name=apr_constants.DEFAULT_FILE_NAME,
                save_path=apr_constants.PROJECT_ROOT):
    if target_names is None:
        target_names = []
    columns__data_frame = ['CLASSIFIER_NAME', 'ACC', 'ERR', 'LOSS', 'K', 'MSE', 'RMSE', 'MAE', 'WEIGHTED_F1_SCORE',
                           'WEIGHTED_PRECISION', 'WEIGHTED_RECALL', 'EXECUTION_TIME']
    all_models_results = {}
    all_models_reports = {}
    for key in classifier_models.keys():
        model_name = key
        if model_name == 'SVM':
            use_predict_proba = True
        elif model_name == 'RF' or model_name == 'ANN':
            use_predict_proba = False
        image_file_name = model_name + '_' + file_name
        single_data_result, report = model_assess(classifier_models.get(model_name), x_train, x_test, y_train, y_test,
                                                  True,
                                                  True, True, model_name, use_predict_proba, target_names,
                                                  image_file_name, save_path)
        all_models_results[model_name] = single_data_result
        all_models_reports[model_name] = report
    # print()
    # print(f'CLASSIFICATION MODELS RESULTS:')
    rows, columns = dictionary_to_data_frame(all_models_results, columns__data_frame)

    results = pd.DataFrame(rows, columns=columns)
    results_reports = pd.DataFrame.from_dict(all_models_reports)
    # print('results_reports: ', results_reports)
    if export_csv:
        if not os.path.exists(save_path + apr_constants.DATA):
            os.makedirs(save_path + apr_constants.DATA)
        results.to_csv(save_path + apr_constants.DATA + file_name + '_results.csv', index=False, header=True, sep='\t',
                       encoding='utf-8')
        results_reports.to_csv(save_path + apr_constants.DATA + file_name + '_reports_results.csv', index=False,
                               header=True, sep='\t', encoding='utf-8')
    if export_json:
        if not os.path.exists(save_path + apr_constants.DATA):
            os.makedirs(save_path + apr_constants.DATA)
        with open(save_path + apr_constants.DATA + file_name + '_results.json', 'w') as res:
            json.dump(all_models_results, res, indent=4)
        with open(save_path + apr_constants.DATA + file_name + '_reports_results.json', 'w') as rep:
            json.dump(all_models_reports, rep, indent=4)
    return results
