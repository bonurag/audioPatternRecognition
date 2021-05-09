import pickle
import os

from pathlib import Path

from sklearn.model_selection import train_test_split

import plot_functions
import apr_constants
import common_functions

features_file_path = apr_constants.FEATURES_FILE_PATH


def get_genres():
    genre_target_names = []
    for folder in os.listdir(os.path.join(features_file_path)):
        genre_target_names.append(str(folder))

    if len(genre_target_names) == 0:
        genre_target_names = apr_constants.GENRE_TARGET_NAMES
    return genre_target_names


def save_model(inputClassifier, saveModelName):
    print('Save Model {} - {} '.format(inputClassifier, saveModelName))
    filename = saveModelName + '_Model.sav'
    pickle.dump(inputClassifier, open(filename, 'wb'))


def save_test_data(x_test, y_test, saveDataName):
    tuple_objects = (x_test, y_test)
    filename = saveDataName + '_TestData.pkl'
    pickle.dump(tuple_objects, open(filename, 'wb'))


def load_test_data(inputPath, inputFileName):
    if inputFileName.endswith('.pkl'):
        inputFileName = inputFileName[0:len(inputFileName)-4]
    file_to_load = inputPath + inputFileName + '.pkl'
    pickled_x_test, pickled_y_test = pickle.load(open(file_to_load, 'rb'))
    print('Test Files loading complete successfully!')
    return pickled_x_test, pickled_y_test


def load_model(inputPath, inputFileName):
    if inputFileName.endswith('.sav'):
        inputFileName = inputFileName[0:len(inputFileName)-4]
    model_to_load = inputPath + inputFileName + '.sav'
    model = pickle.load(open(model_to_load, 'rb'))
    print('Model loading complete successfully!')
    return model

def prepare_datasets(X, y, test_size=0.3, fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    # create train, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=100)
    common_functions.check_create_directory(savePath + apr_constants.TEST)
    save_test_data(X_test, y_test, savePath + apr_constants.TEST + fileName)
    return X_train, X_test, y_train, y_test


def check_create_directory(input_path):
    if not os.path.exists(input_path):
        Path(input_path).mkdir(parents=True, exist_ok=True)


def getCorrelatedFeatures(inputData, corrValue=0.9, dropFeatures=True, plotMatrix=True, savePlot=False,
                          fileName=apr_constants.DEFAULT_FILE_NAME, savePath=apr_constants.PROJECT_ROOT):
    correlation_matrix = inputData.corr(method='pearson', min_periods=50)
    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) >= corrValue:
                correlated_features.add(correlation_matrix.columns[i])

    if dropFeatures == False:
        # print('Features Uneleted')
        if plotMatrix == True:
            # print('Print Correlation Matrix')
            plot_functions.plot_correlation_matrix(correlation_matrix, savePlot, fileName, savePath)
        return correlated_features
    elif dropFeatures == True:
        if len(correlated_features) > corrValue:
            # print('Features Deleted')
            inputData.drop(labels=correlated_features, axis=1, inplace=True)
        if plotMatrix == True:
            # print('Print Correlation Matrix')
            drop_correlation_matrix = inputData.corr(method='pearson', min_periods=50)
            plot_functions.plot_correlation_matrix(drop_correlation_matrix, savePlot, fileName, savePath)