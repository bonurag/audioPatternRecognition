import pickle
import os

from pathlib import Path

from sklearn.model_selection import train_test_split

import plot_functions
import apr_constants

features_file_path = apr_constants.FEATURES_FILE_PATH


def get_genres():
    genre_target_names = []
    for folder in os.listdir(os.path.join(features_file_path)):
        genre_target_names.append(str(folder))

    if len(genre_target_names) == 0:
        genre_target_names = apr_constants.GENRE_TARGET_NAMES
    return genre_target_names


def save_model(input_classifier, save_model_name):
    print('Save Model {} - {} '.format(input_classifier, save_model_name))
    filename = save_model_name + '_Model.sav'
    pickle.dump(input_classifier, open(filename, 'wb'))


def save_test_data(x_test, y_test, save_data_name):
    tuple_objects = (x_test, y_test)
    filename = save_data_name + '_TestData.pkl'
    pickle.dump(tuple_objects, open(filename, 'wb'))


def load_test_data(input_path, input_file_name):
    if input_file_name.endswith('.pkl'):
        input_file_name = input_file_name[0:len(input_file_name) - 4]
    file_to_load = input_path + input_file_name + '.pkl'
    pickled_x_test, pickled_y_test = pickle.load(open(file_to_load, 'rb'))
    print('Test Files loading complete successfully!')
    return pickled_x_test, pickled_y_test


def load_model(input_path, input_file_name):
    if input_file_name.endswith('.sav'):
        input_file_name = input_file_name[0:len(input_file_name) - 4]
    model_to_load = input_path + input_file_name + '.sav'
    model = pickle.load(open(model_to_load, 'rb'))
    print('Model loading complete successfully!')
    return model


def prepare_datasets(x, y, test_size=0.3, file_name=apr_constants.DEFAULT_FILE_NAME,
                     save_path=apr_constants.PROJECT_ROOT):
    # create train, test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=100)
    check_create_directory(save_path + apr_constants.TEST)
    save_test_data(x_test, y_test, save_path + apr_constants.TEST + file_name)
    return x_train, x_test, y_train, y_test


def check_create_directory(input_path):
    if not os.path.exists(input_path):
        Path(input_path).mkdir(parents=True, exist_ok=True)


def get_correlated_features(input_data, corr_value=0.9, drop_features=True, plot_matrix=True, save_plot=False,
                            file_name=apr_constants.DEFAULT_FILE_NAME, save_path=apr_constants.PROJECT_ROOT):
    correlation_matrix = input_data.corr(method='pearson', min_periods=50)
    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) >= corr_value:
                correlated_features.add(correlation_matrix.columns[i])

    if not drop_features:
        if plot_matrix:
            plot_functions.plot_correlation_matrix(correlation_matrix, save_plot, file_name, save_path)
        return correlated_features
    elif drop_features:
        if len(correlated_features) > corr_value:
            input_data.drop(labels=correlated_features, axis=1, inplace=True)
        if plot_matrix:
            drop_correlation_matrix = input_data.corr(method='pearson', min_periods=50)
            plot_functions.plot_correlation_matrix(drop_correlation_matrix, save_plot, file_name, save_path)
