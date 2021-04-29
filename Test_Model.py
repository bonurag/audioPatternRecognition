import os
import time

import pandas as pd
import numpy as np

import shutil

import plot_functions
import common_functions
import apr_constants

MODEL_FILE_NAME = 'RF_20MFCC_10000_10GEN_GTZAN_Model'
TEST_FILE_NAME = '20MFCC_10000_10GEN_GTZAN_TestData'
MFCCS_FOLDER = '20'

SAVE_ROOT = apr_constants.ROOT_SAVE_PATH_TEST_MODEL

FILE_NAME = 'Test_Model_Predictions.csv'
FILE_NAME_COUNT = 'Test_Model_Predictions_Count.csv'

genres = {i: apr_constants.GENRE_TARGET_NAMES[i] for i in range(0, len(apr_constants.GENRE_TARGET_NAMES))}

if __name__ == "__main__":
    start_time = time.time()
    if os.path.exists(SAVE_ROOT):
        shutil.rmtree(SAVE_ROOT)
        print('Folder Deleted!')

    model_path = apr_constants.ROOT_SAVE_PATH + apr_constants.SUPERVISED_LEARNING + apr_constants.RESULTS_ALL_FEATURES + MFCCS_FOLDER + '/' + apr_constants.MODEL
    test_path = apr_constants.ROOT_SAVE_PATH + apr_constants.SUPERVISED_LEARNING + apr_constants.RESULTS_ALL_FEATURES + MFCCS_FOLDER + '/' + apr_constants.TEST

    # Load the model from disk
    load_model = common_functions.load_model(model_path, MODEL_FILE_NAME)

    # Load the test data from disk
    x_test, y_test = common_functions.load_test_data(test_path, TEST_FILE_NAME)

    result = load_model.predict(x_test)

    data = pd.DataFrame(columns=['real_genre_num', 'predict_genre_num', 'real_genre_text', 'predict_genre_text'])
    data['real_genre_num'] = y_test.astype(int)
    data['predict_genre_num'] = result.astype(int)

    comparison_column = np.where(data['real_genre_num'] == data['predict_genre_num'], True, False)
    data['check'] = comparison_column

    data['real_genre_text'] = data['real_genre_num'].replace(genres)
    data['predict_genre_text'] = data['predict_genre_num'].replace(genres)

    visual_data = pd.DataFrame()
    visual_data[['Genre', 'Real_Value']] = data[['real_genre_text', 'predict_genre_text']].groupby(['real_genre_text'], as_index=False).count()
    visual_data[['Genre', 'Predict_Value']] = data[['real_genre_text', 'predict_genre_text']].groupby(['predict_genre_text'], as_index=False).count()

    plot_functions.plot_predictions_compare(None, y_test, result, common_functions.get_genres(), 'Random_Forest',
                                            True, 'Complex Test Model Evaluation', SAVE_ROOT+apr_constants.PLOT)

    plot_functions.plot_predictions_simple_compare(visual_data, common_functions.get_genres(),
                                                   True, 'Simple Test Model Evaluation', SAVE_ROOT+apr_constants.PLOT)

    if not os.path.exists(SAVE_ROOT+apr_constants.DATA):
        os.makedirs(SAVE_ROOT+apr_constants.DATA)
    data.to_csv(SAVE_ROOT+apr_constants.DATA+FILE_NAME, index=False)
    visual_data.to_csv(SAVE_ROOT+apr_constants.DATA+FILE_NAME_COUNT, index=False)
    executionTime = time.time() - start_time
    print('Save CSV!')
    print('Execution Time: ', executionTime)
