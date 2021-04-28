import os
import time
import common_functions
import apr_constants
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt

MODEL_FILE_NAME = 'RF_20MFCC_10000_10GEN_GTZAN_Model'
TEST_FILE_NAME = '20MFCC_10000_10GEN_GTZAN_TestData'
MFCCS_FOLDER = '20'
genres = {i: apr_constants.GENRE_TARGET_NAMES[i] for i in range(0, len(apr_constants.GENRE_TARGET_NAMES))}
SAVE_ROOT = 'test_model/'
FILE_NAME = 'Prediction_Model.csv'

if __name__ == "__main__":
    start_time = time.time()
    if os.path.exists(SAVE_ROOT):
        shutil.rmtree(SAVE_ROOT)
        print('Folder Deleted!')

    model_path = apr_constants.ROOT_SAVE_PATH + apr_constants.SUPERVISED_LEARNING + apr_constants.RESULTS_FREQUENCY_DOMAIN_FEATURES + MFCCS_FOLDER + '/' + apr_constants.MODEL
    test_path = apr_constants.ROOT_SAVE_PATH + apr_constants.SUPERVISED_LEARNING + apr_constants.RESULTS_FREQUENCY_DOMAIN_FEATURES + MFCCS_FOLDER + '/' + apr_constants.TEST

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

    visual_data = data[['real_genre_num', 'predict_genre_num']].copy()
    visual_data.to_dict()

    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)
    data.to_csv(SAVE_ROOT + FILE_NAME, index=False)
    executionTime = time.time() - start_time
    print('Save CSV!')
    print('Execution Time: ', executionTime)
