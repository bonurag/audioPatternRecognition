import os

GENRE_TARGET_NAMES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
COLORS_LIST = {'red', 'blue', 'green', 'purple', 'orange', 'deeppink', 'skyblue', 'aquamarine', 'teal', 'dodgerblue'}
ROC_COLOR_LIST = {'blue', 'red', 'green', 'darkorange', 'chocolate', 'lime', 'deepskyblue', 'silver', 'tomato',
                  'purple'}

TIME_DOMAIN_FEATURES = ['tempo', 'energy', 'energy_entropy', 'rmse', 'zcr']
FREQUENCY_DOMAIN_FEATURES = ['chroma_stft', 'chroma_cqt', 'chroma_cens', 'spec_cent', 'spec_bw', 'spec_contrast',
                             'rolloff']
TITLE_FONT_SIZE = 30
LEGEND_SIZE = 16
DEFAULT_FILE_NAME = 'Default File Name'
DEFAULT_CLASSIFIER_NAME = 'Default Classifier Name'

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))+'/'
GENRES = '10'
FEATURES_FILE_PATH = 'genres_' + GENRES
DATASET_PATH = 'feature_csv/data/'

RESULTS_ALL_FEATURES = 'results_all_features/'
RESULTS_FREQUENCY_DOMAIN_FEATURES = 'results_frequency_domain_features/'
RESULTS_TIME_DOMAIN_FEATURES = 'results_time_domain_features/'

MODEL = 'Model/'
DATA = 'Data/'
TEST = 'Test/'
PLOT = 'Plot/'

SUPERVISED_LEARNING = 'Supervised_Learning/'
UNSUPERVISED_LEARNING = 'Unsupervised_Learning/'

ROOT_SAVE_PATH = 'export_results/'
ROOT_SAVE_PATH_TEST_MODEL = 'test_model/'
ROOT_SAVE_PATH_NOTEBOOKS = 'Notebooks_Results/'

# Constants and Path definitions for feature extraction utils

FEATURES_DATASET_PATH = 'genres_collections_new'
FEATURES_EXCLUDE_FOLDER = {'results'}
FEATURES_SAVE_ROOT = FEATURES_DATASET_PATH + '/results/'
FEATURES_MFCC_VALUE = 13
FEATURES_SAMPLE_DURATION = 30
