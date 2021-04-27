import apr_functions
import apr_functions_ul
import apr_constants
import os

features_file_path = apr_constants.FEATURES_FILE_PATH
dataset_path = apr_constants.DATASET_PATH


def get_genres():
    genre_target_names = []
    for folder in os.listdir(os.path.join(features_file_path)):
        genre_target_names.append(str(folder))

    if len(genre_target_names) == 0:
        genre_target_names = apr_constants.GENRE_TARGET_NAMES
    return genre_target_names


def init_data_and_model_sl(input_file_path, features_to_drop, image_file_name, save_path):
    # Load Data
    x, y, df = apr_functions.load_data(input_file_path, 'min_max', True, features_to_drop)

    # get train, validation, test splits
    x_train, x_test, y_train, y_test = apr_functions.prepare_datasets(x, y, 0.3)

    # Get Correlation Matrix and Plot
    apr_functions.getCorrelatedFeatures(x, 0.7, True, True, True, image_file_name, save_path)

    # Get PCA and Plot
    apr_functions.getPCA(x, y, 2, True, True, get_genres(), image_file_name, save_path)

    if 'tempo' in df:
        # Get BMP aggregate results and PLot
        apr_functions.plot_BPM_Bar(df, True, get_genres(), image_file_name, save_path)

    # Load Classifier Models
    load_models = apr_functions.getModel()

    # Calculate classifier results
    apr_functions.getResults(load_models, x_train, x_test, y_train, y_test, True, True,
                             get_genres(), image_file_name, save_path)


def init_data_and_model_ul(input_file_path, features_to_drop, image_file_name, save_path):
    # Load Data
    x, y, df = apr_functions_ul.load_data(input_file_path, 'min_max', False, features_to_drop)

    # Get Correlation Matrix and Plot
    apr_functions_ul.getCorrelatedFeatures(x, 0.7, True, True, True, image_file_name)

    # Get PCA Variance Ratio
    apr_functions_ul.getPCA_VarRatio_Plot(x, True, image_file_name, save_path)

    # Get K-means results
    labels, centroids = apr_functions_ul.runKmeans(x, 10, 20, image_file_name, save_path + apr_constants.MODEL)

    # Get PCA and Plot
    num_components = 2
    if num_components == 2:
        pca_data, centroids = apr_functions_ul.getPCAWithCentroids(x, y, num_components, True, True,
                                                                   get_genres(), image_file_name,
                                                                   save_path,
                                                                   centroids)
    elif num_components == 3:
        pca_data = apr_functions_ul.getPCAWithCentroids(x, y, num_components, True, True,
                                                        get_genres(), image_file_name,
                                                        save_path,
                                                        centroids)

    # Get Clusters Plot
    apr_functions_ul.plot_Clusters(pca_data, centroids, labels, apr_constants.COLORS_LIST,
                                   get_genres(), True, True, image_file_name, save_path)

    # Get K-means Confusion Matrix Plot
    apr_functions_ul.plot_confusion_matrix_kmeans(df, True, labels, get_genres(), image_file_name, save_path)

    # Get K-means Classification Report
    apr_functions_ul.plot_Classification_Report(df, True, labels, get_genres(), image_file_name,
                                                save_path + apr_constants.DATA)

    # Get ROC Plot
    apr_functions_ul.plot_roc(y.values, labels, 'K-Means', True, get_genres(), image_file_name, save_path)
    apr_functions_ul.plot_roc(y.values, labels, 'K-Means', True, get_genres(), image_file_name, save_path)

    # Get Silhouette Plot
    apr_functions_ul.plot_Silhouette(x, 2, 12, True, image_file_name, save_path)


def start_evaluation(input_dataset_path, drop_time_features=[], drop_frequency_features=[],
                     type_learning='SL'):
    features_to_drop = []
    drop_frequency_features_check = False
    drop_time_features_check = False

    try:
        if len(drop_time_features) > 0 and len(drop_frequency_features) == 0:
            features_to_drop = drop_time_features
            drop_time_features_check = True
        elif len(drop_time_features) == 0 and len(drop_frequency_features) > 0:
            features_to_drop = drop_frequency_features
            drop_frequency_features_check = True
        elif len(drop_time_features) == 0 and len(drop_frequency_features) == 0:
            drop_time_features_check = True
        elif len(drop_time_features) > 0 and len(drop_frequency_features) > 0:
            raise ValueError("Isn't possible drop all features, choose only one between time or frequency")
    except ValueError as ve:
        print(ve)

    for file_name in os.listdir(os.path.join(input_dataset_path)):
        file_path = input_dataset_path + file_name
        image_file_name = file_name.replace('.csv', '')
        mfcc_value = file_name[0:2]

        save_path = apr_constants.ROOT_SAVE_PATH

        if drop_time_features_check:
            result_folder = apr_constants.RESULTS_ALL_FEATURES if len(
                features_to_drop) == 0 else apr_constants.RESULTS_FREQUENCY_DOMAIN_FEATURES
            if type_learning == 'SL':
                save_path += apr_constants.SUPERVISED_LEARNING + str(result_folder) + str(mfcc_value) + '/'
                init_data_and_model_sl(file_path, features_to_drop, image_file_name, save_path)
            elif type_learning == 'UL':
                save_path += apr_constants.UNSUPERVISED_LEARNING + str(result_folder) + str(mfcc_value) + '/'
                init_data_and_model_ul(file_path, features_to_drop, image_file_name, save_path)
        else:
            if drop_frequency_features_check:
                for num_mfcc in range(int(mfcc_value)):
                    features_to_drop.append('mfcc_' + str(num_mfcc))
            image_file_name = image_file_name.replace(str(mfcc_value) + 'MFCC_', '')
            if type_learning == 'SL':
                save_path += apr_constants.SUPERVISED_LEARNING + apr_constants.RESULTS_TIME_DOMAIN_FEATURES
                init_data_and_model_sl(file_path, features_to_drop, image_file_name, save_path)
                break
            elif type_learning == 'UL':
                save_path += apr_constants.UNSUPERVISED_LEARNING + apr_constants.RESULTS_TIME_DOMAIN_FEATURES
                init_data_and_model_ul(file_path, features_to_drop, image_file_name, save_path)
                break


if __name__ == "__main__":
    start_evaluation(dataset_path, type_learning='SL')
