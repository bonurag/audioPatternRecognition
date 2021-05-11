import apr_functions_sl
import apr_functions_ul
import apr_constants
import common_functions
import plot_functions
import os

dataset_path = apr_constants.DATASET_PATH


def init_data_and_model_sl(input_file_path, features_to_drop, image_file_name, save_path):
    # Load Data
    x, y, df = apr_functions_sl.load_data(input_file_path, 'min_max', True, features_to_drop)

    # get train, validation, test splits
    x_train, x_test, y_train, y_test = common_functions.prepare_datasets(x, y, 0.3, image_file_name, save_path)

    # Get Correlation Matrix and Plot
    common_functions.get_correlated_features(x, 0.7, True, True, True, image_file_name, save_path)

    # Get PCA and Plot
    apr_functions_sl.get_pca(x, y, 2, True, True, common_functions.get_genres(), image_file_name, save_path)

    if 'tempo' in df:
        # Get BMP aggregate results and PLot
        plot_functions.plot_bmp_bar(df, True, common_functions.get_genres(), image_file_name, save_path)

    # Load Classifier Model
    load_models = apr_functions_sl.get_model()

    # Calculate classifier results
    apr_functions_sl.get_results(load_models, x_train, x_test, y_train, y_test, True, True,
                                 common_functions.get_genres(), image_file_name, save_path)


def init_data_and_model_ul(input_file_path, features_to_drop, type_plot, image_file_name, save_path):
    # Load Data
    x, y, df = apr_functions_ul.load_data(input_file_path, 'min_max', False, features_to_drop)

    # Get Correlation Matrix and Plot
    common_functions.get_correlated_features(x, 0.7, True, True, True, image_file_name, save_path)

    # Get PCA Variance Ratio
    pca_components_to_use = apr_functions_ul.get_pca_var_ratio_plot(x, 0.8, True, image_file_name, save_path)

    # Get K-means results
    labels, predict_clusters, centroids, k_means = apr_functions_ul.run_k_means(x, 10, 20, image_file_name, save_path)

    # Get PCA and Plot
    if type_plot == '2D':
        pca_data, pca_centroids = apr_functions_ul.get_pca_with_centroids(x, y, pca_components_to_use, True, type_plot,
                                                                       True,
                                                                       common_functions.get_genres(), image_file_name,
                                                                       save_path,
                                                                       centroids)

        # Get Clusters Plot
        plot_functions.plot_clusters(pca_data[['PC1', 'PC2', 'genre']], pca_centroids, labels,
                                     apr_constants.COLORS_LIST, common_functions.get_genres(), True, True,
                                     image_file_name, save_path)

    elif type_plot == '3D':
        pca_data, pca_centroids = apr_functions_ul.get_pca_with_centroids(x, y, pca_components_to_use, True, type_plot,
                                                                       True,
                                                                       common_functions.get_genres(), image_file_name,
                                                                       save_path,
                                                                       centroids)

        # Get Clusters Plot
        plot_functions.plot_clusters(pca_data[['PC1', 'PC2', 'genre']], pca_centroids, labels,
                                     apr_constants.COLORS_LIST, common_functions.get_genres(), True, True,
                                     image_file_name, save_path)

    # Get K-means Confusion Matrix Plot
    plot_functions.plot_confusion_matrix_k_means(df, True, labels, common_functions.get_genres(), image_file_name,
                                                save_path)

    # Get K-means Classification Report
    plot_functions.plot_classification_report(df, True, labels, common_functions.get_genres(), image_file_name,
                                              save_path)

    # Get ROC Plot
    plot_functions.plot_roc(y.values, labels, 'K-Means', True, common_functions.get_genres(), image_file_name,
                            save_path, 'UL')

    # Get Silhouette Plot
    plot_functions.plot_silhouette(x, 2, 12, True, image_file_name, save_path)


def start_evaluation(input_dataset_path, drop_time_features=[], drop_frequency_features=[],
                     type_learning='SL'):
    features_to_drop = []
    drop_frequency_features_check = False
    drop_time_features_check = False
    ul_type_plot = '2D'

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
                init_data_and_model_ul(file_path, features_to_drop, ul_type_plot, image_file_name, save_path)
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
                init_data_and_model_ul(file_path, features_to_drop, ul_type_plot, image_file_name, save_path)
                break


if __name__ == "__main__":
    start_evaluation(dataset_path, drop_time_features=[], drop_frequency_features=[], type_learning='SL')
