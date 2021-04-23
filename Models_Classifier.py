import apr_functions
import apr_functions_ul
import os

genres = '10'
features_file_path = 'genres_' + genres
dataset_path = 'feature_csv/GTZAN_' + genres + '_Genres/data/'

Time_Domain_Audio_Features = ['tempo', 'energy', 'energy_entropy', 'rmse', 'zcr']
Frequency_Domain_Audio_Features = ['chroma_stft', 'chroma_cqt', 'chroma_cens', 'spec_cent', 'spec_bw', 'spec_contrast',
                                   'rolloff']


def getGenres(features_file_path):
    genre_target_names = []
    for folder in os.listdir(os.path.join(features_file_path)):
        genre_target_names.append(str(folder))
    return genre_target_names


def initDataAndModelSL(input_file_path, features_file_path, features_to_drop, image_file_name):
    # Load Data
    X, y, df = apr_functions.load_data(input_file_path, 'min_max', features_to_drop)

    # get train, validation, test splits
    X_train, X_test, y_train, y_test = apr_functions.prepare_datasets(X, y, 0.3)

    # Get Correlation Matrix and Plot
    apr_functions.getCorrelatedFeatures(X, 0.7, True, True, True, image_file_name)

    # Get PCA and Plot
    apr_functions.getPCA(X, y, 2, True, True, getGenres(features_file_path), image_file_name)

    if 'tempo' in df:
        # Get BMP aggregate results and PLot
        apr_functions.plot_BPM_Bar(df, True, image_file_name, len(getGenres(features_file_path)))

    # Load Classifier Models
    load_models = apr_functions.getModel()

    # Calculate classifier results
    finalData = apr_functions.getResults(load_models, X_train, X_test, y_train, y_test, image_file_name, True, True,
                                         getGenres(features_file_path))


def initDataAndModelUL(input_file_path, features_file_path, features_to_drop, image_file_name):
    # Load Data
    X, y, df = apr_functions_ul.load_data(input_file_path, 'min_max', False, features_to_drop)

    # Get Correlation Matrix and Plot
    apr_functions_ul.getCorrelatedFeatures(X, 0.7, True, True, True, image_file_name)

    # Get PCA Variance Ratio
    apr_functions_ul.getPCA_VarRatio_Plot(X, True, image_file_name)

    # Get K-means results
    labels, centroids = apr_functions_ul.runKmeans(X, 10, 20)

    # Get PCA and Plot
    num_components = 2
    if num_components == 2:
        pca_data, centroids = apr_functions_ul.getPCAWithCentroids(X, y, num_components, True, True,
                                                                   getGenres(features_file_path), image_file_name,
                                                                   centroids)
    elif num_components == 3:
        pca_data = apr_functions_ul.getPCAWithCentroids(X, y, num_components, True, True, getGenres(features_file_path),
                                                        image_file_name, centroids)

    # Get Clusters Plot
    apr_functions_ul.plot_Clusters(pca_data, centroids, labels, apr_functions_ul.colors_list,
                                   getGenres(features_file_path), True, True, image_file_name)

    # Get K-means Confusion Matrix Plot
    apr_functions_ul.plot_confusion_matrix_kmeans(df, True, labels, getGenres(features_file_path), image_file_name)

    # Get K-means Classification Report
    apr_functions_ul.plot_Classification_Report(df, True, labels, getGenres(features_file_path), image_file_name)

    # Get ROC Plot
    apr_functions_ul.plot_roc(y.values, labels, 'K-Means', True, getGenres(features_file_path), image_file_name)

    # Get Silhouette Plot
    apr_functions_ul.plot_Silhouette(X, 2, 12, True, image_file_name)


def startEvaluation(input_dataset_path, features_file_path, drop_Time_Features=[], drop_Frequency_Features=[],
                    type_learning='SL'):
    features_to_drop = []
    drop_Frequency_Features_Check = False
    drop_Time_Features_Check = False

    try:
        if len(drop_Time_Features) > 0 and len(drop_Frequency_Features) == 0:
            features_to_drop = drop_Time_Features
            drop_Time_Features_Check = True
        elif len(drop_Time_Features) == 0 and len(drop_Frequency_Features) > 0:
            features_to_drop = drop_Frequency_Features
            drop_Frequency_Features_Check = True
        elif len(drop_Time_Features) == 0 and len(drop_Frequency_Features) == 0:
            drop_Time_Features_Check = True
        elif len(drop_Time_Features) > 0 and len(drop_Frequency_Features) > 0:
            raise ValueError("Isn't possible drop all features, choose only one between time or frequency")
    except ValueError as ve:
        print(ve)

    for file_name in os.listdir(os.path.join(input_dataset_path)):
        file_path = input_dataset_path + file_name
        image_file_name = file_name.replace('.csv', '')
        mfcc_value = file_name[0:2]
        print('MFCCs Value: ', mfcc_value)

        if drop_Time_Features_Check:
            # print('IF drop_Time_Features_Check: ', file_path)
            # print(features_to_drop)
            if type_learning == 'SL':
                initDataAndModelSL(file_path, features_file_path, features_to_drop, image_file_name)
            elif type_learning == 'UL':
                initDataAndModelUL(file_path, features_file_path, features_to_drop, image_file_name)
        else:
            if drop_Frequency_Features_Check:
                for num_mfcc in range(int(mfcc_value)):
                    features_to_drop.append('mfcc_' + str(num_mfcc))
            # print('ELSE drop_Frequency_Features_Check: ', file_path)
            # print(features_to_drop)
            image_file_name = image_file_name.replace(mfcc_value + 'MFCC_', '')
            if type_learning == 'SL':
                initDataAndModelSL(file_path, features_file_path, features_to_drop, image_file_name)
                break
            elif type_learning == 'UL':
                initDataAndModelUL(file_path, features_file_path, features_to_drop, image_file_name)
                break

if __name__ == "__main__":
    startEvaluation(dataset_path, features_file_path, type_learning='UL')
