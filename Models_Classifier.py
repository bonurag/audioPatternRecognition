import apr_functions
import os

genres = '10'
features_file_path = 'genres_'+genres
dataset_path = 'feature_csv/GTZAN_'+genres+'_Genres/data/'

Time_Domain_Audio_Features = ['tempo', 'energy', 'energy_entropy', 'rmse', 'zcr']
Frequency_Domain_Audio_Features = ['chroma_stft', 'chroma_cqt', 'chroma_cens', 'spec_cent', 'spec_bw', 'spec_contrast', 'rolloff']

def getGenres(features_file_path):
    genre_target_names = []
    for folder in os.listdir(os.path.join(features_file_path)):
        genre_target_names.append(str(folder))
    return genre_target_names


def initDataAndModel(input_file_path, features_file_path, features_to_drop, image_file_name):
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
    finalData = apr_functions.getResults(load_models, X_train, X_test, y_train, y_test, image_file_name, True, True, getGenres(features_file_path))


def startEvaluation(input_dataset_path, features_file_path, drop_Time_Features = [], drop_Frequency_Features= []):
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
            initDataAndModel(file_path, features_file_path, features_to_drop, image_file_name)
        else:
            if drop_Frequency_Features_Check:
                for num_mfcc in range(int(mfcc_value)):
                    features_to_drop.append('mfcc_' + str(num_mfcc))
            # print('ELSE drop_Frequency_Features_Check: ', file_path)
            # print(features_to_drop)
            image_file_name = image_file_name.replace(mfcc_value + 'MFCC_', '')
            initDataAndModel(file_path, features_file_path, features_to_drop, image_file_name)
            break


if __name__ == "__main__":
    startEvaluation(dataset_path, features_file_path, Time_Domain_Audio_Features, [])
