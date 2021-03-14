import apr_functions
import os

# genre_target_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
# genre_target_names = ['classical', 'disco', 'jazz', 'reggae', 'rock']
genre_target_names = []

genres = '10'
features_file_path = 'genres_'+genres
dataset_path = 'feature_csv/GTZAN_'+genres+'_Genres/data/'

for folder in os.listdir(os.path.join(features_file_path)):
    genre_target_names.append(str(folder))

print(genre_target_names)

for file_name in os.listdir(os.path.join(dataset_path)):
    Time_Domain_Audio_Features = ['tempo', 'rmse', 'zcr']
    Frequency_Domain_Audio_Features = ['chroma_stft', 'chroma_cqt', 'chroma_cens', 'spec_cent', 'spec_bw',
                                       'spec_contrast', 'rolloff']
    file_path = dataset_path+file_name
    image_file_name = file_name.replace('.csv', '')
    print(file_path)

    mfcc_value = file_name[0:2]
    print(mfcc_value)

    for num_mfcc in range(int(mfcc_value)):
        Frequency_Domain_Audio_Features.append('mfcc_' + str(num_mfcc))

    # Load Data
    X, y, df = apr_functions.load_data(file_path, 'min_max')

    # get train, validation, test splits
    X_train, X_test, y_train, y_test = apr_functions.prepare_datasets(X, y, 0.3)

    # Get Correlation Matrix and Plot
    apr_functions.getCorrelatedFeatures(X, 0.7, True, True, True, image_file_name)

    # Get PCA and Plot
    apr_functions.getPCA(X, y, 2, True, True, genre_target_names, image_file_name)

    # Get BMP aggregate results and PLot
    apr_functions.plot_BPM_Bar(df, True, image_file_name, len(genre_target_names))

    # Load Classifier Models
    load_models = apr_functions.getModel()

    # Calculate classifier results
    finalData = apr_functions.getResults(load_models, X_train, X_test, y_train, y_test, image_file_name, True, genre_target_names)
