import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import apr_constants
import common_functions
import plot_functions

from itertools import combinations


def load_data(data_path, normalization='std', remove_null_value=True, columns_to_drop=None):
    if columns_to_drop is None:
        columns_to_drop = []
    df = pd.read_csv(data_path)

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    if remove_null_value:
        for check in df.isnull().sum().sort_values(ascending=False):
            if check > 0:
                df = df.fillna(0)

    ord_enc = preprocessing.OrdinalEncoder()
    df['genre'] = ord_enc.fit_transform(df[['genre']])

    # Split df into x and Y
    target_col = 'genre'
    x = df.loc[:, df.columns != target_col]
    y = df.loc[:, target_col]

    x_cols = x.columns
    if normalization == 'std':
        # NORMALIZE X WITH STANDARD SCALER #
        resized_data = preprocessing.StandardScaler()
        np_scaled = resized_data.fit_transform(x)
    elif normalization == 'min_max':
        # NORMALIZE X WITH Min Max SCALER #
        resized_data = preprocessing.MinMaxScaler()
        np_scaled = resized_data.fit_transform(x)
    elif normalization is None:
        np_scaled = x

    x = pd.DataFrame(np_scaled, columns=x_cols)
    y = pd.DataFrame(y).fillna(0).astype(int)

    return x, y, df


def get_pca_var_ratio_plot(input_data, variance_ratio=0.8, save_plot=True, file_name=apr_constants.DEFAULT_FILE_NAME,
                           save_path=apr_constants.PROJECT_ROOT):
    cov_mat = np.cov(input_data.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    tot = sum(eigen_vals)

    # var_exp ratio is fraction of eigen_val to total sum
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]

    # calculate the cumulative sum of explained variances
    cum_var_exp = np.cumsum(var_exp)
    plt.figure(figsize=(15, 10))
    plt.bar(range(1, len(input_data.columns) + 1), var_exp, alpha=0.75, align='center',
            label='individual explained variance')
    plt.xticks(np.arange(1, len(input_data.columns) + 1, 1))
    plt.step(range(1, len(input_data.columns) + 1), cum_var_exp, where='mid', label='cumulative explained variance',
             c='red')

    # take the first occurrence of number of components for a specific value of variance #
    temp_variance_components = next(
        cum_var_exp[0] for cum_var_exp in enumerate(cum_var_exp) if cum_var_exp[1] > variance_ratio)
    first_occurrence_comp_spec_var = int(temp_variance_components) + 1
    print('first_occurrence_comp_spec_var: ', first_occurrence_comp_spec_var)

    plt.ylim(0, 1.1)
    plt.xlabel('Principal Components', fontsize=22)
    plt.ylabel('Explained variance ratio', fontsize=22)
    plt.legend(loc='best', prop={'size': apr_constants.LEGEND_SIZE})
    if save_plot:
        print('Save PCA Variance Ratio Plot')
        common_functions.check_create_directory(save_path)
        plt.savefig(save_path + file_name + ' - ' + 'PCA Variance Ratio Plot.jpg')
    plt.show()
    return first_occurrence_comp_spec_var


def get_pca_with_centroids(input_data, input_columns, num_of_components=1, plot_matrix=True, type_plot='2D',
                           save_plot=False,
                           target_names=None,
                           file_name=apr_constants.DEFAULT_FILE_NAME, save_path=apr_constants.PROJECT_ROOT,
                           centroids_value=None):
    if centroids_value is None:
        centroids_value = []
    if target_names is None:
        target_names = []
    use_data = input_data.copy()
    column_data = input_columns.copy()
    if num_of_components > 1:
        columns_components = []
        for col in range(num_of_components):
            columns_components.append('PC' + str(col + 1))
    elif num_of_components == 1:
        columns_components = ['PC1']

    # PCA Components #
    pca = PCA(n_components=num_of_components)
    pca_fit = pca.fit(use_data)
    principal_components = pca_fit.transform(use_data)
    principal_df = pd.DataFrame(data=principal_components, columns=columns_components)

    components_value = "Component" if num_of_components == 1 else "Components"
    print('PCA Variance Ratio For {} {}: {}'.format(num_of_components, components_value,
                                                    pca.explained_variance_ratio_.sum()))

    # Concatenate With Target Label
    concat_data = pd.concat([principal_df.reset_index(drop=True), column_data.reset_index(drop=True)], axis=1)

    # Transform Clusters Centroids
    c_transformed = pca_fit.transform(centroids_value)
    if type_plot == '2D':
        if plot_matrix:
            plot_functions.plot_pca(concat_data[['PC1', 'PC2', 'genre']], save_plot, target_names, file_name, save_path)
        return concat_data, c_transformed
    elif type_plot == '3D':
        if plot_matrix:
            plot_functions.plot_3d_pca(concat_data[['PC1', 'PC2', 'PC3', 'genre']], save_plot, file_name, save_path)
        return concat_data, c_transformed


def run_k_means(input_data, clusters_number=1, random_state=10, save_model=True,
                model_file_name=apr_constants.DEFAULT_FILE_NAME,
                save_path=apr_constants.PROJECT_ROOT):
    k_means = KMeans(clusters_number, random_state=random_state)
    k_means.fit(input_data)
    labels = k_means.labels_
    centroids = k_means.cluster_centers_
    predict_clusters = k_means.predict(input_data)
    if save_model:
        common_functions.check_create_directory(save_path + apr_constants.MODEL)
        common_functions.save_model(k_means, save_path + apr_constants.MODEL + model_file_name)
    return labels, predict_clusters, centroids, k_means


def get_permutation(input_list, num_element):
    permutations_list = list(combinations(input_list, num_element))
    to_eval = []
    temp_list = []
    correlated = [False, True]

    for x in permutations_list:
        for j in correlated:
            temp_list.append(x)
            temp_list.append(j)
            to_eval.append(temp_list)
            temp_list = []
    return to_eval
