import numpy as np
import pandas as pd
from scipy import sparse
from math import sqrt
from sklearn.metrics import mean_squared_error
from .. preprocessing.manipulation import convert_df_mat
from .. preprocessing.manipulation import convert_df_spar


def get_dataset(df, k_fold, offset):
    mat_spar = convert_df_mat(df)
    n_1 = mat_spar.nonzero()[0]
    n_2 = mat_spar.nonzero()[1]
    np_sparse = np.dstack((n_1, n_2))[0, :, :]

    df_sparse_nonzero_index = pd.DataFrame(np_sparse, columns=['user', 'movie'])
    df_group = df_sparse_nonzero_index.groupby('user')['movie'].apply(np.array).reset_index(name='movies')

    list_movie_2d = [np_movie for np_movie in df_group['movies']]

    list_test = list(map(lambda x: np.array_split(x, k_fold)[offset], list_movie_2d))
    list_train = []
    for i, movies_all in enumerate(list_movie_2d):
        movies_test = list_test[i]
        movies_train = list(set(movies_all) - set(movies_test))

        list_train.append(movies_train)

    lens_train = [len(movies) for movies in list_train]
    df_train_without_rating = pd.DataFrame({'user_id': np.repeat(df_group['user'], lens_train),
                                            'item_id': np.concatenate(list_train)})

    lens_test = [len(movies) for movies in list_test]
    df_test_without_rating = pd.DataFrame({'user_id': np.repeat(df_group['user'], lens_test),
                                           'item_id': np.concatenate(list_test)})

    df_train = pd.merge(df_train_without_rating, df, on=['user_id', 'item_id'])
    df_test = pd.merge(df_test_without_rating, df, on=['user_id', 'item_id'])

    return df_train, df_test


def _get_duplicate_2d_array(np_2d_1, np_2d_2):
    nrows, ncols = np_2d_1.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [np_2d_1.dtype]}

    np_duplicate_2d = np.intersect1d(np_2d_1.view(dtype), np_2d_2.view(dtype))
    np_duplicate_2d = np_duplicate_2d.view(np_2d_1.dtype).reshape(-1, 2)

    return np_duplicate_2d


def get_rmse(mat_actual, mat_pred):
    np_exist_index_test = np.c_[mat_actual.nonzero()[0], mat_actual.nonzero()[1]]
    np_exist_index_pred = np.c_[mat_pred.nonzero()[0], mat_pred.nonzero()[1]]

    np_duplicate_2d = _get_duplicate_2d_array(np_exist_index_test, np_exist_index_pred)
    tupl_exist_index = np_duplicate_2d[:, 0], np_duplicate_2d[:, 1]
    actu = mat_actual[tupl_exist_index]
    pred = mat_pred[tupl_exist_index]

    return sqrt(mean_squared_error(actu, pred))


def cross_validation(df, model, dict_args, k_fold=5):
    list_validations = []
    max_id_movies = df.item_id.max()
    for offset in range(k_fold):
        df_train_set, df_validation_set = get_dataset(df, k_fold, offset)
        df_spar_train_set = convert_df_spar(df_train_set, max_id_movies)
        df_spar_validation_set = convert_df_spar(df_validation_set, max_id_movies)
        np_result = model(df_spar_train_set, **dict_args)

        mat_spar_test = sparse.csr_matrix(df_spar_validation_set)
        mat_spar_predict = sparse.csr_matrix(np_result)

        list_validations.append(get_rmse(mat_spar_test, mat_spar_predict))

    avg_rmse = np.mean(list_validations)
    return list_validations, avg_rmse

