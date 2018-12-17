import numpy as np
import pandas as pd
from .. preprocessing.manipulation import convert_df_mat


def split_train_test_random(df, ratio_test=0.25):
    mat_spar = convert_df_mat(df)
    n_1 = mat_spar.nonzero()[0]
    n_2 = mat_spar.nonzero()[1]
    np_sparse = np.dstack((n_1, n_2))[0, :, :]

    df_sparse_nonzero_index = pd.DataFrame(np_sparse, columns=['user', 'movie'])
    df_group = df_sparse_nonzero_index.groupby('user')['movie'].apply(np.array).reset_index(name='movies')

    list_movie_2d = [np_movie for np_movie in df_group['movies']]

    list_test = list(map(lambda x: np.random.choice(
        x, size=int(len(x) / int(1 / ratio_test)), replace=False), list_movie_2d))
    list_train = []
    for i, movies_all in enumerate(list_movie_2d):
        movies_test = list_test[i]
        movies_train = list(set(movies_all) - set(movies_test))

        list_train.append(movies_train)

    lens_test = [len(movies) for movies in list_test]
    df_test = pd.DataFrame({'user_id': np.repeat(df_group['user'], lens_test),
                            'item_id': np.concatenate(list_test)})

    lens_train = [len(movies) for movies in list_train]
    df_train = pd.DataFrame({'user_id': np.repeat(df_group['user'], lens_train),
                             'item_id': np.concatenate(list_train)})

    df_train = pd.merge(df_train, df, on=['user_id', 'item_id'])
    df_test = pd.merge(df_test, df, on=['user_id', 'item_id'])

    return df_train, df_test

