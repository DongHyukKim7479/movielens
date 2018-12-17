import numpy as np
import pandas as pd
from scipy import sparse


def get_mapping_table(df, col_name, col_name_unique):
    i = df[col_name]
    np_item_unique = np.unique(np.array(i), axis=0)

    list_item = []
    for i, id_item in enumerate(np_item_unique):
        list_item.append((i + 1, id_item))

    df_mapping = pd.DataFrame(list_item, columns=[col_name_unique, col_name])

    return df_mapping


def change_col_with_merge(df, df_mapping, col_name_join, columns_new):
    df_join = pd.merge(df, df_mapping, on=[col_name_join])

    del df_join[col_name_join]
    columns_new_input = columns_new
    df_join = df_join[columns_new_input]
    df = df_join.rename(columns={'movie_unique_id': 'item_id'})

    return df


def convert_df_mat(df):
    r = df['rating'].astype(float)
    u = df['user_id'].astype(int)
    i = df['item_id'].astype(int)

    mat_spar = sparse.csr_matrix(
        (r, (u, i)),
        dtype=np.float
    )

    return mat_spar


def delete_zero_idx_mat(mat_spar):
    df_spar = pd.DataFrame(mat_spar.toarray())
    df_spar = df_spar.drop(df_spar.index[0], axis=0)
    df_spar = df_spar.drop(df_spar.columns[0], axis=1)

    return sparse.csr_matrix(df_spar)


def make_df_from_mat(mat_spar, num_movie):
    df_spar = pd.DataFrame(mat_spar.toarray())

    if df_spar.shape[1] < num_movie:
        num_row = df_spar.shape[0]
        num_col = df_spar.shape[1]
        num_diff = num_movie - num_col

        np_add = np.zeros([num_row, num_diff])
        df_add = pd.DataFrame(np_add, index=np.arange(num_row),
                              columns=(num_col + np.arange(num_diff)))

        df_spar = pd.concat([df_spar, df_add], axis=1)

    return df_spar


def convert_df_spar(df, max_id_item):
    mat = delete_zero_idx_mat(convert_df_mat(df))
    df_spar = make_df_from_mat(mat, max_id_item)

    return df_spar

