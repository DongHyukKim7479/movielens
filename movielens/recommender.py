import numpy as np
import pandas as pd
from .preprocessing.manipulation import convert_df_mat
from .preprocessing.manipulation import convert_df_spar
from .preprocessing.manipulation import get_mapping_table
from .model.collaborative.user_based import predict_userbased_user_allitems
from .model.collaborative.item_based import get_sim_matrix
from .model.collaborative.item_based import predict_itembased_user_allitems
from .model.collaborative.item_based_adj import get_sim_matrix_sub_all
from .model.collaborative.item_based_adj import get_sim_matrix_sub_exist
from .model.collaborative.item_based_adj import predict_itembased_user_allitems_adjcos
from .model.matrix_factorization.svd import model_svd
from .decorator import check_operation_time


# seen movie
def get_list_seen_movie_id(df, id_user):
    mat_spar = convert_df_mat(df)
    sparse_nonzero = mat_spar.nonzero()
    list_id_movie_seen = []
    for i, id_user_exist_rating in enumerate(sparse_nonzero[0]):
        if id_user_exist_rating == id_user:
            list_id_movie_seen.append(sparse_nonzero[1][i])

    return list_id_movie_seen


def get_df_movie_from_id(list_id_movie, df_rating, df_movies):
    df_mapping = get_mapping_table(df_rating, 'item_id', 'movie_unique_id')
    np_insert = np.array(list_id_movie).reshape(-1, 1)
    df_id_movie = pd.DataFrame(np_insert, columns=['movie_unique_id'])
    df_joined = pd.merge(df_id_movie, df_mapping, on=['movie_unique_id'])
    df_movie = pd.merge(df_joined, df_movies, on=['item_id']).drop(columns=['movie_unique_id'])

    return df_movie


def get_seen_movie(df_rating, df_movies, id_user):
    list_seen_movie = get_list_seen_movie_id(df_rating, id_user)

    return get_df_movie_from_id(list_seen_movie, df_rating, df_movies)


# recommend item
def get_recomm_list_user(df_spar, id_user, flag, dict_args):
    list_pred_rank = []
    if flag == 'user-based':
        list_pred_rank = predict_userbased_user_allitems(id_user, df_spar, **dict_args)
    elif flag == 'item-based':
        sim_matrix = get_sim_matrix(df_spar)
        list_pred_rank = predict_itembased_user_allitems(id_user, df_spar, sim_matrix, **dict_args)
    elif flag == 'item-based-adjall':
        sim_matrix = get_sim_matrix_sub_all(df_spar)
        list_pred_rank = predict_itembased_user_allitems_adjcos(id_user, df_spar, sim_matrix, **dict_args)
    elif flag == 'item-based-adjexist':
        sim_matrix = get_sim_matrix_sub_exist(df_spar)
        list_pred_rank = predict_itembased_user_allitems_adjcos(id_user, df_spar, sim_matrix, **dict_args)
    elif flag == 'svd':
        np_spar_result = model_svd(df_spar, **dict_args)
        list_pred_rank = np_spar_result[id_user - 1]

    return list_pred_rank


def recommend_item(df, id_user, flag, dict_args):
    df_spar = convert_df_spar(df, df.item_id.max())
    list_pred_rank = get_recomm_list_user(df_spar, id_user, flag, dict_args)
    list_seen_movie = get_list_seen_movie_id(df, id_user)

    list_recommendation = []
    for i, rank in enumerate(list_pred_rank):
        if rank >= 3:
            if i + 1 not in list_seen_movie:
                list_recommendation.append(i + 1)

    return list_recommendation


@check_operation_time
def get_recomm_movie(df, df_movies, id_user, flag='user-based', dict_args={}):
    list_recomm = recommend_item(df, id_user, flag, dict_args)[:5]

    return get_df_movie_from_id(list_recomm, df, df_movies)

