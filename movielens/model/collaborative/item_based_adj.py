import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from ... decorator import check_operation_time
import multiprocessing
from multiprocessing import Pool
from functools import partial


def get_sim_matrix_sub_all(df_spar):
    m = np.array(df_spar)
    m_u = m.mean(axis=1)
    m_adj = m - m_u[:, None]

    sim_matrix_sub_all = 1 - squareform(pdist(m_adj.T, 'cosine'))
    sim_matrix_sub_all = pd.DataFrame(sim_matrix_sub_all)

    return sim_matrix_sub_all


def get_sim_matrix_sub_exist(df_spar):
    m = np.array(df_spar)
    list_mean = []
    for rating_user in m:
        np_idx_nonzero = rating_user.nonzero()[0]
        mean = np.mean(rating_user[np_idx_nonzero])
        list_mean.append(mean)
    m_u = np.array(list_mean)

    m_adj = np.zeros([df_spar.shape[0], df_spar.shape[1]])

    for row, np_ratings in enumerate(m):
        np_idx_nonzero = np_ratings.nonzero()[0]
        m_adj[row][np_idx_nonzero] = np_ratings[np_idx_nonzero] - m_u[row]

    sim_matrix_sub_exist = 1 - squareform(pdist(m_adj.T, 'cosine'))

    mask = np.isnan(sim_matrix_sub_exist)
    sim_matrix_sub_exist[mask] = 0

    sim_matrix_sub_exist = pd.DataFrame(sim_matrix_sub_exist)

    return sim_matrix_sub_exist


def find_similar_k_items_adjcos(sim_matrix, id_item, k=5):
    np_top_k = sim_matrix[id_item-1].sort_values(ascending=False)[:k+1]
    similarities = np_top_k.values
    indices = np.array(np_top_k.index)

    return similarities, indices


def predict_itembased_user_item_adjcos(id_user, id_item, df_spar, sim_matrix, k=5):
    similarities, indices = find_similar_k_items_adjcos(sim_matrix, id_item, k)

    sum_wtd = sum_sim = 0
    for i, indice in enumerate(indices):
        if indice + 1 == id_item:
            continue
        else:
            rating_ui = df_spar.iloc[id_user - 1, indice]
            if rating_ui != 0:
                sim_item = similarities[i]

                product = rating_ui * sim_item
                sum_wtd += product
                sum_sim += sim_item

    if sum_sim == 0:
        prediction = 0
    else:
        prediction = (sum_wtd / sum_sim)

    return prediction


def predict_itembased_user_allitems_adjcos(id_user, df_spar, sim_matrix, k=5):
    list_prediction = []
    for id_item in range(1, df_spar.shape[1] + 1):
        similarities, indices = find_similar_k_items_adjcos(sim_matrix, id_item, k)

        sum_wtd = sum_sim = 0
        for i, indice in enumerate(indices):
            if indice + 1 == id_item:
                continue
            else:
                rating_ui = df_spar.iloc[id_user - 1, indice]
                if rating_ui != 0:
                    sim_item = similarities[i]

                    product = rating_ui * sim_item
                    sum_wtd += product
                    sum_sim += sim_item

        if sum_sim == 0:
            prediction = 0
        else:
            prediction = (sum_wtd / sum_sim)
        list_prediction.append(prediction)

    return list_prediction


def predict_itembased_item_allusers_adjcos(id_item, df_spar, sim_matrix, k=5):
    similarities, indices = find_similar_k_items_adjcos(sim_matrix, id_item, k)

    list_pred_all_rating = []
    for id_user in range(1, df_spar.shape[0] + 1):
        sum_wtd = sum_sim = 0
        for i, indice in enumerate(indices):
            if indice + 1 == id_item:
                continue
            else:
                rating_ui = df_spar.iloc[id_user - 1, indice]
                if rating_ui != 0:
                    sim_item = similarities[i]

                    product = rating_ui * sim_item
                    sum_wtd += product
                    sum_sim += sim_item

        if sum_sim == 0:
            prediction = 0
        else:
            prediction = (sum_wtd / sum_sim)

        list_pred_all_rating.append(prediction)

    return list_pred_all_rating


def work_all(df_spar, sim_matrix, k, id_item):
    return id_item, predict_itembased_item_allusers_adjcos(id_item, df_spar, sim_matrix, k=k)


@check_operation_time
def recommend_itembased_adjcosine_all(df_spar, k=5, cores=4):
    if cores == '*':
        cores = multiprocessing.cpu_count()

    sim_matrix = get_sim_matrix_sub_all(df_spar)

    p = Pool(processes=cores)
    iterable = list(range(1, df_spar.shape[1] + 1))
    func = partial(work_all, df_spar, sim_matrix, k)
    result_pred = p.map(func, iterable)
    p.close()

    list_pred_ratings_all = sorted(result_pred, key=lambda x: x[0])
    np_pred_ratings_all = np.array(list(map(lambda x: x[1], list_pred_ratings_all)))

    return np_pred_ratings_all.T


def work_exist(df_spar, sim_matrix, k, id_item):
    return id_item, predict_itembased_item_allusers_adjcos(id_item, df_spar, sim_matrix, k=k)


@check_operation_time
def recommend_itembased_adjcosine_exist(df_spar, k=5, cores=4):
    if cores == '*':
        cores = multiprocessing.cpu_count()

    sim_matrix = get_sim_matrix_sub_exist(df_spar)

    p = Pool(processes=cores)
    iterable = list(range(1, df_spar.shape[1] + 1))
    func = partial(work_exist, df_spar, sim_matrix, k)
    result_pred = p.map(func, iterable)
    p.close()

    list_pred_ratings_exist = sorted(result_pred, key=lambda x: x[0])
    list_pred_ratings_exist = np.array(list(map(lambda x: x[1], list_pred_ratings_exist)))

    return list_pred_ratings_exist.T

