import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from ... decorator import check_operation_time
import multiprocessing
from multiprocessing import Pool
from functools import partial
from scipy.spatial.distance import pdist, squareform


def get_sim_matrix(df_spar):
    m = np.array(df_spar)

    sim_matrix_sub_all = 1 - squareform(pdist(m.T, 'cosine'))
    sim_matrix_sub_all = pd.DataFrame(sim_matrix_sub_all)

    return sim_matrix_sub_all


def find_k_similar_items_by_simmatrix(sim_matrix, id_item, k=5):
    np_top_k = sim_matrix[id_item-1].sort_values(ascending=False)[:k+1]
    similarities = np_top_k.values
    indices = np.array(np_top_k.index)

    return similarities, indices


def find_k_similar_items(id_item, df_spar, metric='cosine', k=5):
    df_t = df_spar.T
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(df_t)

    similarities, indices = model_knn.kneighbors(df_t.iloc[id_item - 1, :].values.reshape(1, -1), n_neighbors=k + 1)
    similarities = 1 - similarities.flatten()

    return similarities, indices.flatten()


def predict_itembased_user_item(id_user, id_item, df_spar, metric='cosine', k=5):
    similarities, indices = find_k_similar_items(id_item, df_spar, metric, k)

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


def predict_itembased_user_allitems(id_user, df_spar, sim_matrix, k=5):
    list_prediction = []
    for id_item in range(1, df_spar.shape[1] + 1):
        similarities, indices = find_k_similar_items_by_simmatrix(sim_matrix, id_item, k)

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


def predict_itembased_item_allusers(id_item, df_spar, metric='cosine', k=5):
    similarities, indices = find_k_similar_items(id_item, df_spar, metric, k)

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


def work(df_spar, metric, k, id_item):
    return id_item, predict_itembased_item_allusers(id_item, df_spar, metric, k=k)


@check_operation_time
def recommend_itembased(df_spar, metric='cosine', k=5, cores=4):
    if cores == '*':
        cores = multiprocessing.cpu_count()

    p = Pool(processes=cores)
    iterable = list(range(1, df_spar.shape[1] + 1))
    func = partial(work, df_spar, metric, k)
    result_pred = p.map(func, iterable)
    p.close()

    list_pred_ratings_all = sorted(result_pred, key=lambda x: x[0])
    np_pred_ratings_all = np.array(list(map(lambda x: x[1], list_pred_ratings_all)))

    return np_pred_ratings_all.T

