import numpy as np
from sklearn.neighbors import NearestNeighbors
from ... decorator import check_operation_time
import multiprocessing
from multiprocessing import Pool
from functools import partial


def find_k_similar_users(id_user, df_spar, metric='cosine', k=5):
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(df_spar)

    distances, indices = model_knn.kneighbors(df_spar.iloc[id_user - 1, :].values.reshape(1, -1), n_neighbors=k + 1)
    similarities = 1 - distances.flatten()

    return similarities, indices.flatten()


def predict_userbased_user_item(id_user, id_item, df_spar, metric='cosine', k=5):
    similarities, indices = find_k_similar_users(id_user, df_spar, metric, k)
    series_au = df_spar.iloc[id_user - 1, :]
    mean_rating_au = series_au.sum() / len(series_au.nonzero()[0])

    sum_sim = sum_wtd = 0
    for i, indice in enumerate(indices):
        if indice + 1 == id_user:
            continue
        else:
            rating_ui = df_spar.iloc[indice, id_item - 1]
            if rating_ui != 0:
                series_u = df_spar.iloc[indice, :]
                sim = similarities[i]

                mean_rating_u = sum(series_u) / len(series_u.nonzero()[0])
                rating_diff = rating_ui - mean_rating_u
                product = rating_diff * sim
                sum_wtd += product
                sum_sim += sim

    if sum_sim == 0:
        prediction = 0
    else:
        prediction = mean_rating_au + (sum_wtd / sum_sim)

    return prediction


def predict_userbased_user_allitems(id_user, df_spar, metric='cosine', k=5):
    similarities, indices = find_k_similar_users(id_user, df_spar, metric, k)
    series_au = df_spar.iloc[id_user - 1, :]
    mean_rating_au = series_au.sum() / len(series_au.nonzero()[0])

    list_pred_user = []
    for id_item in range(1, df_spar.shape[1] + 1):
        sum_sim = sum_wtd = 0
        for i, indice in enumerate(indices):
            if indice + 1 == id_user:
                continue
            else:
                rating_ui = df_spar.iloc[indice, id_item - 1]
                if rating_ui != 0:
                    series_u = df_spar.iloc[indice, :]
                    sim = similarities[i]

                    mean_rating_u = sum(series_u) / len(series_u.nonzero()[0])
                    rating_diff = rating_ui - mean_rating_u
                    product = rating_diff * sim
                    sum_wtd += product
                    sum_sim += sim

        if sum_sim == 0:
            prediction = 0
        else:
            prediction = mean_rating_au + (sum_wtd / sum_sim)

        list_pred_user.append(prediction)

    return list_pred_user


def work(df_spar, metric, k, id_user):
    return id_user, predict_userbased_user_allitems(id_user, df_spar, metric=metric, k=k)


@check_operation_time
def recommend_userbased(df_spar, metric='cosine', k=5, cores=4):
    if cores == '*':
        cores = multiprocessing.cpu_count()

    p = Pool(processes=cores)
    iterable = list(range(1, df_spar.shape[0] + 1))
    func = partial(work, df_spar, metric, k)
    result_pred = p.map(func, iterable)
    p.close()

    list_pred_ratings_all = sorted(result_pred, key=lambda x: x[0])
    np_pred_ratings_all = np.array(list(map(lambda x: x[1], list_pred_ratings_all)))

    return np_pred_ratings_all

