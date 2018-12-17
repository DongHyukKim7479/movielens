import numpy as np
from ... decorator import check_operation_time


@check_operation_time
def model_svd(df_spar, k_input, val_adj):
    np_spar = np.array(df_spar)
    mean_user_ratings = np.mean(np_spar, axis=1)
    np_train_demeaned = np_spar - mean_user_ratings.reshape(-1, 1)

    from scipy.sparse.linalg import svds
    u, sigma, vt = svds(np_train_demeaned, k=k_input)

    sigma = np.diag(sigma)

    np_pred_ratings_all_users = np.dot(np.dot(u, sigma), vt) + mean_user_ratings.reshape(-1, 1) + val_adj

    return np_pred_ratings_all_users

