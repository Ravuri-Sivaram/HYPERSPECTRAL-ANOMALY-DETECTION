# coding=utf-8
import numpy as np
from scipy import linalg

def compute_lrsr(background_dict, anomaly_dict, data_matrix, beta, lamda):
    """
    Compute Low-Rank and Sparse Representation (LRSR)
    
    :param background_dict: background dictionary
    :param anomaly_dict: anomaly dictionary
    :param data_matrix: normalized data (bands x pixels)
    :param beta: regularization parameter for sparsity
    :param lamda: regularization parameter for noise
    :return:
        lowrank_coef: low-rank coefficients (Z)
        noise_matrix: noise component (E)
        sparse_coef: sparse coefficients (S)
    """
    data_rows, data_cols = data_matrix.shape
    b_rows, b_cols = background_dict.shape
    a_rows, a_cols = anomaly_dict.shape

    ILRR = np.eye(b_cols)
    ISRC = np.eye(a_cols)

    lowrank_coef = np.zeros((b_cols, data_cols))
    J = np.zeros((b_cols, data_cols))
    noise_matrix = np.zeros((data_rows, data_cols))
    sparse_coef = np.zeros((a_cols, data_cols))
    L = np.zeros((a_cols, data_cols))

    Y1 = np.zeros((data_rows, data_cols))
    Y2 = np.zeros((b_cols, data_cols))
    Y3 = np.zeros((a_cols, data_cols))

    mu = 1e-4
    mu_max = 1e10
    p = 1.1
    tol = 1e-6
    iteration = 1

    inv_Z = np.linalg.inv(background_dict.T @ background_dict + ILRR)
    inv_S = np.linalg.inv(anomaly_dict.T @ anomaly_dict + ISRC)

    while iteration < 500:
        print(f"Iteration: {iteration}")

        # --- Update J ---
        operator1 = 1 / mu
        tmpJ = lowrank_coef + Y2 / mu
        U, sigma, Vt = linalg.svd(tmpJ, full_matrices=False)
        evp = sigma[sigma > operator1].shape[0]
        if evp >= 1:
            sigma[0:evp] -= operator1
            SigmaM = np.diag(sigma[0:evp])
        else:
            evp = 1
            SigmaM = np.zeros((1, 1))  # <-- fixed
            J = U[:, :evp] @ SigmaM @ Vt[:evp, :]


        # --- Update noise matrix E ---
        operator3 = lamda / mu
        tmpE = data_matrix - background_dict @ lowrank_coef - anomaly_dict @ sparse_coef + Y1 / mu
        for i in range(tmpE.shape[1]):
            norm_val = linalg.norm(tmpE[:, i])
            if norm_val > operator3:
                noise_matrix[:, i] = ((norm_val - operator3) / norm_val) * tmpE[:, i]
            else:
                noise_matrix[:, i] = 0

        # --- Update L ---
        operator2 = beta / mu
        tmpL = sparse_coef + Y3 / mu
        tmpL[tmpL > operator2] -= operator2
        tmpL[tmpL < -operator2] += operator2
        tmpL[(tmpL >= -operator2) & (tmpL <= operator2)] = 0
        L = tmpL.copy()

        # --- Update lowrank_coef Z ---
        tmpZ = background_dict.T @ (data_matrix - anomaly_dict @ sparse_coef - noise_matrix) + J + \
               (background_dict.T @ Y1 - Y2) / mu
        lowrank_coef = inv_Z @ tmpZ

        # --- Update sparse_coef S ---
        tmpS = anomaly_dict.T @ (data_matrix - background_dict @ lowrank_coef - noise_matrix) + L + \
               (anomaly_dict.T @ Y1 - Y3) / mu
        sparse_coef = inv_S @ tmpS

        # --- Update Lagrange multipliers ---
        T1 = data_matrix - background_dict @ lowrank_coef - noise_matrix - anomaly_dict @ sparse_coef
        T2 = lowrank_coef - J
        T3 = sparse_coef - L

        Y1 += mu * T1
        Y2 += mu * T2
        Y3 += mu * T3

        # --- Update mu ---
        err1 = linalg.norm(T1, np.inf)
        err2 = linalg.norm(T2, np.inf)
        err3 = linalg.norm(T3, np.inf)
        max_err = max(err1, err2, err3)
        mu = min(p * mu, mu_max)

        iteration += 1
        print(f"Max error: {max_err}, mu: {mu}")
        if max_err < tol:
            break

    return lowrank_coef, noise_matrix, sparse_coef
