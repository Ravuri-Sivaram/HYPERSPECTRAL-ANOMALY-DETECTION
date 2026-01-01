# coding=utf-8

import numpy as np
import scipy.io as sio

from dictionary import build_dictionaries
from lrsr_ import compute_lrsr
from visualization import display_results, compute_roc_auc
from utils import load_dataset  # optional if needed

# --- Load hyperspectral data ---
data_file = r"D:\Administrator\Documents\Spectral_Image_Phase-2_Major_4-2\Hyperspectral-Anomaly-Detection-master\Sandiego.mat"
groundtruth_file = r"D:\Administrator\Documents\Spectral_Image_Phase-2_Major_4-2\Hyperspectral-Anomaly-Detection-master\PlaneGT.mat"

data_mat = sio.loadmat(data_file)
data3d = np.array(data_mat["Sandiego"], dtype=float)
data3d = data3d[0:100, 0:100, :]

# Remove noisy bands
remove_bands = np.hstack((
    range(6),
    range(32, 35),
    range(93, 97),
    range(106, 113),
    range(152, 166),
    range(220, 224)
))
data3d = np.delete(data3d, remove_bands, axis=2)
rows, cols, bands = data3d.shape

# Load groundtruth
groundtruth_mat = sio.loadmat(groundtruth_file)
groundtruth = np.array(groundtruth_mat["PlaneGT"])

# --- Construct background and anomaly dictionaries ---
data2d, background_dict, anomaly_dict, bg_dict_labels, anomaly_dict_labels = build_dictionaries(
    hyperspectral_cube=data3d,
    groundtruth=groundtruth,
    window_size=3,
    num_clusters=10,
    sparsity_level=10,
    selected_dict_percent=0.05,
    anomaly_dict_num=200
)

# --- Compute Low-Rank and Sparse Representation ---
lowrank_coef, noise_matrix, sparse_coef = compute_lrsr(
    background_dict, anomaly_dict, data2d,
    beta=0.001, lamda=0.01
)

# --- Visualize results ---
background2d, anomaly2d = display_results(
    background_dict, anomaly_dict, lowrank_coef, sparse_coef, noise_matrix,
    rows, cols, bands, bg_dict_labels, anomaly_dict_labels
)

# --- Compute ROC-AUC ---
auc_value = compute_roc_auc(anomaly2d, groundtruth)
print(f"The AUC is: {auc_value:.4f}")
