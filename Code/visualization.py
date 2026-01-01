# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import metrics
from utils import convert_matrix_to_cube

def display_results(background_dict, anomaly_dict, lowrank_coef, sparse_coef, noise_matrix,
                    rows, cols, bands, bg_dict_labels, anomaly_dict_labels):
    """
    Visualize the results of LRSR: background, anomaly, noise, dictionary atoms, segmentation.
    
    :param background_dict: background dictionary
    :param anomaly_dict: anomaly dictionary
    :param lowrank_coef: low-rank coefficients
    :param sparse_coef: sparse coefficients
    :param noise_matrix: noise matrix
    :param rows: number of rows in original HSI
    :param cols: number of columns
    :param bands: number of spectral bands
    :param bg_dict_labels: indices of background dictionary atoms
    :param anomaly_dict_labels: indices of anomaly dictionary atoms
    :return:
        background2d: reconstructed background (bands x pixels)
        anomaly2d: reconstructed anomaly (bands x pixels)
    """
    # Reconstruct background, anomaly, and noise
    background2d = background_dict @ lowrank_coef
    anomaly2d = anomaly_dict @ sparse_coef
    background3d = convert_matrix_to_cube(background2d, rows, cols, bands)
    anomaly3d = convert_matrix_to_cube(anomaly2d, rows, cols, bands)
    noise3d = convert_matrix_to_cube(noise_matrix, rows, cols, bands)

    # Visualize dictionary atoms
    bg_show = np.zeros((1, rows * cols))
    anomaly_show = np.zeros((1, rows * cols))
    bg_show[0, bg_dict_labels] = 1
    anomaly_show[0, anomaly_dict_labels] = 1
    bg_show = bg_show.reshape(rows, cols)
    anomaly_show = anomaly_show.reshape(rows, cols)

    # Segmentation
    cluster_file = sio.loadmat("cluster_assment.mat")
    cluster_labels = np.array(cluster_file["cluster_assment"]).T
    segm_show = cluster_labels.reshape(rows, cols)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(background3d.mean(2))
    plt.xlabel('Background')

    plt.subplot(2, 3, 2)
    plt.imshow(anomaly3d.mean(2))
    plt.xlabel('Anomaly')

    plt.subplot(2, 3, 3)
    plt.imshow(noise3d.mean(2))
    plt.xlabel('Noise')

    plt.subplot(2, 3, 4)
    plt.imshow(bg_show)
    plt.xlabel('Background Dictionary')

    plt.subplot(2, 3, 5)
    plt.imshow(anomaly_show)
    plt.xlabel('Anomaly Dictionary')

    plt.subplot(2, 3, 6)
    plt.imshow(segm_show)
    plt.xlabel('Segmentation')

    plt.tight_layout()
    plt.show()

    return background2d, anomaly2d


def compute_roc_auc(anomaly2d, groundtruth):
    """
    Compute ROC curve and AUC for anomaly detection results.
    
    :param anomaly2d: 2D anomaly component (bands x pixels)
    :param groundtruth: 2D groundtruth matrix
    :return:
        auc_value: AUC value
    """
    rows, cols = groundtruth.shape
    label = groundtruth.T.reshape(1, rows * cols)
    result = np.linalg.norm(anomaly2d, axis=0).reshape(1, -1)

    fpr, tpr, _ = metrics.roc_curve(label.T, result.T)
    auc_value = metrics.auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC={auc_value:.4f})')
    plt.grid(True)
    plt.show()

    return auc_value
