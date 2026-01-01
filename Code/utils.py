# coding=utf-8
import numpy as np
from scipy import linalg, spatial

def load_dataset(file_name):
    """Load dataset from a tab-separated file."""
    dataset = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            dataset.append(list(map(float, line.strip().split('\t'))))
    return np.mat(dataset)

def convert_cube_to_matrix(hyperspectral_cube):
    """Convert 3D hyperspectral cube to 2D matrix (bands x pixels)."""
    rows, cols, bands = hyperspectral_cube.shape
    return hyperspectral_cube.reshape(rows * cols, bands, order='F').T

def convert_matrix_to_cube(data_matrix, rows, cols, bands):
    """Convert 2D matrix back to 3D hyperspectral cube."""
    return data_matrix.T.reshape(rows, cols, bands, order='F')

def compute_autocorrelation(data_matrix):
    """Compute sample autocorrelation of 2D matrix."""
    _, cols = data_matrix.shape
    return np.dot(data_matrix.T, data_matrix) / cols

def compute_covariance(data_matrix):
    """Compute sample covariance of 2D matrix."""
    _, cols = data_matrix.shape
    mean_vec = np.mean(data_matrix, axis=1)
    data_centered = data_matrix - mean_vec[:, None]
    return np.dot(data_centered.T, data_centered) / (cols - 1)

def normalize_matrix(data_matrix, method="L2"):
    """Normalize 2D data matrix using 'minmax' or 'L2' normalization."""
    norm_data = np.zeros_like(data_matrix)
    if method == "minmax":
        min_val = np.min(data_matrix)
        max_val = np.max(data_matrix)
        norm_data = (data_matrix - min_val) / (max_val - min_val + 1e-12)
    elif method == "L2":
        norms = linalg.norm(data_matrix, axis=0)
        norm_data = data_matrix / (norms + 1e-12)
    return norm_data

def convert_to_image(data_matrix):
    """Convert 2D matrix to 0-255 image scale."""
    min_val = data_matrix.min()
    max_val = data_matrix.max()
    return 256 * (data_matrix - min_val) / (max_val - min_val + 1e-12)

def create_hyperspectral_windows(hyperspectral_cube, window_size):
    """Create 3D window blocks from hyperspectral cube."""
    rows, cols, bands = hyperspectral_cube.shape
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    if window_size >= rows or window_size >= cols:
        raise ValueError("Window size too large for image dimensions")

    r = (window_size - 1) // 2
    padded = np.pad(hyperspectral_cube, ((r, r), (r, r), (0, 0)), mode='symmetric')
    num_pixels = rows * cols
    win_matrix = np.zeros((bands, window_size*window_size, num_pixels))

    tmp_block = np.zeros((bands, window_size*window_size))
    for i in range(rows):
        for j in range(cols):
            idx = 0
            for m in range(window_size):
                for n in range(window_size):
                    tmp_block[:, idx] = padded[i+m, j+n, :]
                    idx += 1
            win_matrix[:, :, i*cols+j] = tmp_block
    return win_matrix

def compute_patch_distance(data_matrix, center_matrix):
    """Compute image patch distance between two matrices."""
    _, cols_data = data_matrix.shape
    _, cols_center = center_matrix.shape
    dist_matrix = spatial.distance.cdist(center_matrix.T, data_matrix.T)
    n = cols_data // cols_center
    dist_matrix_3d = dist_matrix.reshape(cols_center, cols_center, n, order='F')
    tmp_min1 = np.min(dist_matrix_3d, axis=1)
    tmp_min2 = np.min(dist_matrix_3d, axis=0)
    combined = np.column_stack((tmp_min1, tmp_min2)).reshape(cols_center, n, 2, order='F')
    return np.sum(np.max(combined, axis=2), axis=0)

def random_centroids(data_cube, k):
    """Generate initial centroids for K-means clustering."""
    rows, cols, n = data_cube.shape
    centroids = np.zeros((rows, cols, k))
    indices = np.arange(n - k*10)
    np.random.shuffle(indices)
    for i in range(k):
        centroids[:, :, i] = data_cube[:, :, indices[i]:indices[i]+200].mean(axis=2)
    return centroids

def kmeans_windows(data_cube, k):
    """Perform K-means clustering on hyperspectral windows."""
    rows, cols, n = data_cube.shape
    data_2d = data_cube.reshape(rows, cols*n, order='F')
    centroids = random_centroids(data_cube, k)
    cluster_labels = np.zeros(n)
    distance_matrix = np.zeros((k, n))
    old_labels = np.zeros(n)
    converged = False
    iteration = 0

    while not converged and iteration < 50:
        iteration += 1
        for j in range(k):
            distance_matrix[j, :] = compute_patch_distance(data_2d, centroids[:, :, j])
        labels_tmp = distance_matrix.argmin(axis=0)
        converged = np.all(labels_tmp == old_labels)
        old_labels = labels_tmp.copy()

        for c in range(k):
            idx = np.where(labels_tmp == c)[0]
            if idx.size > 0:
                centroids[:, :, c] = data_cube[:, :, idx].mean(axis=2)
    return labels_tmp

def simultaneous_omp(dictionary, signal, sparsity_level):
    """
    Compute joint sparse representation using SOMP.
    Returns: alpha, alpha_indices, chosen_atoms, residual
    """
    signal_rows, signal_cols = signal.shape
    index = np.zeros(sparsity_level)
    residual = signal.copy()
    chosen_atoms = np.zeros((signal_rows, sparsity_level))
    alpha = []

    for i in range(sparsity_level):
        projection = residual.T @ dictionary
        tmp_norms = np.linalg.norm(projection, axis=0)
        max_idx = np.argmax(tmp_norms)
        chosen_atoms[:, i] = dictionary[:, max_idx]
        index[i] = max_idx
        tmp2 = chosen_atoms[:, :i+1].T @ chosen_atoms[:, :i+1]
        alpha = np.linalg.pinv(tmp2) @ chosen_atoms[:, :i+1].T @ signal
        residual = signal - chosen_atoms[:, :i+1] @ alpha
    return alpha, index, chosen_atoms, residual
