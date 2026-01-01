# coding=utf-8
import numpy as np
import scipy.io as sio
from sklearn import decomposition
from utils import convert_cube_to_matrix, convert_matrix_to_cube, normalize_matrix, create_hyperspectral_windows, kmeans_windows, simultaneous_omp

def build_dictionaries(hyperspectral_cube, groundtruth, window_size, num_clusters, sparsity_level,
                       selected_dict_percent, anomaly_dict_num):
    """
    Build background and anomaly dictionaries from hyperspectral data.
    
    :param hyperspectral_cube: 3D hyperspectral image (rows x cols x bands)
    :param groundtruth: 2D groundtruth labels
    :param window_size: size of local window (odd number)
    :param num_clusters: number of clusters for K-means
    :param sparsity_level: sparsity level for SOMP
    :param selected_dict_percent: fraction of atoms to select for background dictionary
    :param anomaly_dict_num: number of atoms for anomaly dictionary
    :return:
        data_matrix: normalized data (bands x pixels)
        background_dict: selected background dictionary
        anomaly_dict: selected anomaly dictionary
        bg_dict_labels: indices of background dictionary atoms
        anomaly_dict_labels: indices of anomaly dictionary atoms
    """
    # Convert and normalize
    data_matrix = convert_cube_to_matrix(hyperspectral_cube)
    data_matrix = normalize_matrix(data_matrix, method="L2")
    rows, cols, bands = hyperspectral_cube.shape
    cube_dim = convert_matrix_to_cube(data_matrix, rows, cols, bands)
    
    # PCA for dimensionality reduction
    pca = decomposition.PCA(n_components=20)
    dim_data = pca.fit_transform(data_matrix.T)
    cube_dim = convert_matrix_to_cube(dim_data.T, rows, cols, 20)
    
    # Create local windows
    win_dim = create_hyperspectral_windows(cube_dim, window_size)
    
    # K-means clustering
    cluster_labels = kmeans_windows(win_dim, num_clusters)
    
    win_matrix = create_hyperspectral_windows(hyperspectral_cube, window_size)
    wm_rows, wm_cols, wm_n = win_matrix.shape
    
    residual_stack = np.zeros((bands, window_size * window_size, wm_n))
    save_counter = 0
    background_dict_list = []
    bg_dict_labels_list = []
    
    class_order_list = []
    anomaly_weight_list = []
    
    # Process each cluster
    for cluster_id in range(num_clusters):
        print(f"Processing cluster {cluster_id} ...")
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if cluster_indices.size == 0:
            continue
        
        class_data = win_matrix[:, :, cluster_indices]
        cd_rows, cd_cols, cd_n = class_data.shape
        dictionary = class_data[:, int((window_size*window_size+1)/2), :]
        dic_rows, dic_cols = dictionary.shape
        
        # Initialize sparse representation matrices
        class_alpha = np.zeros((sparsity_level, cd_cols, cd_n))
        class_index = np.zeros((sparsity_level, cd_n))
        
        # SOMP for each window in the cluster
        for j in range(cd_n):
            X = class_data[:, :, j]
            dictionary[:, j*cd_cols:(j*cd_cols+cd_cols-1)] = 0
            alpha, index, _, residual = simultaneous_omp(dictionary, X, sparsity_level)
            class_alpha[:, :, j] = alpha
            class_index[:, j] = index.T
            residual_stack[:, :, save_counter+j] = residual
        
        save_counter += cd_n
        class_index = class_index.astype(int)
        
        # Global alpha & frequency
        class_global_alpha = np.zeros((dic_cols, cd_cols, cd_n))
        class_global_frequency = np.zeros((dic_cols, cd_cols, cd_n))
        for n_idx in range(cd_n):
            class_global_alpha[class_index[:, n_idx], :, n_idx] = class_alpha[:, :, n_idx]
            class_global_frequency[class_index[:, n_idx], :, n_idx] = 1
        
        # Sparsity & anomaly weight
        abs_alpha = np.fabs(class_global_alpha)
        data_frequency = class_global_frequency[:, 0, :]
        frequency = np.sum(data_frequency, axis=1)
        norm_frequency = frequency / np.sum(frequency)
        data_mean_alpha = np.mean(abs_alpha, axis=1)
        sum_alpha_2 = np.sum(data_mean_alpha, axis=1)
        sparsity_score = sum_alpha_2 / np.linalg.norm(sum_alpha_2)
        anomaly_weight = norm_frequency.copy()
        anomaly_weight[frequency > 0] = sparsity_score[frequency > 0] / frequency[frequency > 0]
        
        # Select dictionary atoms
        sparsity_sort_idx = np.argsort(-sparsity_score).astype(int)
        selected_num = int(round(selected_dict_percent * cd_n))
        bg_dict_labels_list.append(cluster_indices[sparsity_sort_idx[:selected_num]])
        background_dict_list.append(dictionary[:, sparsity_sort_idx[:selected_num]])
        anomaly_weight_list.append(anomaly_weight)
        class_order_list.append(np.array(cluster_indices))
    
    background_dict = np.column_stack(background_dict_list)
    bg_dict_labels = np.hstack(bg_dict_labels_list)
    
    # Compute anomaly dictionary
    norm_res = np.zeros((wm_n, window_size*window_size))
    for i in range(wm_n):
        norm_res[i, :] = np.linalg.norm(residual_stack[:, :, i], axis=0)
    mean_norm_res = np.mean(norm_res, axis=1) * np.hstack(anomaly_weight_list).T
    anomaly_level = mean_norm_res / np.linalg.norm(mean_norm_res)
    tg_sort_idx = np.argsort(-anomaly_level)
    
    anomaly_dict = data_matrix[:, np.hstack(class_order_list)[tg_sort_idx[:anomaly_dict_num]]]
    anomaly_dict_labels = np.hstack(class_order_list)[tg_sort_idx[:anomaly_dict_num]]
    
    print("Dictionary construction completed!")
    
    # Save results (optional)
    sio.savemat("background_dict.mat", {'background_dict': background_dict})
    sio.savemat("bg_dict_labels.mat", {'bg_dict_labels': bg_dict_labels})
    sio.savemat("anomaly_dict.mat", {'anomaly_dict': anomaly_dict})
    sio.savemat("anomaly_dict_labels.mat", {'anomaly_dict_labels': anomaly_dict_labels})
    
    return data_matrix, background_dict, anomaly_dict, bg_dict_labels, anomaly_dict_labels
