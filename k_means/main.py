import pdb

import matplotlib.pyplot as plt
import numpy as np


def list_of_sets_eq(los1, los2):
    if los1 is None or los2 is None:
        return False

    assert len(los1) == len(
        los2), "Lists must have same size to be comparable!"
    for i in range(len(los1)):
        if not (los1[i] == los2[i]):
            return False
    return True


def silhouette_score(data, centroids):
    """
    Function implementing the k-means clustering.

    :param data
        data
    :param centroids
        centroids
    :return
        mean Silhouette Coefficient of all samples
    """
    ### START CODE HERE ###
    _sum = 0
    for centroid in centroids:
        print(centroid)
        _number_of_points_in_cluster = 0
        for data_point in data:
            _number_of_points_in_cluster += 1
            _temp = np.power(np.linalg.norm(data_point - centroid, axis=0), 2)
            # print(data_point, ",", centroid, ",", _temp)
            _sum += _temp
        print("Number of points in cluster[", np.where(centroids == centroid)[
              0][0], "]: ", _number_of_points_in_cluster)
    return _sum
    ### END CODE HERE ###


def k_means(data, n_clusters):
    """
    Compute kmeans cluster centers

    Args:
        data (ndarray): num_data_points-by-data_point_dimension matrix (data assumed to be normalized).
        n_clusters (int): number of cluster centers to compute.

    Returns:
        ndarray: n_clusters-by-data_point_dimensions matrix where each row is a cluster center
    """
    if n_clusters > len(data):
        return None, None, None

    num_data_points, data_point_dimension = data.shape

    # Randomly initing cluster centers
    centers = np.random.uniform(0, 1, (n_clusters, data_point_dimension))

    labels = [None for i in range(num_data_points)]

    prev_clusters, clusters = None, None

    while not list_of_sets_eq(prev_clusters, clusters):
        prev_clusters = clusters

        clusters = [set() for _ in range(n_clusters)]
        for i in range(num_data_points):
            # Retrieve single data point
            data_point = data[i, :].reshape(1, data_point_dimension)
            # Calculate square of the distances between data point and all cluster centers
            dist_to_centers = np.power(
                np.linalg.norm(data_point - centers, axis=1), 2)

            index_of_cluster_with_min_dist_to_data_point = np.argmin(
                dist_to_centers
            )
            labels[i] = index_of_cluster_with_min_dist_to_data_point
            clusters[index_of_cluster_with_min_dist_to_data_point].add(i)

        # Using computed clusters (from previous loop), compute cluster centers
        for i in range(n_clusters):
            if len(clusters[i]) > 0:
                centers[i, :] = np.mean(data[list(clusters[i]), :], axis=0)

    return centers, clusters, labels


def silhouette_score(data, centroids):
    """
    Function implementing the k-means clustering.

    :param data
        data
    :param centroids
        centroids
    :return
        mean Silhouette Coefficient of all samples
    """
    from sklearn.metrics import silhouette_score as ss
    # _centroids, _clusters, _labels = k_means(data, centroids)
    # return ss(data, _labels, metric='euclidean')
    if sample_size is not None:
        X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr'])
        random_state = check_random_state(random_state)
        indices = random_state.permutation(X.shape[0])[:sample_size]
        if metric == "precomputed":
            X, labels = X[indices].T[indices].T, labels[indices]
        else:
            X, labels = X[indices], labels[indices]
    return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))


if __name__ == "__main__":
    num_data_points, data_point_dimension = 100, 2
    data = np.random.uniform(0, 1, (num_data_points, data_point_dimension))
    num_clusters = 3
    centers, clusters, labels = k_means(data, num_clusters)

    _, ax = plt.subplots()
    for i in range(num_clusters):
        tmp = ax.scatter(*data[list(clusters[i]), :].T)
        ax.scatter(*centers[i, :].T, color=tmp.get_facecolor())
        ax.annotate("Center", xy=(centers[i, 0], centers[i, 1]))

    print()

    # plt.show()
