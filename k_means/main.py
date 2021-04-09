import pdb

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score as sklearn_silhouette_score


def list_of_sets_eq(los1, los2):
    if los1 is None or los2 is None:
        return False

    assert len(los1) == len(
        los2), "Lists must have same size to be comparable!"
    for i in range(len(los1)):
        if not (los1[i] == los2[i]):
            return False
    return True


def average_intra_cluster_distance(data, cluster_labels):
    """
    Computes the average intra-cluster distance i.e the average distance between each point within a cluster.
    Args:
        data (ndarray): num_data_points-by-data_point_dimension matrix (data assumed to be normalized)
        cluster_labels (narray): array of length num_data_points. cluster_labels[i] is the cluster
        label for the data point data[i] for each 0 <= i < length num_data_points.
    Returns:
        Average intra-cluster distance
    """
    distinct_clusters = set(cluster_labels)
    # Initialize dictionary where each cluster label maps to all data points in said cluster
    cluster_label_to_data_points_in_cluster = dict()
    for cluster_label in distinct_clusters:
        data_points_in_cluster = []
        for i in range(len(data)):
            if (cluster_label == cluster_labels[i]):
                data_points_in_cluster.append(data[i])
        cluster_label_to_data_points_in_cluster[cluster_label] = data_points_in_cluster
    # For each cluster compute the intra cluster distances
    for data_points_in_cluster in cluster_label_to_data_points_in_cluster.items():
        data_points = data_points_in_cluster[1]
        _temp_distances = []
        for i in range(0, len(data_points) - 1):
            data_point = data_points[i]
            _rest_data_post = data_points[i:]
            dist_rest_of_data_points_in_cluster = sum(np.power(
                np.linalg.norm(data_point - _rest_data_post, axis=1), 2))
            # print(f'Cluster[{data_points_in_cluster[0]}], data point: {data_point}, sum(distance to other points in cluster): {dist_rest_of_data_points_in_cluster}')
            _temp_distances.append(dist_rest_of_data_points_in_cluster)
        _temp_distances = sum(
            _temp_distances) / len(_temp_distances) if len(_temp_distances) > 0 else 0
        # print(f'Cluster[{data_points_in_cluster[0]}], sum(intra cluster distances): {_temp_distances}')
        cluster_label_to_data_points_in_cluster[data_points_in_cluster[0]
                                                ] = _temp_distances
    # Compute and return the global average intra cluster distance
    _distances = [
        x[1] for x in cluster_label_to_data_points_in_cluster.items()]
    _num_clusters = len(_distances)
    return sum(_distances) / _num_clusters


def average_inter_cluster_distance(data, cluster_labels, centroids):
    """
    Computes the average inter-cluster distance i.e the average distance between all clusters. 
    We have assumed the distance between two clusters is the distance between the centroids of 
    said clusters
    Args:
        data (ndarray): num_data_points-by-data_point_dimension matrix (data assumed to be normalized)
        cluster_labels (narray): array of length num_data_points. cluster_labels[i] is the cluster
        label for the data point data[i] for each 0 <= i < length num_data_points.
    Returns:
        Average inter-cluster distance
    """
    distinct_clusters = set(cluster_labels)
    number_of_distances = len(centroids)*(len(centroids) - 1) / 2
    _sum = 0
    for i in range(0, len(centroids) - 1):
        string = i, '\t',  centroids[i:]
        centroid = centroids[i]
        dist_to_other_clusters = sum(np.power(
            np.linalg.norm(centroid - centroids[i:], axis=1), 2))
        _sum += dist_to_other_clusters
    return _sum / number_of_distances


def silhouette_coefficient(data=None, centroids=None):
    """
    Function implementing the k-means clustering.

    Distance metric: eucledian

    Args:
        data (ndarray):
        centroids ():
    Returns:
        mean Silhouette Coefficient of all samples
    """
    ### START CODE HERE ###
    if data is None or centroids is None:
        return
    centers, clusters, labels = k_means(data, centroids)
    a = average_intra_cluster_distance(data, labels)
    # print(f'Average intra cluster distance: {a}')
    b = average_inter_cluster_distance(data, labels, centroids)
    # print(f'Average inter cluster distance: {b}')
    return (b - a) / max(a, b)
    ### END CODE HERE ###


def k_means(data, centroids):
    """
    Compute kmeans cluster centers

    Args:
        data (ndarray): num_data_points-by-data_point_dimension matrix (data assumed to be normalized).
        n_clusters (int): number of cluster centers to compute.

    Returns:
        ndarray: n_clusters-by-data_point_dimensions matrix where each row is a cluster center
    """
    n_clusters = len(centroids)
    if n_clusters > len(data):
        return None, None, None
    num_data_points, data_point_dimension = data.shape
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
                np.linalg.norm(data_point - centroids, axis=1), 2)
            index_of_cluster_with_min_dist_to_data_point = np.argmin(
                dist_to_centers
            )
            labels[i] = index_of_cluster_with_min_dist_to_data_point
            clusters[index_of_cluster_with_min_dist_to_data_point].add(i)
        # Using computed clusters (from previous loop), compute cluster centers
        for i in range(n_clusters):
            if len(clusters[i]) > 0:
                centroids[i, :] = np.mean(data[list(clusters[i]), :], axis=0)
    return centroids, clusters, labels


if __name__ == "__main__":
    num_data_points, num_clusters, data_point_dimension = 30, 3, 2
    data = np.random.uniform(0, 1, (num_data_points, data_point_dimension))
    # Randomly initing cluster centers
    centers = np.random.uniform(0, 1, (num_clusters, data_point_dimension))
    centers, clusters, labels = k_means(data, centers)
    silhouette_score = silhouette_coefficient(data, centers)
    # print(silhouette_score)
    # print(sklearn_silhouette_score(data, labels, metric='euclidean'))
    # _, ax = plt.subplots()
    # for i in range(num_clusters):
    #     tmp = ax.scatter(*data[list(clusters[i]), :].T)
    #     ax.scatter(*centers[i, :].T, color=tmp.get_facecolor())
    #     ax.annotate("Center", xy=(centers[i, 0], centers[i, 1]))
    # plt.show()
    test_data = np.ndarray([
        [0.64105096, 0.49883149],
        [0.32758003, 0.72580733],
        [0.61195803, 0.08976073]
    ])
    test_data = [[round(x[0], 2), round(x[1], 2)] for x in test_data]
    print(test_data)
    print(
        sum(np.power(np.linalg.norm(test_data[0] - test_data[1], axis=1), 2)))
