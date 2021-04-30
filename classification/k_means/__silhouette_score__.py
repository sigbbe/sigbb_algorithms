import pdb

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import silhouette_score as sklearn_silhouette_score

from main import k_means


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


def test_silhouette_coefficient():
    num_data_points, num_clusters, data_point_dimension = 30, 3, 2
    data = np.random.uniform(0, 1, (num_data_points, data_point_dimension))
    # Randomly initing cluster centers
    centers = np.random.uniform(0, 1, (num_clusters, data_point_dimension))
    centers, clusters, labels = k_means(data, centers)
    assert silhouette_coefficient(
        data, labels) == sklearn_silhouette_score(data, labels, metric='euclidean')


def main():
    return
