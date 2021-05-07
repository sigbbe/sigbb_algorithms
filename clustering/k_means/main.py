import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
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


def __init_data__(num_data_points=100, num_clusters=4):
    data_point_dimension = 2
    # Randomly initing cluster centers
    centers = np.random.uniform(0, 1, (num_clusters, data_point_dimension))
    data = np.random.uniform(0, 1, (num_data_points, data_point_dimension))
    return data, centers


# if __name__ == "__main__":
#     num_data_points, num_clusters, data_point_dimension = 30, 3, 2
#     data, centers = __init_data__(num_data_points, num_clusters)
#     centers, clusters, labels = k_means(data, centers)
#     silhouette_score = silhouette_coefficient(data, centers)
#     # print(silhouette_score)
#     # print(sklearn_silhouette_score(data, labels, metric='euclidean'))
#     # _, ax = plt.subplots()
#     # for i in range(num_clusters):
#     #     tmp = ax.scatter(*data[list(clusters[i]), :].T)
#     #     ax.scatter(*centers[i, :].T, color=tmp.get_facecolor())
#     #     ax.annotate("Center", xy=(centers[i, 0], centers[i, 1]))
#     # plt.show()
#     test_data = np.ndarray([
#         [0.64105096, 0.49883149],
#         [0.32758003, 0.72580733],
#         [0.61195803, 0.08976073]
#     ])
#     test_data = [[round(x[0], 2), round(x[1], 2)] for x in test_data]
#     print(test_data)
#     print(
#         sum(np.power(np.linalg.norm(test_data[0] - test_data[1], axis=1), 2)))


# Pythagoras
def calculate_distance(a, b):
    x1 = b[0] - a[0]
    x2 = np.absolute(b[1] - a[1])

    distance = np.sqrt(np.power(x1, 2) + np.power(x2, 2))
    return distance


def test__(a, b):
    # return np.power(np.linalg.norm(a - b, axis=1), 2)
    _a = np.array((a[0], a[1]))
    _b = np.array((b[0], b[1]))
    return np.linalg.norm(_a - _b)


def f(x):
    return x[1:]


def do_preprocessing(data, f):
    return np.array([f(x) for x in data_array.tolist()])


if __name__ == '__main__':
    data_frame = pd.read_csv('data/v_2017.csv')
    data_array = data_frame.to_numpy()
    data_array_list = do_preprocessing(data_array, f)
    data_array = data_array_list
    num_data_points, num_clusters, data_point_dimension = 30, 3, 2
    data, centroids = __init_data__(num_data_points, num_clusters)
    centroids, clusters, labels = k_means(data_array, centroids)

    # silhouette_score = silhouette_coefficient(data, centers)
    # print(silhouette_score)
    metric = 'euclidean'
    # print(sklearn_silhouette_score(data, labels, metric=metric))
    _, ax = plt.subplots()
    colors = [np.cos(t) for t in [1.7, 3.2, 5.1]]
    for i in range(num_clusters):
        print(centroids[i])
        if len(clusters[i]) < 1:
            continue
        print(f'Datapoints in cluster {i}:')
        print(data_array[list(clusters[i]), :].T)
        data_points_in_cluster = data_array[list(clusters[i]), :].T
        cluster_centroid = centroids[i, :].T
        tmp = ax.scatter(*data_points_in_cluster,
                         [colors[0]]*len(data_points_in_cluster))
        color = tmp.get_facecolor()
        tmp_color = '#fff000'
        ax.scatter(*cluster_centroid, c=colors[i])
        ax.annotate(f"Centroid[{i}]", xy=(centroids[i, 0], centroids[i, 1]))
    plt.show()

    # test_data = np.ndarray([
    #     [0.64105096, 0.49883149],
    #     [0.32758003, 0.72580733],
    #     [0.61195803, 0.08976073]
    # ])
    # test_data = [[round(x[0], 2), round(x[1], 2)] for x in test_data]
    # print(test_data)
    # print(
    #     sum(np.power(np.linalg.norm(test_data[0] - test_data[1], axis=1), 2)))
