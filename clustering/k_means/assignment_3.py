import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score as sklearn_silhouette_score


# %% ----------------------------------------------------------------------------------------
def plot_clusters(data, centroids):
    """
    Shows a scatter plot with the data points clustered according to the centroids.
    """
    # Assigning the data points to clusters/centroids.
    clusters = [[] for _ in range(centroids.shape[0])]
    for i in range(data.shape[0]):
        distances = np.linalg.norm(data[i] - centroids, axis=1)
        clusters[np.argmin(distances)].append(data[i])

    # Plotting clusters and centroids.
    fig, ax = plt.subplots()
    for c in range(centroids.shape[0]):
        if len(clusters[c]) > 0:
            cluster = np.array(clusters[c])
            ax.scatter(cluster[:, 0], cluster[:, 1], s=7)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red')
# -------------------------------------------------------------------------------------------


# We would like to have some control over the randomly generated data.
# This is just for development purposes.
# %% ----------------------------------------------------------------------------------------
np.random.seed(0)

# Euclidean space.
DIMENSIONS = 2

# We will generate clusters.
CLUSTERS = [
    {
        'mean': (10, 10),
        'std': (10, 5),
        'size': 300
    },
    {
        'mean': (10, 85),
        'std': (10, 3),
        'size': 100
    },
    {
        'mean': (50, 50),
        'std': (6, 6),
        'size': 200
    },
    {
        'mean': (80, 75),
        'std': (5, 10),
        'size': 200
    },
    {
        'mean': (80, 20),
        'std': (5, 5),
        'size': 100
    }
]
# -------------------------------------------------------------------------------------------

# %% ----------------------------------------------------------------------------------------
# Initializing the dataset with zeros.
synthetic_data = np.zeros((np.sum([c['size'] for c in CLUSTERS]), DIMENSIONS))

# Generating the clusters.
start = 0
for c in CLUSTERS:
    for d in range(DIMENSIONS):
        synthetic_data[start:start + c['size'],
                       d] = np.random.normal(c['mean'][d], c['std'][d], (c['size']))
    start += c['size']
# -------------------------------------------------------------------------------------------


# %% ----------------------------------------------------------------------------------------
plt.figure()
plt.scatter(*synthetic_data.T, s=3)
# -------------------------------------------------------------------------------------------

# %% ----------------------------------------------------------------------------------------


def list_of_sets_eq(los1, los2):
    if los1 is None or los2 is None:
        return False

    assert len(los1) == len(
        los2), "Lists must have same size to be comparable!"
    for i in range(len(los1)):
        if not (los1[i] == los2[i]):
            return False
    return True


def kmeans(data, centroids):
    """
    Function implementing the k-means clustering.
    :param data
        data
    :param centroids
        initial centroids
    :return
        final centroids
    """
    ### START CODE HERE ###
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
    ### END CODE HERE ###
# -------------------------------------------------------------------------------------------


# %% ----------------------------------------------------------------------------------------
test_data = np.array([
    [66.24345364, 57.31053969],
    [43.88243586, 39.69929645],
    [44.71828248, 48.38791398],
    [39.27031378, 48.07972823],
    [58.65407629, 55.66884721],
    [26.98461303, 44.50054366],
    [67.44811764, 49.13785896],
    [42.38793099, 45.61070791],
    [53.19039096, 50.21106873],
    [47.50629625, 52.91407607],
    [2.29566576, 20.15837474],
    [18.01306597, 22.22272531],
    [16.31113504, 20.1897911],
    [13.51746037, 19.08356051],
    [16.30599164, 20.30127708],
    [5.21390499, 24.91134781],
    [9.13976842, 17.17882756],
    [3.44961396, 26.64090988],
    [8.12478344, 36.61861524],
    [13.71248827, 30.19430912],
    [74.04082224, 23.0017032],
    [70.56185518, 16.47750154],
    [71.26420853, 8.57481802],
    [83.46227301, 16.50657278],
    [75.25403877, 17.91105767],
    [71.81502177, 25.86623191],
    [75.95457742, 28.38983414],
    [85.50127568, 29.31102081],
    [75.60079476, 22.85587325],
    [78.08601555, 28.85141164]
])
test_centroids = np.array([
    [25, 50],
    [50, 50],
    [75, 50]
])

test_centroids, _clusters, _labels = kmeans(test_data, test_centroids)

print('c0 =', test_centroids[0])
print('c1 =', test_centroids[1])
print('c2 =', test_centroids[2])
plot_clusters(test_data, test_centroids)
# -------------------------------------------------------------------------------------------

# %% ----------------------------------------------------------------------------------------
# Number of clusters.
K = 5

# Boundaries of our data.
x_min = np.min(synthetic_data[:, 0])
x_max = np.max(synthetic_data[:, 0])
y_min = np.min(synthetic_data[:, 1])
y_max = np.max(synthetic_data[:, 1])

# Generating random centroids within the data boundaries.
centroids = np.zeros((K, synthetic_data.shape[1]))
centroids[:, 0] = np.random.randint(x_min, x_max, size=K)
centroids[:, 1] = np.random.randint(y_min, y_max, size=K)

for i in range(len(centroids)):
    print('c%d =' % i, centroids[i])
plot_clusters(synthetic_data, centroids)
# -------------------------------------------------------------------------------------------

# %% ----------------------------------------------------------------------------------------
# centroids, _clusters, _labels = kmeans(synthetic_data, centroids)
data, _clusters, _labels = kmeans(synthetic_data, centroids)

# plt.scatter(data[:, 0], data[:, 1], s=3)
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red')

for i in range(len(centroids)):
    print('c%d =' % i, centroids[i])
plot_clusters(synthetic_data, centroids)
# -------------------------------------------------------------------------------------------

# %% ----------------------------------------------------------------------------------------


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
    centers, clusters, labels = kmeans(data, centroids)
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
    _, __, labels = kmeans(data, centers)
    assert silhouette_coefficient(
        data, labels) == sklearn_silhouette_score(data, labels, metric='euclidean')


def distance(a, b):
    _a = np.array((a[0], a[1]))
    _b = np.array((b[0], b[1]))
    return np.linalg.norm(_a - _b)


def closest_centroid(data, centroids):
    index_of_close_centroid = []
    for i in range(len(data)):
        point = data[i]
        closest_index = 0
        closest_distance = distance(point, centroids[0])
        for j in range(len(centroids)):
            current = distance(point, centroids[j])
            if not(current >= closest_distance):
                closest_index = j
                closest_distance = current
        index_of_close_centroid.append(closest_index)
    return index_of_close_centroid


def fetch_clusters(data, centroids):
    clusters = []
    for _ in range(len(centroids)):
        clusters.append([])
    closest = closest_centroid(data, centroids)
    for i in range(len(closest)):
        clusters[closest[i]].append(data[i])
    return clusters


def average(a, b):
    _sum = 0
    for i in range(len(a)):
        score = (b[i] - a[i]) / max([a[i], b[i]])
        _sum += score
    return _sum / len(a)


def silhouette_score(data, centroids):
    clusters = fetch_clusters(data, centroids)
    a = []
    for cluster in clusters:
        for point in cluster:
            a.append(np.sum(np.linalg.norm(np.array(point) -
                     np.array(cluster), axis=1)) / (len(cluster)-1))

    b = []
    for i in range(len(clusters)):
        for point in clusters[i]:
            tmp_clusters = list(clusters)
            del tmp_clusters[i]
            avg_distances = []
            for tmp in tmp_clusters:
                avg = np.sum(np.linalg.norm(np.array(point) -
                             np.array(tmp), axis=1)) / (len(tmp))
                avg_distances.append(avg)
            b.append(min(avg_distances))
    return average(a, b)
    ### END CODE HERE ###


# -------------------------------------------------------------------------------------------

# %% ----------------------------------------------------------------------------------------
print(f'{0.67} == {silhouette_score(test_data, test_centroids)}')
# -------------------------------------------------------------------------------------------


# %% ----------------------------------------------------------------------------------------
centroids = np.zeros((K, synthetic_data.shape[1]))
centroids[:, 0] = np.random.randint(x_min, x_max, size=K)
centroids[:, 1] = np.random.randint(y_min, y_max, size=K)

_, _clusters, _labels = kmeans(synthetic_data, centroids)
silhouette_coefficient = silhouette_score(synthetic_data, centroids)


print('silhouette_coefficient =', round(silhouette_coefficient, 3))
print('correct_silhouette_coefficient =',
      round(sklearn_silhouette_score(synthetic_data, _labels, metric='euclidean'), 3))
plot_clusters(synthetic_data, centroids)

# The silhouette score primarily seems to fluctuate between values in the range 0.513 to 0.733
# -------------------------------------------------------------------------------------------


# %-% ----------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
