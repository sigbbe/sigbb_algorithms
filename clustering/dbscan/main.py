import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import shape

__author__ = 'Sigjørn Berdal'
__date__ = '07.05.2021'


def main(**args):
    min_pts = 3
    e_ps = 3
    dataset = pd.read_csv('./data/v_2013.csv').to_numpy()
    clusters, noise = dbscan(dataset, min_pts, e_ps)
    # print(f'Clusters: {clusters}')
    # print(f'Noise: {noise}')
    print(f'Number of clusters: {len(clusters)}')
    print('Noise')
    for n in noise:
        print(n)
    plot(clusters, noise)
# ---------------- Distance measures ---------------- #


def euclidean_distance(X, Y):
    return np.linalg.norm(X - Y)


def manhattan_distance(X, Y):
    return -1


def supremum(X, Y):
    return -1

# --------------------------------------------------- #


def no_object_unvisited(objects_visited):
    for object in objects_visited:
        if object == 0:
            return False
    return True


def choose_random_unvisited_object(objects_visited):
    unvisited_object_indexes = []
    for i in range(len(objects_visited)):
        if (objects_visited[i] == 0):
            unvisited_object_indexes.append(i)
    random_index = random.randint(0, len(unvisited_object_indexes) - 1)
    return unvisited_object_indexes[random_index]


def get_objects_within_radius(data_object, dataset, radius=3):
    neighbours = np.reshape(np.array([]), (0, 0))
    for i in range(len(dataset)):
        ith_data_object = dataset[i]
        distance = euclidean_distance(data_object, ith_data_object)
        if distance <= radius:
            neighbours = np.append(neighbours, ith_data_object)
    return np.reshape(np.array(neighbours), (int(len(neighbours) / 2), 2))


def is_member_of_a_cluster(data_object, clusters):
    def arr_contains_element(arr, element):
        if len(np.where(arr == element)) == 0:
            return False
        return len(np.where(arr == element)[0]) > 0
    for cluster in clusters:
        if arr_contains_element(cluster, data_object):
            return True
    return False


def dbscan(dataset, min_pts=3, e_ps=3):
    """
    Algorithm - DBSCAN: a density-based clustering algorithm.

    Args:
        dataset (n-d-array): a dataset containing n d-dimensional objects

        e_ps (int): the radius distance parameter

        min_pts (int): the neighborhood density threshold

    Returns:
        A set of density-based clusters

    Comments:
        Each value i in data_objects_visited represents the visited state of data object i in dataset.
        0 means the data object has not yet been visited
        1 means the data object has been visited
        -1 means the data object is noise
    """

    # Mark all objects as unvisited
    data_objects_visited = np.array([0 for i in dataset])
    clusters, noise = [], []
    while not(no_object_unvisited(data_objects_visited)):
        # Randomly chosen index, from the unvisited objects
        random_index = choose_random_unvisited_object(data_objects_visited)
        # p
        p = dataset[random_index]
        if (p.shape[0] != 2):
            continue
        # print(f'p: {p}')
        # Mark p as visited
        data_objects_visited[random_index] = 1

        neighbours_of_p = get_objects_within_radius(p, dataset, e_ps)
        # Create new cluster C, and add p to C
        cluster = np.array(p)
        # If the e_ps-neighborhood of p has at least min_pts objects
        if (neighbours_of_p.shape[0] >= min_pts):
            for p_prime in neighbours_of_p:
                p_prime = np.array([int(p_prime[0]), int(p_prime[1])])
                # p_prime = dataset[neighbourhood_index_of_p_prime]
                index_of_p_prime = index_of_element_in_n_d_array(
                    p_prime, dataset)
                # print(index_of_p_prime)
                # If p' is unvisited
                # {data_objects_visited[index_of_p_prime]}
                # print(
                #     f"i = {p_prime}, p' = data_objects_visited[{index_of_p_prime}] = Nan")
                # print(
                #     f"index_of_p_prime = {index_of_p_prime}")
                if (not(data_objects_visited[index_of_p_prime])):
                    # Mark p' as visited
                    data_objects_visited[index_of_p_prime] = 1
                    # The e_ps-neighborhood of p'
                    neighbours_of_p_prime = np.array(list(map(lambda x: np.array([int(x[0]), int(x[1])]), get_objects_within_radius(
                        p_prime, dataset, e_ps))))
                    # If the e_ps-neighborhood of p'has at least min_pts number of points, add those points to neighbours_of_p
                    if (neighbours_of_p_prime.shape[0] >= min_pts):
                        for neighbour in neighbours_of_p_prime:
                            np.append(
                                neighbours_of_p, neighbour)
                # If p'is not yet a member of any cluster, add p' to C
                if (not(is_member_of_a_cluster(p_prime, clusters))):
                    cluster = np.append(cluster, p_prime)
            cluster = np.reshape(cluster, (int(cluster.shape[0]/2), 2))
            """
            Generally, nested NumPy arrays of NumPy arrays are not very useful. 
            If we use NumPy for speed, usually it is best to stick with 
            NumPy arrays with a homogenous, basic numeric dtype.
            """
            # Me not knowing how to use np.array

            # print(np.append([[]], cluster, axis=0))
            # _cluster = [point for point in cluster]
            # print(_cluster)
            # print([cluster])
            # print(np.reshape(np.append(clusters, cluster),
            #       (clusters.shape[0] - 2, 2)))
            # clusters = np.append(clusters, cluster)
            # if (len(clusters.shape) == 1):
            #     clusters.shape = (int(clusters.shape[0] / 2), 2)
            # clusters = np.array([clusters, cluster])
            # if (clusters.shape[0] == 0):
            #     clusters = np.array(cluster)
            # else:
            clusters.append(cluster)
        else:
            # Mark p as noise
            data_objects_visited[random_index] = -1
            noise.append(p)
    # Return the calculated clusters as well as the noise objects
    return clusters, noise


def plot(clusters=None, noise=None):
    if clusters == None:
        return
    num_clusters = len(clusters)
    figure, axes = plt.subplots()
    for i in range(num_clusters):
        x = list(map(lambda x: x[0], clusters[i]))
        y = list(map(lambda x: x[1], clusters[i]))
        tmp = axes.scatter(x, y)
        # axes.scatter(*centers[i, :].T, color=tmp.get_facecolor())
        # axes.annotate("Center", xy=(centers[i, 0], centers[i, 1]))
    if (noise != None):
        for i in range(len(noise)):
            print(noise[i])
            x = noise[i][0]
            y = noise[i][1]
            tmp = axes.scatter(x, y)
            # axes.scatter(*centers[i, :].T, color=tmp.get_facecolor())
            # axes.annotate("Center", xy=(centers[i, 0], centers[i, 1]))
    plt.show()


def index_of_element_in_n_d_array(element, n_d_array):
    for i in range(len(n_d_array)):
        ith_element = n_d_array[i]
        if all(ith_element == element):
            return i
    return -1


if __name__ == '__main__':
    # dataset = pd.read_csv('./data/v_2013.csv').to_numpy()
    main()
    # print(dataset.shape)
    # element = [7, 5]
    # print(index_of_element_in_n_d_array(element, dataset))
    # print(dataset)
