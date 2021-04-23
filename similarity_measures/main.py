
import numpy as np
from numpy.lib import math

data = np.array([
    dict(
        x=np.array([1, 1, 1, 1]),
        y=np.array([3, 3, 3, 3])
    ),
    dict(
        x=np.array([0, 2, 0, 2]),
        y=np.array([2, 0, 2, 0])
    ),
    dict(
        x=np.array([0, 1, 0, -1]),
        y=np.array([-1, 0, 1, 0])
    ),
    dict(
        x=np.array([1, 1, 0, 1, 0, 1]),
        y=np.array([1, 1, 1, 0, 0, 1])
    ),
])


def main():
    tasks = ['a)', 'b)', 'c)', 'd)']
    i = 0
    for vectors in data:
        x, y = vectors['x'], vectors['y']
        print(tasks[i])
        print('Eucledian distance:', eucledian_distance(x, y))
        print('Cosine similarity:', cosine_similarity(x, y))
        print('Jaccard index:', jaccard_index(x, y))
        print()
        i += 1


def cosine_similarity(x_0, x_1):
    # https://neo4j.com/docs/graph-algorithms/current/labs-algorithms/cosine/
    if (len(x_0) != len(x_1)):
        return 420
    num = 0
    den_x_0, den_x_1 = 0, 0
    for i in range(len(x_0)):
        num += x_0[i] * x_1[i]
        den_x_0 += math.pow(x_0[i], 2)
        den_x_1 += math.pow(x_1[i], 2)
    den_x_0 = math.sqrt(den_x_0)
    den_x_1 = math.sqrt(den_x_1)
    den = den_x_0 * den_x_1
    return num / den


def jaccard_index(x_0, x_1):
    # https://en.wikipedia.org/wiki/Jaccard_index
    if (len(x_0) == 0 and len(x_1) == 1):
        return 1

    size_of_intersection = len(np.intersect1d(x_0, x_1))
    size_of_union = len(np.union1d(x_0, x_1))
    return size_of_intersection / size_of_union


def jaccard_distance(x_0, x_1):
    return 1 - jaccard_index(x_0, x_1)


def eucledian_distance(x_0, x_1):
    return np.power(
        np.linalg.norm(x_0 - x_1, axis=0), 2)


if __name__ == '__main__':
    main()
