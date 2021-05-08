
import numpy as np
from numpy.lib import math
from scipy.spatial import distance
from scipy.stats import pearsonr
from sklearn.metrics import jaccard_score as scikit_jaccard
from sklearn.metrics.pairwise import cosine_similarity as scikit_cosine
from sklearn.metrics.pairwise import euclidean_distances as scikit_euclidean

from .sigbbe_algorithms.utils import plot

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


def gini_index(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


def dot_product(a, b):
    sum = 0
    for i in range(len(a)):
        sum += a[i]*b[i]
    return sum


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


def my_cosine_similarity(x_0, x_1):
    return None


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


def main():
    tasks = ['a)', 'b)', 'c)', 'd)']
    i = 0
    for vectors in data:
        x, y = vectors['x'], vectors['y']
        print()
        print(tasks[i])
        print(f'x={x}')
        print(f'y={y}')
        print(f'Eucledian distance:', distance.euclidean(x, y))
        print(
            f'Cosine distance/similarity: {distance.cosine(x, y)}/{cosine_similarity(x, y)}')
        print(f'Jaccard index:', (1) *
              calculate_motherfucking_jaccard_distance(x, y))
        print(f'Pearson’s product moment coefficient: {pearsonr(x, y)[0]}')
        print(f'Gini index: {gini_index(x, y)}')
        print()
        i += 1


def calculate_motherfucking_jaccard_distance(a, b):
    _den, _num = len(a), 0
    for i in range(len(a)):
        if (a[i] == b[i]):
            _num += 1
        if (b[i] not in a):
            _den += 1

    return _num / _den


if __name__ == '__main__':
    print(plot)
