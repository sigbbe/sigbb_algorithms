#! /usr/bin/python3.8

import collections

import numpy as np
import pandas as pd
from anytree import Node, RenderTree
from numpy.lib import math


def correlation_coefficient(x, y):
    """
    Description ...
    Args:
        x: n dimensional vector
        y: n dimensional vector
    Returns:
       The computed Pearson’s product moment coefficient for the two argument vectors.
    """
    n = len(x)
    if not(len(y) == n):
        return -2
    r_of_x_y = 0
    std_x, std_y = np.std(x), np.std(y)
    mean_of_x, mean_of_y = np.mean(x), np.mean(y)
    for i in range(n):
        r_of_x_y += (x[i] - mean_of_x) * (y[i] - mean_of_y)
    print(std_y)
    return r_of_x_y / (n * std_x * std_y)


def get_flat_file_data_frame(url):
    """
    Description

    Args:
        url (string): path to the csv file.

    Returns:
        data (ndarray): an n-dimensional array containing the data of the specified file.
    """
    df = pd.read_csv(url, keep_default_na=False,
                     encoding='utf-8', chunksize=1)
    return np.array([chunk.values.tolist()[0] for chunk in df])


# Data:
#               Age: {Young (0), Middle (1), Old (2)}
#            Income: {Low (0), Medium (1), High (2)}
#           Student: {Yes (0), No (1)}
#  Creditworthiness: {Pass (0), High (1)}
#      PC on Credit: {Yes (0), No (1)}

age = ['Young', 'Middle', 'Old']
income = ['Low', 'Medium', 'High']
student = ['Yes', 'No']
creditworthiness = ['Pass', 'High']
pc_on_credit = student
columns = ['ID', age, income, student, creditworthiness, pc_on_credit]


def gini_index(data_partition=None):
    shape = data_partition.shape
    if len(shape) == 1:
        len_of_d = len(data_partition)
        p_i = None
        # Probability that a tuple i in D belongs to class C_i
        # p_i = |C_i,D|/|D|
        counter = collections.Counter(data_partition)
        zum = 0
        for i in range(len(data_partition)):
            c_i_d = counter[data_partition[i]]
            p_i = c_i_d / len_of_d
            zum = math.pow(p_i, 2)
        return 1 - zum
    elif len(shape) == 2:
        return 0
    return -1


def gini(x):
    mean_absolute_distance = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    relative_mean_absolute_distance = mean_absolute_distance/np.mean(x)
    # Gini coefficient
    # This implementation of the Gini coefficient uses the fact that the
    # Gini coefficient is half the relative mean absolute difference.
    return 0.5 * relative_mean_absolute_distance


def is_discrete(x):
    return x != None


def is_empty(_list):
    try:
        return len(_list) < 1
    except Exception as e:
        print(e.message)
        return True


def get_majority_class(data_partition):
    return hash(data_partition)


def generate_decision_tree(data_partition=None, attribute_list=None, attribute_selection_method=None, multivalued_splits_allowed=True):
    restricted_to_binary_trees = True

    if (data_partition == None or attribute_list == None or attribute_selection_method == None):
        raise ValueError("All arguments must be defined")

    # Create a node N
    n = Node()

    # Are all tuples in D of the same class, C?
    classes = set(map(data_partition, lambda data: data['label']))
    if (is_empty(classes)):
        # Yes, they are
        return Node()

    # No, there is more than one label

    splitting_criterion = attribute_selection_method(
        data_partition, attribute_list)
    n = Node(splitting_criterion)

    if (is_discrete(splitting_criterion['splitting_attribute']) and multivalued_splits_allowed):
        # Remove splitting attribute
        attribute_list = filter(attribute_list, lambda attribute: not(
            attribute in splitting_attribute))

    # Partition the tuples and grow subtrees for each partition
    for outcome in splitting_criterion:
        # Let Dj be the set of data tuples in D satisfying outcome j, a partition
        j = 0
        d_j = data_partition[j]
        if is_empty(d_j):
            majority_class_in_data_partition = get_majority_class(
                data_partition)
            new_node = Node(majority_class_in_data_partition, parent=n)
        else:
            return
    return None


def main():
    udo = Node("Udo")
    marc = Node("Marc", parent=udo)
    lian = Node("Lian", parent=marc)
    dan = Node("Dan", parent=udo)
    jet = Node("Jet", parent=dan)
    jan = Node("Jan", parent=dan)
    joe = Node("Joe", parent=dan)
    print(udo)
    Node('/Udo')
    print(joe)
    Node('/Udo/Dan/Joe')
    for pre, fill, node in RenderTree(udo):
        print("%s%s" % (pre, node.name))
    print(dan.children)
    (Node('/Udo/Dan/Jet'), Node('/Udo/Dan/Jan'), Node('/Udo/Dan/Joe'))

    print('*'*20)
    print(udo)


if __name__ == '__main__':
    dataset = get_flat_file_data_frame('./data/oeving_5.csv')
    n_dimensions = len(dataset.shape)
    dataset_transposed = dataset.T
    for i in range(len(dataset_transposed)):
        column = dataset_transposed[i]
        # print(
        #     f'Standard deviation of {columns[i]}: {np.std(dataset_transposed[i])}')
    x_0 = np.array([1, 1, 1, 1])
    y_0 = [3, 3, 3, 3]
    pc_on_credit = dataset_transposed[5]
    print(len(list(pc_on_credit[pc_on_credit == 1])))

    print(
        f'Gini index for dataset: {float(1 - (math.pow(12/20, 2) + math.pow(8/20, 2)))}')
