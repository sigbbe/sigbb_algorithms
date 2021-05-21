#!/bin/python3.8

import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

columns = ['Weekday', 'Tournament', 'Location', 'Time', 'Result']


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset"""
    col_index = columns.index(col)
    return set([row[col_index] for row in rows])


def class_counts(rows, class_label_name):
    """Counts the number of each type of example in a dataset"""
    counts = {}  # A dictionary of label ->
    for row in rows:
        key = columns.index(class_label_name)
        counts[key] = counts[key] + 1 if counts.has_key(key) else 1
    return counts


def main():
    try:
        path_to_data = sys.path[0] + '/data/v_2015.csv'
        data_frame = pd.read_csv(path_to_data)
        data_set = data_frame.values.tolist()
        print(unique_vals(data_set, 'Location'))
        print(class_counts(data_set, 'Result'))
    except Exception as err:
        print(err)
        return 1
    return 0


if __name__ == '__main__':
    print(f'\nExited with code: {main()}')
