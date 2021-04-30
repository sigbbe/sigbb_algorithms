#! /usr/bin/python3.8

import argparse

import numpy as np
import pandas as pd
from fpgrowth_py import fpgrowth


def remove_nan_values(np_nd_array=None):
    dataset = np_nd_array.tolist()
    for row in dataset:
        filter_array = []
        for column in row:
            filter_array.append(True if str(column) != 'nan' else False)
        print(filter_array)
        print(np.array(row)[filter_array])
        dataset[row] = np.array(row)[filter_array]
        print(row)
    return dataset

# def encode_labels(data_frame):
#     encoder = {}
#     for column in data_frame.columns:
#         label_encoder = sklearn.preprocessing.LabelEncoder()
#         label_encoder.fit(data_frame[column])
#         data_frame[column] = label_encoder.transform(data_frame[column])
#         encoder[column] = label_encoder
#     return data_frame, encoder

# mushrooms_encoded_df, encoder = encode_labels(mushrooms_df)


def main(dataset, print_freq=False, print_rules=False):
    dataset = remove_nan_values(dataset)
    print(dataset)
    # freqItemSet, rules = fpgrowth(dataset, minSupRatio=0.6, minConf=0.5)
    if print_freq:
        for itemset in freqItemSet:
            print(itemset)
    if print_rules:
        for rule in rules:
            print(rule)


if __name__ == '__main__':
    data = np.array([
        (1, ['A', 'C', 'D', 'F', 'G', 'I', 'M', 'P']),
        (2, ['A', 'B', 'C', 'F', 'L', 'M', 'O']),
        (3, ['B', 'F', 'H', 'J', 'O']),
        (4, ['B', 'C', 'K', 'P', 'S']),
        (5, ['A', 'C', 'E', 'F', 'L', 'M', 'N', 'P']),
    ], dtype=object)
    data_frame = pd.read_csv('data/oeving_5_formatted.csv')
    main(data_frame.to_numpy())
    # for row in data:
    #     _id = row[0]
    #     _data = row[1]
    #     filter_array = []
    #     for column in _data:
    #         filter_array.append(True if column != 'A' else False)
    #     _data = np.array(_data)[filter_array]
    #     print(_id, _data)
