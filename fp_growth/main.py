#! /usr/bin/python3.8

import argparse

import numpy as np
import pandas as pd
from fpgrowth_py import fpgrowth


def main(dataset, print_freq=True, print_rules=False):
    freqItemSet, rules = fpgrowth(dataset, minSupRatio=0.6, minConf=0.5)
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
    dataset = []
    for row in data:
        dataset.append(row[1])
    main(dataset)
