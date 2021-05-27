#!/bin/python3.8

"""
    DecisionTreeClassifier is a class capable of performing multi-class classification on a dataset.
    As with other classifiers, DecisionTreeClassifier takes as input two arrays: an array X, sparse
    or dense, of shape(n_samples, n_features) holding the training samples, and an array Y of integer
    values, shape(n_samples,), holding the class labels for the training samples:
"""

import os
import sys
from optparse import OptionParser

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


def gini(p):
    return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))


def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))


def classification_error(p):
    return 1 - np.max([p, 1 - p])


def preprocess(data):
    encoder = {}
    for col in data.columns:
        label_encoder = LabelEncoder()
        label_encoder.fit(data[col])
        data[col] = label_encoder.transform(data[col])
        encoder[col] = label_encoder
    return data, encoder


def split_data_and_labels(data, class_label_column='label'):
    X = data.drop(class_label_column, axis=1)
    Y = data[class_label_column]
    return X, Y


def create_classifier(data_frame, criterion):
    try:
        target_class_name = data_frame.columns[-1]
        all_class_labels = list(set(data_frame[target_class_name].values))
        number_of_class_labels = len(all_class_labels)
        data_frame, encoder = preprocess(data_frame)
        print('Number of rows: ', len(data_frame))
        X, Y = split_data_and_labels(data_frame, target_class_name)
        # Don't split dataset into test and traning tuples
        # X_train, X_test, Y_train, Y_test = train_test_split(
        #     X, Y, test_size=None, train_size=None)
        classifier = tree.DecisionTreeClassifier(criterion=criterion).fit(X, Y)
        # Y_pred = classifier.predict(X_test)
        tree.plot_tree(classifier)
        dot_data = tree.export_graphviz(classifier, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render('decision_tree_induction_output')
        return classifier, encoder, all_class_labels
    except Exception as e:
        print(e)
        return None


def main(data_path, instances_to_classify, criterion, encode=True):
    if criterion == None:
        criterion = 'gini'
    non_encoded_instances = instances_to_classify
    data_frame = pd.read_csv(data_path)
    attributes = data_frame.columns[:-1].tolist()
    print(attributes)
    classifier, encoder, class_labels = create_classifier(
        data_frame, criterion)
    # Encoded the instances that will be classified if it is specified to do so
    encoded_instances = np.array([np.array(x) for x in instances_to_classify])
    if encode:
        for j in range(encoded_instances.shape[0]):
            instance = encoded_instances[j]
            for i in range(len(instance)):
                try:
                    col = attributes[i]
                    val = instance[i]
                    val = int(val) if val.isnumeric() else val
                    val = np.array([val])
                    encoded_val = encoder[col].transform(val)
                    encoded_instances[j][i] = encoded_val[0]
                except Exception as e:
                    print('\nError during encoding of rows to classify')
                    return 1
    if classifier == None:
        return 1
    try:
        res = [classifier.predict(test.reshape(1, -1))
               for test in encoded_instances]
        for i in range(len(res)):
            print(f'{non_encoded_instances[i]} is classified as: {res[i][0]}')
    except Exception as e:
        print(e)
        return 1
    return 0


# columns_b = ['Not_Heavy', 'Smelly', 'Spotted', 'Smooth', 'Edible']
# columns_a = ['Weekday', 'Tournament', 'Location', 'Time', 'Result']
# columns_c = ['A', 'B', 'C', 'D', 'Klasse']


def parse_decision_tree_script_arguments():
    opt_parser = OptionParser()
    opt_parser.add_option(
        '-f', '--inputFile',
        dest='input',
        help='Filename containing csv',
        default=None
    )
    opt_parser.add_option(
        '-c', '--criterion',
        dest='criterion',
        help='Measure for calculating impurity in split',
        default='gini',
        type='string'
    )
    (options, args) = opt_parser.parse_args()
    # data_path = sys.path[0] + '/data/v_2019_decision_tree.csv'
    file_name = options.input
    criteria = ['gini', 'entropy']
    # criterion = 'gini' | 'entropy'
    criterion = options.criterion if options.criterion in criteria else None
    return file_name, criterion


if __name__ == '__main__':
    instances_to_classify = [['L', 'K', 'S', 2], ['H', 'F', 'S', 4]]
    # instances_to_classify = [[1, 0, 0, 0], [1, 0, 1, 1], [1, 1, 1, 0]]

    file_name, criterion = parse_decision_tree_script_arguments()

    exit_code = main(
        file_name,
        instances_to_classify,
        criterion,
        encode=True,
    )
    print(f'\nExited with code: {exit_code}')
    os.system(
        'rm ./decision_tree_induction_output && mv ./decision_tree_induction_output.pdf ./out/ && google-chrome ./out/decision_tree_induction_output.pdf')
    #  && google-chrome ./decision_tree_induction_output.pdf


def calculate_gini_index(J, N):
    total = J + N
    return 1 - (J/total)**2 - (N/total)**2
