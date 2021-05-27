#!/bin/python3.8

"""
    DecisionTreeClassifier is a class capable of performing multi-class classification on a dataset.
    As with other classifiers, DecisionTreeClassifier takes as input two arrays: an array X, sparse
    or dense, of shape(n_samples, n_features) holding the training samples, and an array Y of integer
    values, shape(n_samples,), holding the class labels for the training samples:
"""

import sys

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


def create_classifier(data_frame):
    try:
        all_class_labels = list(set(data_frame['Result'].values))
        number_of_class_labels = len(all_class_labels)
        data_frame, encoder = preprocess(data_frame)
        X, Y = split_data_and_labels(data_frame, 'Result')
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.20)
        classifier = tree.DecisionTreeClassifier().fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        tree.plot_tree(classifier)
        dot_data = tree.export_graphviz(classifier, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render('decision_tree_induction_output')
        return classifier, encoder, all_class_labels
    except Exception as e:
        print(e)
        return None


columns_a = ['Weekday', 'Tournament', 'Location', 'Time', 'Result']
columns_b = ['Example', 'Not Heavy', 'Smelly', 'Spotted', 'Smooth', 'Edible']


def main(encode=True):
    data_path = sys.path[0] + '/data/v_2015.csv'
    non_encoded_instances = instances_to_classify
    data_frame = pd.read_csv(data_path)
    classifier, encoder, class_labels = create_classifier(data_frame)
    encoded_instances = None
    if encode:
        encoded_instances = [instance[i]
                             for instance in instances_to_classify for i in range(len(instance))]
    for i in range(len(columns) - 1):
        col = columns[i]
        encoded_instances[i] = encoder[list(encoder.keys())[i]].transform(
            np.array([encoded_instances[i]]))
    if classifier == None:
        return 1
    print(encoded_instances, columns[0:len(columns) - 1])
    encoded_instance = pd.DataFrame(
        encoded_instances, columns[0:len(columns) - 1]).iloc[:, :4].values.T
    res_indexes = [classifier.predict(instance)[0]
                   for instance in encoded_instances]
    res = [class_labels[res_i] for res_i in res_indexes]
    try:
        for i in range(len(res)):
            print(
                f'{non_encoded_instances[i]} is classified as: {res[i]}')
    except Exception as e:
        print(e)
        return 1
    return 0


instances_to_classify = None

if __name__ == '__main__':
    # instances_to_classify = [['Friday', 'Elite_League', 'H', 'Evening', 'R'], [
    # 'Sunday', 'Elite_League', 'A', 'Evening', 'V']]
    # columns = columns_a
    # exit_code = main()
    data_frame = pd.read_csv('./data/v_2019_decision_tree_model.csv')
    print(data_frame)
    print(f'\nExited with code: {1}')
