

import csv
import operator

import pandas as pd
from fpgrowth_py import fpgrowth, fpgrowthFromFile
# from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from termcolor import colored


def dataFromFile(file_name):
    """Function which reads from the file and yields a generator"""
    # file_iter = open(file_name, 'rU')
    file_iter = open(file_name, 'r')
    for line in file_iter:
        line = line.strip().rstrip(',')  # Remove trailing comma
        record = frozenset(line.split(','))
        yield record


def parse_csv(file_name):
    import csv
    import os
    file_exists = os.path.exists(file_name)
    if not file_exists:
        print(colored(f'The file: {file_name} does not exist', 'red'))
        return None
    data = []
    try:
        with open(file_name, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                transaction = list(row)
                data.append(transaction)
            f.close()
        return data
    except Exception as e:
        print(e)
        return None


def printResults(items, rules, num_transactions):
    print("\n------------ITEMS-----------------")
    # for item, support in sorted(items, key=operator.itemgetter(1)):
    #     print("item: %s,\t%s" % (str(item), int(float(support)*num_transactions)))
    for item in items:
        print(item)

    print("\n------------RULES-----------------")
    for rule, confidence in sorted(rules, key=operator.itemgetter(1)):
        pre, post = rule
        print("Rule: %s ==> %s,\t%.3f" % (str(pre), str(post), confidence))
    print("\n")


def main():
    dataset = parse_csv('../apriori/data/v_2019.csv')

    for transaction in dataset:
        print(transaction)

    print(f'\nNumber of transactions: {len(dataset)}\n')

    transaction_encoder = TransactionEncoder()

    transaction_encoder_array = transaction_encoder.fit(
        dataset).transform(dataset)

    data_frame = pd.DataFrame(transaction_encoder_array)

    frequent_itemsets, association_rules = fpgrowth(
        dataset, minSupRatio=0.33, minConf=0.6)

    print(association_rules)
    # printResults(frequent_itemsets, association_rules, len(dataset))
    return 0


if __name__ == '__main__':
    # print(f'\nExited with code: {main()}')
    frequent_itemsets, association_rules = fpgrowthFromFile(
        '../apriori/data/v_2019.csv', 0.33, 0.6)
    print(frequent_itemsets)
