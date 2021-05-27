import operator
import sys
from collections import defaultdict
from itertools import chain, combinations
from optparse import OptionParser

import pandas as pd

__author__ = 'Not me'


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
       of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)

    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet


def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
    return itemSet, transactionList


def runApriori(data_iter, minSupport, minConfidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """

    itemSet, transactionList = getItemSetTransactionList(data_iter)
    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict()
    # Dictionary which stores Association Rules

    oneCSet = returnItemsWithMinSupport(itemSet,
                                        transactionList,
                                        minSupport,
                                        freqSet)
    currentLSet = oneCSet
    k = 2
    k_candidates = dict()
    while currentLSet != set([]):
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        k_candidates[k] = currentLSet
        currentCSet = returnItemsWithMinSupport(currentLSet,
                                                transactionList,
                                                minSupport,
                                                freqSet)
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item))
                           for item in value])

    toRetRules = []
    # for key, value in largeSet.items()[1:]:
    for key, value in list(largeSet.items())[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = getSupport(item) / getSupport(element)
                    if confidence >= minConfidence:
                        toRetRules.append(
                            ((tuple(element), tuple(remain)), confidence))
    return toRetItems, toRetRules, k_candidates


def printResults(items, candidates, rules, num_transactions):
    print("\n------------ITEMS-----------------")
    for item, support in sorted(items, key=operator.itemgetter(1)):
        print("%s,\t\t\t%s" %
              (''.join(list(item)), int(float(support)*num_transactions)))

    print("\n------------Candidates-----------------")

    for k in candidates.keys():
        print(f'k = {k}:')
        k_candidates = candidates[k]
        candidate = k_candidates.pop() if k_candidates != set([]) else None
        while k_candidates != set([]):
            candidate = ''.join([list(candidate) for x in candidate][0])
            print(candidate)
            # candidate = []
            candidate = k_candidates.pop()

    if len(rules) < 1:
        return None

    print("\n------------RULES-----------------")
    for rule, confidence in sorted(rules, key=operator.itemgetter(1)):
        pre, post = rule
        print("%s ==> %s,\t\t%.3f" %
              (''.join(list(pre)), ''.join(list(post)), confidence))
    print("\n")
    return 0


def number_of_transactions(file, head=False):
    import csv
    count = -1
    try:
        with open(file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = None
            if head:
                next(reader)
            count = 0
            for row in reader:
                if len(row) > 0:
                    count += 1
    except Exception as e:
        print(e)
        return -1
    return count


def dataFromFile(file_name):
    """Function which reads from the file and yields a generator"""
    # file_iter = open(file_name, 'rU')
    file_iter = open(file_name, 'r')
    for line in file_iter:
        line = line.strip().rstrip(',')  # Remove trailing comma
        record = frozenset(line.split(','))
        yield record


def print_1_itemset(file_name):
    data = dataFromFile(file_name)
    counts = {}
    itemSet, transactionList = getItemSetTransactionList(data)
    print(next(itemSet.pop()))
    return 1


min_sup = 0.0
conf = 0.0
if __name__ == "__main__":

    opt_parser = OptionParser()
    opt_parser.add_option(
        '-f', '--inputFile',
        dest='input',
        help='filename containing csv',
        default=None
    )
    opt_parser.add_option(
        '-s', '--minSupport',
        dest='minS',
        help='minimum support value',
        default=0.15,
        type='float'
    )
    opt_parser.add_option(
        '-c', '--minConfidence',
        dest='minC',
        help='minimum confidence value',
        default=0.6,
        type='float'
    )
    opt_parser.add_option(
        '--count-1-itemsets',
        dest='count_1_itemsets',
        action='store_true',
        help='Count all k=1 itemsets',
        default=False,
    )
    (options, args) = opt_parser.parse_args()

    inFile = None
    file_path = None
    if options.input is None:
        inFile = sys.stdin
        file_path = sys.stdin
    elif options.input is not None:
        file_path = options.input
        inFile = dataFromFile(options.input)
    else:
        print('No dataset filename specified, system with exit\n')
        sys.exit('System will exit')

    if (options.count_1_itemsets):
        print_1_itemset(file_path)

    num_transactions = number_of_transactions(file_path)
    minSupport = options.minS
    min_sup = minSupport
    minConfidence = options.minC
    conf = minConfidence

    items, rules, candidates = runApriori(inFile, minSupport, minConfidence)
    printResults(items, candidates, rules, num_transactions)
