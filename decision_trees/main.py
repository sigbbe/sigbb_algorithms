#! /usr/bin/python3.8

from anytree import Node, RenderTree

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


def gini_index(data_partition=None):
    if data_partition == None:
        return -1
    len_of_d = len(data_partition)

    p_i = None
    return data_partition


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
    # main()
    a = 3.14
    a_prime = 3.145
    print(hash(a))
    print(hash(a_prime))
