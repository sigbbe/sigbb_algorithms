
import graphviz
from sklearn import tree

X = [[1, 0, 2, 0, 0], [2, 0, 2, 0, 1], [3, 1, 2, 0, 0], [4, 2, 1, 0, 0], [5, 2, 0, 0, 0], [6, 2, 0, 1, 1], [7, 1, 0, 1, 1], [8, 0, 1, 0, 0], [9, 0, 0, 1, 0], [10, 2, 1, 1, 0], [
    11, 0, 1, 1, 1], [12, 1, 1, 0, 1], [13, 1, 2, 1, 0], [14, 2, 1, 0, 1], [15, 1, 1, 1, 0], [16, 1, 1, 1, 1], [17, 0, 0, 1, 1], [18, 2, 2, 0, 0], [19, 2, 0, 0, 1], [20, 0, 1, 1, 1]]
Y = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1]

if __name__ == '__main__':
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    print('Plot tree')
    plot = tree.plot_tree(clf)
    print(plot)
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    # feature_names=iris.feature_names,
                                    # class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.view()
