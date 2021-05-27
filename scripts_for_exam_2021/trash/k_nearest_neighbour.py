
import os
from optparse import OptionParser

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

print(__doc__)

name_of_the_target_attribute = ''

h = 0.02


# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ['darkorange', 'c', 'darkblue']


def preprocess(data):

    return None, None


def main(dataset, k):
    try:
        nearest_neighbours = NearestNeighbors(
            n_neighbors=k, algorithm='kd_tree').fit(dataset)
        distances, indices = nearest_neighbours.kneighbors(dataset)
        print('\n\n', distances, '\n\n', indices)
        for weights in ['uniform', 'distance']:
            # we create an instance of Neighbours Classifier and fit the data.
            classifier = KNeighborsClassifier(k, weights=weights)
            rows, targets = preprocess(dataset)
            classifier.fit(rows, targets)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = rows[:, 0].min() - 1, rows[:, 0].max() + 1
            y_min, y_max = rows[:, 1].min() - 1, rows[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, cmap=cmap_light)

            # Plot also the training points
            sns.scatterplot(x=rows[:, 0], y=rows[:, 1],
                            palette=cmap_bold, alpha=1.0, edgecolor="black")
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title("3-Class classification (k = %i, weights = '%s')"
                      % (k, weights))
            plt.xlabel('iris.feature_names[0]')
            plt.ylabel('iris.feature_names[1]')

        plt.show()
        return 0
    except Exception as e:
        print(e)
        return -1


if __name__ == '__main__':
    opt_parser = OptionParser()
    opt_parser.add_option(
        '-f', '--inputFile',
        dest='input',
        help='filename containing csv',
        default=None
    )
    opt_parser.add_option(
        '-k', '--k-nearest',
        dest='kValue',
        help='algorithm parameter k',
        default=2
    )
    options, args = opt_parser.parse_args()
    print(f'PATH: {options.input}')
    if options.input == None:
        print(
            f'\nExited with code: {1}, script must be called with -f flags set followed by path to csv file')
        exit(1)
    elif not os.path.exists(options.input):
        print(f'\nExited with code: {1}, file does not exist')
        exit(1)
    else:
        data_frame = pd.read_csv(options.input)
        dataset = data_frame.to_numpy()
        k = options.kValue
        exit_code = main(dataset, k)
        print(f'\nExited with code: {exit_code}')
