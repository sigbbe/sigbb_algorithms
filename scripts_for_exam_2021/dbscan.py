import csv
import re
import sys
from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import (cityblock, correlation, cosine, euclidean,
                                    minkowski)


class DBScanner:

    def __init__(self, config):
        self.eps = config['eps']
        self.min_pts = config['min_pts']
        self.dim = config['dim']
        self.clusters = set()
        self.cluster_count = 0
        self.visited = []
        dist_arg = config['dist_measure']
        print(f'MinPts: {self.min_pts}')
        print(f'Eps: {self.eps}')
        print(f'Distance measure: {dist_arg}')
        if dist_arg == 'manhattan' or dist_arg == 'city-block':
            self.distance_measure = euclidean
        elif dist_arg == 'euclidean':
            self.distance_measure = cityblock
        elif dist_arg == 'cosine':
            self.distance_measure = cosine
        elif dist_arg == 'minkowski':
            self.distance_measure = cosine
        else:
            print(f'Distance measure not found, defaulting to euclidean distance...\n')
            self.distance_measure = euclidean
        self.color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    def dbscan(self, data):
        self.init_params()
        self.data = data
        # Setting up the plot
        fig = plt.figure()

        axis_proj = 'rectilinear'
        if self.dim > 2:
            axis_proj = '%dd' % self.dim

        ax = fig.add_subplot(111, projection=axis_proj)

        # default noise cluster
        noise = Cluster('Noise', self.dim)
        self.clusters.add(noise)

        for point in data:
            if point not in self.visited:
                self.visited.append(point)
                neighbour_pts = self.region_query(point)
                number_of_points_in_neighborhood = len(neighbour_pts)
                if number_of_points_in_neighborhood < self.min_pts:
                    noise.add_point(point)
                else:
                    name = 'cluster-%d' % self.cluster_count
                    new_cluster = Cluster(name, self.dim)

                    self.cluster_count += 1
                    self.expand_cluster(new_cluster, point, neighbour_pts)

                    if self.dim == 2:
                        ax.scatter(new_cluster.get_X(), new_cluster.get_Y(), c=self.color[self.cluster_count % len(self.color)],
                                   marker='o', label=name)
                    elif self.dim == 3:
                        ax.scatter(new_cluster.get_X(), new_cluster.get_Y(), new_cluster.get_Z(), marker='o',
                                   c=self.color[self.cluster_count % len(self.color)], label=name)
                    # ax.hold(True)

        if len(noise.get_points()) != 0:
            if self.dim > 2:
                ax.scatter(noise.get_X(), noise.get_Y(),
                           noise.get_Z(), marker='x', label=noise.name)
            else:
                ax.scatter(noise.get_X(), noise.get_Y(),
                           marker='x', label=noise.name)

        print("Number of clusters found: %d" % self.cluster_count)

        # ax.hold(False)
        ax.legend(loc='lower left')
        ax.grid(True)
        plt.title(r'DBSCAN Clustering', fontsize=18)
        plt.show()

    def expand_cluster(self, cluster, point, neighbour_pts):
        cluster.add_point(point)
        for p in neighbour_pts:
            if p not in self.visited:
                self.visited.append(p)
                np = self.region_query(p)
                if len(np) >= self.min_pts:
                    for n in np:
                        if n not in neighbour_pts:
                            neighbour_pts.append(n)

                for other_cluster in self.clusters:
                    if not other_cluster.has(p):
                        if not cluster.has(p):
                            cluster.add_point(p)

                if self.cluster_count == 0:
                    if not cluster.has(p):
                        cluster.add_point(p)

        self.clusters.add(cluster)

    def get_distance(self, from_point, to_point):
        p1 = [from_point['value'][k] for k in range(self.dim)]
        p2 = [to_point['value'][k] for k in range(self.dim)]
        # print(f'dist({p1}, {p2}) = {self.distance_measure(p1, p2)}')
        return self.distance_measure(p1, p2)

    def region_query(self, point):
        result = []
        for d_point in self.data:
            if d_point != point:
                if self.get_distance(d_point, point) <= self.eps:
                    result.append(d_point)
        return result

    def export(self, export_file="./out/cluster_dump"):
        with open(export_file, 'w') as dump_file:
            for cluster in self.clusters:
                for point in cluster.points:
                    csv_point = ','.join(map(str, point['value']))
                    dump_file.write("%s;%s\n" % (csv_point, cluster.name))
        print("Cluster dumped at: %s" % export_file)

    def init_params(self):
        self.clusters = set()
        self.cluster_count = 0
        self.visited = []


class Cluster(object):
    def __init__(self,  name, dim):
        self.name = name
        self.dim = dim
        self.points = []
        self.core_points = []
        self.border_points = []

    def add_point(self, point):
        self.points.append(point)

    def get_points(self):
        return self.points

    def erase(self):
        self.points = []

        self.core_points = []
        self.border_points = []

    def get_X(self):
        return [p['value'][0] for p in self.points]

    def get_Y(self):
        return [p['value'][1] for p in self.points]

    def get_Z(self):
        if self.dim > 2:
            return [p['value'][2] for p in self.points]
        return None

    def has(self, point):
        return point in self.points

    def __str__(self):
        return "%s: %d points" % (self.name, len(self.points))


def get_data(config):
    data = []
    with open(config['file'], 'r') as file_obj:
        csv_reader = csv.reader(file_obj)
        for id_, row in enumerate(csv_reader):
            print(id_, row)
            if len(row) < config['dim']:
                print("ERROR: The data you have provided has fewer \
                    dimensions than expected (dim = %d < %d)"
                      % (config['dim'], len(row)))
                sys.exit()
            else:
                point = {'id': id_, 'value': []}
                for dim in range(0, config['dim']):
                    point['value'].append(float(row[dim]))
                data.append(point)
    return data


def parse_options():
    opt_parser = OptionParser()
    opt_parser.add_option(
        '-f', '--input-file',
        dest='input',
        default=None
    )
    opt_parser.add_option(
        '-p', '--min-points',
        dest='min_pts',
        default=4,
        type='int'
    )
    opt_parser.add_option(
        '-e', '--epsilon',
        dest='eps',
        default=3,
        type='int'
    )
    opt_parser.add_option(
        '-d', '--distance-measure',
        dest='dist',
        default='euclidean',
        type='string'
    )
    opt_parser.add_option(
        '--dimensions',
        dest='dim',
        default=2,
        type='int'
    )
    (options, args) = opt_parser.parse_args()
    opt = [
        options.input,
        options.min_pts,
        options.eps,
        options.dist,
        options.dim
    ]
    dbscan_options = dict()
    args = ['file', 'min_pts', 'eps', 'dist_measure', 'dim']
    opt = [x for x in opt if x != '' and x != None]
    for i in range(len(args)):
        argument = args[i]
        dbscan_options[argument] = opt[i]

    return dbscan_options


def main():
    config = parse_options()
    dbc = DBScanner(config)
    data = get_data(config)
    dbc.dbscan(data)
    dbc.export()


if __name__ == "__main__":
    main()
