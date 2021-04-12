import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

x = np.array([[1, 1], [3, 3], [3, 4], [2, 4], [6, 5], [7, 6], [7, 8], [6, 10], [12, 4], [
             5, 11], [6, 11], [5, 10], [16, 8], [11, 9], [13, 8], [10, 7], [12, 8], [15, 3]])

db = DBSCAN(eps=2, min_samples=3).fit(x)

labels = db.labels_
cores = db.core_sample_indices_

# Not familiar with Numpy at all, so refining all lists to regular python
refined_x = []
for i in range(len(x)):
    refined_x.append([f"P{i + 1}", list(x[i])])

print("All points:")
# Extracting noisy points
noisy_points = []
for i in range(len(labels)):
    opt = f"P{i + 1} ({x[i][0]},{x[i][1]})"
    if labels[i] != -1:
        opt += "\t=> cluster " + str(labels[i] + 1)
    else:
        noisy_points.append([f"P{i + 1}", list(x[i])])
        opt += "\t=> NOISE"
    print(opt)

refined_cores = []
for i in range(len(cores)):
    refined_cores.append([f"P{cores[i] + 1}", list(x[cores[i]])])

# Remove cores from all points
print("\nCores:")
y = list.copy(refined_x)
for core in refined_cores:
    print(f"{core[0]} ({core[1][0]}, {core[1][1]})")
    y.remove(core)

# Remove noise
print("\nNoise:")
for noise in noisy_points:
    print(f"{noise[0]} ({noise[1][0]}, {noise[1][1]})")
    y.remove(noise)

# Borders
print("\nBorder points:")
for p in y:
    print(f"{p[0]} ({p[1][0]}, {p[1][1]})")


def plot_clusters(data, centroids):
    """
    Shows a scatter plot with the data points clustered according to the centroids.
    """
    # Assigning the data points to clusters/centroids.
    clusters = [[] for _ in range(centroids.shape[0])]
    for i in range(data.shape[0]):
        distances = np.linalg.norm(data[i] - centroids, axis=0)
        clusters[np.argmin(distances)].append(data[i])

    # Plotting clusters and centroids.
    fig, ax = plt.subplots()
    for c in range(centroids.shape[0]):
        if len(clusters[c]) > 0:
            cluster = np.array(clusters[c])
            ax.scatter(cluster[:, 0], cluster[:, 1], s=7)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red')


# print(cores)
# plot_clusters(x, cores)
