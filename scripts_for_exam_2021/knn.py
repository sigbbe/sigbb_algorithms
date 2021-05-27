from termcolor import colored
# Distance calculation


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Known data
data = {
    "P1": (4, 8),
    "P2": (8, 8),
    "P3": (8, 4),
    "P4": (6, 7),
    "P5": (1, 10),
    "P6": (3, 6),
    "P7": (2, 4),
    "P8": (1, 7),
    "P9": (6, 4),
    "P10": (6, 2),
    "P11": (6, 3),
    "P12": (4, 3),
    "P13": (4, 4),
}

# Clusters
clusters = {
    "c1": ["P1", "P2", "P3", "P4"],
    "c2": ["P5", "P6", "P7", "P8"],
    "c3": ["P9", "P10", "P11", "P12", "P13"]
}

# Do be defined
samples = {
    "A": (6, 6),
    "B": (4, 6),
    "C": (4, 5),
    "D": (2, 6)
}


def determine_k_closest(sample_coord, known_data, k):
    # Store all distances
    distances = []

    # Calculate distance from sample to all orther coordinates
    for label, coordinate in known_data.items():
        distance = manhattan(sample_coord, coordinate)
        distances.append({label: coordinate, "distance": distance})

    # Sort the list and return k nearest elements
    k_closest = sorted(distances, key=lambda key: key['distance'])[:k]

    return k_closest


def determine_label(nodes, clusters):
    cluster_count = {}
    # Iterate clusters
    for cluster_key, value in clusters.items():
        cluster_count[str(cluster_key)] = 0

        # Check all node against cluster
        for node in nodes:
            label = list(node.keys())[0]

            # If label is in cluster, increment appearance
            if label in value:
                cluster_count[str(cluster_key)
                              ] = cluster_count[str(cluster_key)] + 1

    # Return cluster with most appearances
    return max(cluster_count, key=cluster_count.get)


def knn(samples, data, clusters):
    print(colored("Data:", "green"))
    for key, value in data.items():
        print("%s: \t%s" % (key, value))

    print(colored("\nSamples:", "blue"))
    for key, value in samples.items():
        print("%s: \t%s" % (key, value))

    print(colored("\nClusters:", "red"))
    for key, value in clusters.items():
        print("%s: \t%s" % (key, value))

    print(colored("\nDetermined samples:", "yellow"))
    for key, value in samples.items():
        closest = determine_k_closest(samples[str(key)], data, 3)
        determined_label = determine_label(closest, clusters)
        print("%s belongs to %s " % (key, determined_label))


knn(samples, data, clusters)
