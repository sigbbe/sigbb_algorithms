from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

if __name__ == '__main__':
    n_digits = 4
    labels = ['Something', 'Something else']
    kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4,
                    random_state=0)
    kmeans.fit_predict(n_digits)

print(82 * '_')
