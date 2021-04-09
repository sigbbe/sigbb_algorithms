from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#
# Load IRIS dataset
#
iris = datasets.load_iris()
X = iris.data
y = iris.target
#
# Instantiate the KMeans models
#
km = KMeans(n_clusters=3, random_state=42)
#
# Fit the KMeans model
#
km.fit_predict(X)
#
# Calculate Silhoutte Score
#
score = silhouette_score(X, km.labels_, metric='euclidean')
#
# Print the score
#
print(km.labels_)
print('Silhouetter Score: %.3f' % score)
