from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target

# Supervised
clf = LogisticRegression(max_iter=200).fit(X, y)
supervised_accuracy = clf.score(X, y)
print("Supervised prediction:", clf.predict([X[0]]))
print("Supervised accuracy:", supervised_accuracy)

# Unsupervised
kmeans = KMeans(n_clusters=10, n_init=10).fit(X)
unsupervised_labels = kmeans.labels_
print("Unsupervised labels:", unsupervised_labels)

# Simple evaluation of clustering
from sklearn.metrics import adjusted_rand_score
unsupervised_accuracy = adjusted_rand_score(y, unsupervised_labels)
print("Unsupervised clustering accuracy (Adjusted Rand Index):", unsupervised_accuracy)