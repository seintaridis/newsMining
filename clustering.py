from __future__ import division
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import cosine_distance
import numpy as nmp
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


# Preproccessing

# read trainset

df = pd.read_csv('dataSets/train_set.csv', sep='\t')

# remove stop words

my_additional_stop_words=['said','th','month','much','thing','say','says']
stop_words = ENGLISH_STOP_WORDS.union(my_additional_stop_words)

# create counter and idf vectors

count_vect = TfidfVectorizer    (stop_words=stop_words)
count_vect.fit(df['Content']) #12266
X_train_counts = count_vect.transform(df['Content']

# reduce size of vector with LSI

svd = TruncatedSVD(n_components=5)
X_train_counts = svd.fit_transform(X_train_counts)


# Clustering

kclusterer = KMeansClusterer(num_means = 5, distance=cosine_distance, repeats=25, avoid_empty_clusters= True)
clusters = kclusterer.cluster(X_train_counts, assign_clusters=True)
# print "Clusters:\n " , clusters
# print "Means" , kclusterer.means()


# Prepare results Matrix

categories_map={
'Politics': 0,
'Business': 1,
'Film': 2,
'Technology': 3,
'Football': 4
}

labels_map={
'Cluster1': 0,
'Cluster2': 1,
'Cluster3': 2,
'Cluster4': 3,
'Cluster5': 4
}

w, h =  len(labels_map), len(categories_map);

Matrix = nmp.array ([[0 for x in range(w)] for y in range(h)], dtype=float)
rowsum = [0 for x in range(w)]

# labels = kclusterer.cluster_names()
# means = kclusterer.means()

for index in range(len(clusters)): # 0 to 19
    Matrix[clusters[index]][categories_map[df['Category'].iloc[index]]] += 1
print "\nWords in clusters per category:\n", Matrix

rowsum = nmp.sum(Matrix, axis=1)
# print "rowsum:\n", rowsum

for i in range(w):
	for j in range(h):
		x  = Matrix[i][j] / rowsum[i]
		Matrix[i][j] = float("{0:.2f}".format(x))

print "\nPercentage of words in clusters per category\n", Matrix

colors = ["g.","r.","c.","y.","m."]

svd = TruncatedSVD(n_components=2)
X_train_counts = svd.fit_transform(X_train_counts)

#reduced = TSNE(perplexity=30.0).fit_transform(X_train_counts)

for i in range(X_train_counts.shape[0]):
    plt.plot(X_train_counts[i, 0], X_train_counts[i, 1],colors[clusters[i]], markersize = 10)

# plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

labels_sorted = sorted(labels_map.keys())

df_cluster = pd.DataFrame(Matrix, index = labels_sorted, columns = categories_map)

df_cluster.to_csv('clustering_KMeans.csv', sep='\t')
plt.show()
