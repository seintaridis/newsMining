import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")



df = pd.read_csv('dataSets/train_set.csv', sep='\t')

my_additional_stop_words=['Antonia','Nikos','Nikolas']
stop_words = ENGLISH_STOP_WORDS.union(my_additional_stop_words)
count_vect = TfidfVectorizer    (stop_words=stop_words)
count_vect.fit(df['Content'].head(12266))
X_train_counts = count_vect.transform(df['Content'].head(12266))

svd = TruncatedSVD(n_components=5)
X_train_counts = svd.fit_transform(X_train_counts)

print X_train_counts.shape           #(12266, 85437) the result means

print df['Category'].iloc[5]

kmeans = KMeans(n_clusters=5)
kmeans.fit(X_train_counts)


centroids = kmeans.cluster_centers_
labels = kmeans.labels_

countPolitics=0


categorys_map={
'Politics': 0,
'Business': 1,
'Film': 2,
'Technology': 3,
'Football': 4
}

#print labels

w, h = 5, 5;
Matrix = [[0 for x in range(w)] for y in range(h)]


print categorys_map['Politics']



#print Matrix
for index in range(len(labels)):
    Matrix[labels[index]][categorys_map[df['Category'].iloc[index]]] += 1
print Matrix

#print(centroids)

colors = ["g.","r.","c.","y.","m."]

svd = TruncatedSVD(n_components=2)
X_train_counts = svd.fit_transform(X_train_counts)

#reduced = TSNE(perplexity=30.0).fit_transform(X_train_counts)

for i in range(X_train_counts.shape[0]):
    plt.plot(X_train_counts[i, 0], X_train_counts[i, 1], colors[labels[i]], markersize = 10)


plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show()


#print(labels)
