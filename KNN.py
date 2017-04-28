import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter

style.use("ggplot")
df = pd.read_csv('dataSets/train_set.csv', sep='\t')

my_additional_stop_words=['Antonia','Nikos','Nikolas']
stop_words = ENGLISH_STOP_WORDS.union(my_additional_stop_words)

count_vect = TfidfVectorizer(stop_words=stop_words)
count_vect.fit(df['Content'].head(100))
svd = TruncatedSVD(n_components=5)
svd.fit(count_vect.transform(df['Content'].head(100)))
X_train_counts = count_vect.transform(df['Content'].head(100))
#print euclidean_distances(X_train_counts[0],X_train_counts[1])
#print euclidean_distances(X_train_counts[1],X_train_counts[0])

#print euclidean_distances(X_train_counts[1],X_train_counts[2])



# create list for distances and targets
distances = []
targets = []

for i in range(X_train_counts.shape[0]):
		# first we compute the euclidean distance
		distance = euclidean_distances(X_train_counts[i],X_train_counts[1])
		# add it to list of distances
		distances.append([distance, i])

	# sort the list
distances = sorted(distances)

print "our test is " + df["Category"].iloc[1]
	# make a list of the k neighbors' targets
for i in range(99):
#    print df["Category"].iloc(2)
#    print df["Category"].iloc(distances[i][1])
    targets.append(df["Category"].iloc[distances[i][1]])
	#index = distances[i][1]
	#targets.append(X_train_counts[index])


	# return most common target
print targets
c = Counter(targets)
print c.most_common(1)
#    print targets
