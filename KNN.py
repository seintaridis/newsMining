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

kf = KFold(n_splits=10)
fold = 0
for train_index, test_index in kf.split(df):
    X_train_counts = count_vect.transform(df['Content'].iloc[train_index])
    X_train_counts = np.add(X_train_counts, count_vect.transform(df['Title'].iloc[train_index]))
    X_test_counts = count_vect.transform(df['Content'].iloc[test_index])
    X_test_counts = np.add(X_test_counts, count_vect.transform(df['Title'].iloc[test_index]))
    X_train_counts = svd.transform(X_train_counts)
    X_test_counts = svd.transform(X_test_counts)


    # X_train_counts = count_vect.transform(df['Content'].head(100))
    #print euclidean_distances(X_train_counts[0],X_train_counts[1])
    #print euclidean_distances(X_train_counts[1],X_train_counts[0])

    #print euclidean_distances(X_train_counts[1],X_train_counts[2])



    yPred = []
    for test, i in enumerate(test_index):
        # create list for distances and targets
        distances = euclidean_distances(X_train_counts, [X_test_counts[test]])
        distances = zip(distances, df["Category"].iloc[train_index])

        # sort the list
        distances.sort()

        # print "our test is " + df["Category"].iloc[i]
        # make a list of the k neighbors' targets
        targets = [distances[x][1] for x in range(5)]

        # print targets
        c = Counter(targets)
        # print c.most_common(1)
        #    print targets
        yPred.append(c.most_common(1)[0][0])

    print(classification_report(yPred,df['Category'].iloc[test_index], target_names=df.Category.unique()))
