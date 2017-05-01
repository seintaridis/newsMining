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

style.use("ggplot")
df = pd.read_csv('dataSets/train_set.csv', sep='\t')

my_additional_stop_words=['Antonia','Nikos','Nikolas']
stop_words = ENGLISH_STOP_WORDS.union(my_additional_stop_words)


#svd = TruncatedSVD(n_components=5)
#X_train_counts = svd.fit_transform(X_train_counts)

#print X_train_counts.shape           #(12266, 85437) the result means


#clf = MultinomialNB().fit(X_train_counts,df.Category.head(10000))  #naive_bayes
#clf = RandomForestClassifier(n_estimators=5).fit(X_train_counts,df.Category.head(10)) #RandomForestClassifier
#clf = svm.SVC().fit(X_train_counts,df.Category.head(10000))


#docs_new = ['Britain new', 'facebook google']
#X_new_counts = count_vect.transform(docs_new)

#print X_new_counts

#predicted = clf.predict(X_new_counts)

#print predicted
#for doc, category in zip(docs_new, predicted):
 # print('%r => %s' % (doc,category))

#cross validation
kf = KFold(n_splits=10)
fold = 0
count_vect = TfidfVectorizer(stop_words=stop_words)
count_vect.fit(df['Content'])
svd = TruncatedSVD(n_components=5)
svd.fit(count_vect.transform(df['Content']))
#X_train_counts = count_vect.transform(df['Content'])
for train_index, test_index in kf.split(df):
     X_train_counts = count_vect.transform(df['Content'].iloc[train_index])
     X_train_counts = np.add(X_train_counts, count_vect.transform(df['Title'].iloc[train_index]))
     X_test_counts = count_vect.transform(df['Content'].iloc[test_index])
     X_test_counts = np.add(X_test_counts, count_vect.transform(df['Title'].iloc[test_index]))
     X_train_counts = svd.transform(X_train_counts)
     X_test_counts = svd.transform(X_test_counts)
     clf_cv = RandomForestClassifier(n_estimators=5).fit(X_train_counts,df['Category'].iloc[train_index])
     yPred = clf_cv.predict(X_test_counts)
     print yPred
     fold += 1
     print "Fold " + str(fold)
     print(classification_report(yPred,df['Category'].iloc[test_index], target_names=df.Category.unique()))
