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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from csvReader import writeStats
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from sklearn.metrics import recall_score

classification_method_array=['naiveBayes','RandomForest','SVM','KNN']

def knn(test_index,X_train_counts,X_test_counts,train_index):
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
    return yPred




def classificationMethod(method,X_train_counts,X_test_counts,category_index,train_index,test_index):
    yPred=None;
    if method == 'naiveBayes':
        clf_cv = GaussianNB().fit(X_train_counts,category_index)
    elif method == 'RandomForest':
        clf_cv = RandomForestClassifier(n_estimators=5).fit(X_train_counts,df['Category'].iloc[train_index])
    elif method == 'SVM':
        clf_cv = svm.SVC().fit(X_train_counts,category_index)
    elif method == 'KNN':
        return knn(test_index,X_train_counts,X_test_counts,train_index)
    yPred = clf_cv.predict(X_test_counts)
    return yPred;


style.use("ggplot")
df = pd.read_csv('dataSets/train_set.csv', sep='\t')
df =df.head(1000)

my_additional_stop_words=['Antonia','Nikos','Nikolas']  #check improvement with stop_words
stop_words = ENGLISH_STOP_WORDS.union(my_additional_stop_words)


#cross validation
averageAccurracy=0
averageAccurracyArray=[0,0,0,0]
averagePrecisionArray=[0,0,0,0]
averageRecallArray=[0,0,0,0]
nFolds=10
kf = KFold(n_splits=nFolds)
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
    # clf_cv = RandomForestClassifier(n_estimators=5).fit(X_train_counts,df['Category'].iloc[train_index])
    # clf_cv = GaussianNB().fit(X_train_counts,df['Category'].iloc[train_index])
     print "Fold " + str(fold)
     for idx,val in enumerate(classification_method_array):
         print val
         yPred = classificationMethod(val,X_train_counts,X_test_counts,df['Category'].iloc[train_index],train_index,test_index)
         averageAccurracyArray[idx] += accuracy_score(df['Category'].iloc[test_index],yPred)
         averagePrecisionArray[idx] += precision_score(df['Category'].iloc[test_index], yPred, average='macro')
         averageRecallArray[idx] += recall_score(df['Category'].iloc[test_index], yPred, average='macro')
         print "accuracy"
         print accuracy_score(df['Category'].iloc[test_index],yPred)
         print "precision"
         print precision_score(df['Category'].iloc[test_index], yPred, average='micro')
         print "recall"
         print recall_score(df['Category'].iloc[test_index], yPred, average='macro')
     #yPred = clf_cv.predict(X_test_counts)
    # print yPred
     fold += 1
    # print "Fold " + str(fold)
    # print(classification_report(yPred,df['Category'].iloc[test_index], target_names=df.Category.unique()))
     #print "accuracy"
     #print accuracy_score(df['Category'].iloc[test_index],yPred)
     #averageAccurracy+=accuracy_score(df['Category'].iloc[test_index],yPred)
     #print "precision"
     #print precision_score(df['Category'].iloc[test_index], yPred, average='micro')
averageAccurracy=averageAccurracy/nFolds
for idx,val in enumerate(averageAccurracyArray):
    averageAccurracyArray[idx] =averageAccurracyArray[idx]/nFolds
    averagePrecisionArray[idx] =averagePrecisionArray[idx]/nFolds
    averageRecallArray[idx]=averageRecallArray[idx]/nFolds
writeStats(averageAccurracyArray,averagePrecisionArray,averageRecallArray)
