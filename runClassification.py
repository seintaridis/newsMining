import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from csvReader import writeStats
from csvReader import createTestSetCategoryCSV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter



def preprocessData(data):
    my_additional_stop_words=['said','th','month','much','thing','say','says']
    stop_words = ENGLISH_STOP_WORDS.union(my_additional_stop_words)
    vectorizer = TfidfVectorizer(stop_words=stop_words) #Stopwords to our vectorizer
    # first get TFIDF matrix
    X=vectorizer.fit_transform(data)

    # second compress to 5 dimensions
    svd = TruncatedSVD(n_components=5)
    reduced=svd.fit_transform(X)
    return reduced;

def knn(X_train_counts,X_test_counts,categories):
    yPred = []
    #print test_index
    #print X_test_counts
    for test ,i in enumerate(X_test_counts):
        # create list for distances and targets
        distances = euclidean_distances(X_train_counts, [X_test_counts[test]])
        distances = zip(distances,categories)

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


def classificationMethod(method,X_train_counts,X_test_counts,categories,train_index,test_index):
    yPred=None;
    C = 3.0
    if method == 'naiveBayes':
        clf_cv = GaussianNB().fit(X_train_counts,categories)
    elif method == 'RandomForest':
        clf_cv = RandomForestClassifier(n_estimators=5).fit(X_train_counts,categories)
    elif method == 'SVM':
        clf_cv = svm.SVC(kernel='linear', C=C,gamma=0.7).fit(X_train_counts,categories)
    elif method == 'KNN':
        return knn(X_train_counts,X_test_counts,categories)
    yPred = clf_cv.predict(X_test_counts)#after training  try to predi
    return yPred;

#find categories for the test dataset
def findCategories(df,test_df):
    my_additional_stop_words=['said','th','month','much','thing','say','says']
    stop_words = ENGLISH_STOP_WORDS.union(my_additional_stop_words)
    count_vect = TfidfVectorizer(stop_words=stop_words)
    #count_vect = CountVectorizer(stop_words=stop_words)
    count_vect.fit(df['Content'])
    svd = TruncatedSVD(n_components=40)
    svd.fit(count_vect.transform(df['Content']))
    X_train_counts = count_vect.transform(df['Content'])
    X_train_counts = np.add(X_train_counts, count_vect.transform(df['Title']))
    X_test_counts = count_vect.transform(test_df['Content'])
    X_test_counts = np.add(X_test_counts, count_vect.transform(test_df['Title']))
    X_train_counts = svd.transform(X_train_counts)
    X_test_counts = svd.transform(X_test_counts)
    yPred = classificationMethod('SVM',X_train_counts,X_test_counts,df['Category'],44,44)
    print yPred
    createTestSetCategoryCSV(test_df['Id'],yPred)

def crossValidation(df,method,n_components):
    avgAccuracy=0
    nFolds=10
    kf = KFold(n_splits=nFolds)
    fold = 0
    my_additional_stop_words=['said','th','month','much','thing','say','says']
    stop_words = ENGLISH_STOP_WORDS.union(my_additional_stop_words)
    count_vect = TfidfVectorizer(stop_words=stop_words)
    #count_vect = CountVectorizer(stop_words=stop_words)
    count_vect.fit(df['Content'])
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(count_vect.transform(df['Content']))
    for train_index, test_index in kf.split(df):
        X_train_counts = count_vect.transform(df['Content'].iloc[train_index])
        X_train_counts = np.add(X_train_counts, count_vect.transform(df['Title'].iloc[train_index]))
        X_test_counts = count_vect.transform(df['Content'].iloc[test_index])
        X_test_counts = np.add(X_test_counts, count_vect.transform(df['Title'].iloc[test_index]))
        X_train_counts = svd.transform(X_train_counts)
        X_test_counts = svd.transform(X_test_counts)
        print "Fold " + str(fold)
        if method=='ALL':
            runAllClassificationMethods(df,nFolds,X_train_counts,X_test_counts,train_index,test_index)
        else:
            yPred = classificationMethod(method,X_train_counts,X_test_counts,df['Category'].iloc[train_index],train_index,test_index)
            print(classification_report(yPred,df['Category'].iloc[test_index], target_names=df.Category.unique()))
            avgAccuracy+=accuracy_score(df['Category'].iloc[test_index],yPred)
        fold += 1
    if method=='ALL':
        produceStats(nFolds)
    avgAccuracy=avgAccuracy/nFolds
    print "the average accuracy of method "+ method
    print avgAccuracy


def runAllClassificationMethods(df,nFolds,X_train_counts,X_test_counts,train_index,test_index):
    classification_method_array=['naiveBayes','RandomForest','SVM','KNN']
    for idx,value in enumerate(classification_method_array):
        yPred = classificationMethod(value,X_train_counts,X_test_counts,df['Category'].iloc[train_index],train_index,test_index)
        averageAccurracyArray[idx] += accuracy_score(df['Category'].iloc[test_index],yPred)
        averagePrecisionArray[idx] += precision_score(df['Category'].iloc[test_index], yPred, average='macro')
        averageRecallArray[idx] += recall_score(df['Category'].iloc[test_index], yPred, average='macro')
        averageFmeasureArray[idx]+= f1_score(df['Category'].iloc[test_index], yPred, average='macro')


def produceStats(nFolds):
    for idx,val in enumerate(averageAccurracyArray):
        averageAccurracyArray[idx] =averageAccurracyArray[idx]/nFolds
        averagePrecisionArray[idx] =averagePrecisionArray[idx]/nFolds
        averageRecallArray[idx]=averageRecallArray[idx]/nFolds
        averageFmeasureArray[idx]=averageFmeasureArray[idx]/nFolds
    writeStats(averageAccurracyArray,averagePrecisionArray,averageRecallArray,averageFmeasureArray,averageAUCarray)




averageAccurracyArray=[0,0,0,0]
averagePrecisionArray=[0,0,0,0]
averageRecallArray=[0,0,0,0]
averageFmeasureArray=[0,0,0,0]
averageAUCarray=[0,0,0,0]
df = pd.read_csv('dataSets/train_set.csv', sep='\t')
crossValidation(df.head(100),'KNN',40)   #ALL TO RUN ALL METHODS OTHERWIRSE PUT ONE METHOD OF THESE classification_method_array=['naiveBayes','RandomForest','SVM','KNN']
testdf =pd.read_csv('dataSets/test_set.csv', sep='\t')
findCategories(df,testdf)
