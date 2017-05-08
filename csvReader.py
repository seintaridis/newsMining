import csv


def writeStats(accuracyArray,precisionArray,recallArray,fmeasureArray,aucArray):
    with open('output/EvaluationMetric_10fold.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Statistic Measure', 'Naive Bayes','Random Forest','SVM','KNN'])
        accuracyArray=['Accuracy']+accuracyArray
        precisionArray=['Precision']+precisionArray
        recallArray=['Recall']+recallArray
        fmeasureArray=['F-Measure']+fmeasureArray
        aucArray=['AUC']+aucArray
        writer.writerow(accuracyArray)
        writer.writerow(precisionArray)
        writer.writerow(recallArray)
        writer.writerow(fmeasureArray)
        writer.writerow(aucArray)


def createTestSetCategoryCSV(id,predCategories):
    with open('output/testSet_categories.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Test_Document_ID','Predicted_Category'])
        for idx,value in enumerate(id):
            writer.writerow([id[idx],predCategories[idx]])
