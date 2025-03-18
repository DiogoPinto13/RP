import utils

def fisherLDA(dfTrain, dfTest, dfTargetTrain, dfTargetTest):
    lda = utils.LinearDiscriminantAnalysis()
    lda.fit(dfTrain, dfTargetTrain)

    predictions = lda.predict(dfTest)

    print("Accuracy:", utils.accuracy_score(dfTargetTest, predictions))
    print("\nConfusion Matrix:")
    print(utils.confusion_matrix(dfTargetTest, predictions))
    print("\nClassification Report:")
    print(utils.classification_report(dfTargetTest, predictions))
