import preProcess
import utils
import classifiers

def main():
    dfData, dfLabels = preProcess.preProcessDataset()
    dfData = preProcess.featureSelectionPCA(dfData)
    dfData = preProcess.featureSelectionLDA(dfData, dfLabels)
    #dfTrain, dfTest, dfTargetTrain, dfTargetTest = utils.train_test_split(dfData, dfLabels, test_size=0.3, random_state=42)
    # classifiers.fisherLDA(dfTrain, dfTest, dfTargetTrain, dfTargetTest)
    

if __name__ == "__main__":
    main()