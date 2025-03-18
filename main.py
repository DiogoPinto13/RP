import preProcess
import utils
import classifiers

def main():
    dfData, dfLabels = preProcess.preProcessDataset()
    #dfData = preProcess.featureSelectionPCA(dfData)
    dfData = preProcess.featureSelectionLDA(dfData, dfLabels)
if __name__ == "__main__":
    main()