import utils
import preProcess
import featureSelectionReduction
import classifiers

def main():
    dfData, dfLabels = preProcess.preProcessDataset()
    featureReductionSelection, classifierSelection = utils.showMenu()

    optionsFeatureReductionSelection = {
        1: featureSelectionReduction.featureReductionPCA,
        2: featureSelectionReduction.featureReductionLDA,
        3: featureSelectionReduction.featureSelectionKsTest
    }
    
    dfData = optionsFeatureReductionSelection[featureReductionSelection](dfData, dfLabels)
    dfTrain, dfTest, dfTargetTrain, dfTargetTest = utils.train_test_split(dfData, dfLabels, test_size=0.3, random_state=42)

    optionsFeatureClassifier = {
        1: classifiers.fisherLDA,
        2: classifiers.eucludianMinimumDistanceClassifier,
        3: classifiers.mahalanobisMinimumDistanceClassifier
    }
    optionsFeatureClassifier[classifierSelection](dfTrain, dfTest, dfTargetTrain, dfTargetTest)
    

if __name__ == "__main__":
    main()