import utils
import preProcess
import featureSelectionReduction
import classifiers
import automaticTestings
import evaluation

def main():
    dfData, dfLabels = preProcess.preProcessDataset()
    (
        optionTrain,
        featureSelectionOption,
        numberFeaturesSelection,
        featureReductionOption,
        numberFeaturesReduction,
        classifierOption
    ) = utils.showMenu()

    optionsFeatureSelection = {
        1: featureSelectionReduction.featureSelectionKsTest,
        2: None
    }
    optionsFeatureReduction = {
        1: featureSelectionReduction.featureReductionPCA,
        2: featureSelectionReduction.featureReductionLDA,
        3: None
    }
    optionsFeatureClassifier = {
        1: classifiers.fisherLDA,
        2: classifiers.eucludeanMinimumDistanceClassifier,
        3: classifiers.mahalanobisMinimumDistanceClassifier
    }

    if optionTrain == 1:
        automaticTestings.generateDimensionalityCurve(
            optionsFeatureSelection,
            optionsFeatureReduction, 
            optionsFeatureClassifier, 
            dfData, 
            dfLabels
        )
    elif optionTrain == 2:
        automaticTestings.trainConfidenceInterval(
            optionsFeatureSelection,
            optionsFeatureReduction, 
            optionsFeatureClassifier, 
            dfData, 
            dfLabels
        )
    else:
        featureSelectionFunction = optionsFeatureSelection[featureSelectionOption]
        dfData, dfLabels = featureSelectionFunction(dfData, dfLabels, numberFeaturesSelection) if featureSelectionFunction is not None else dfData, dfLabels
        
        featureReductionFunction = optionsFeatureReduction[featureReductionOption]
        dfData, dfLabels = featureReductionFunction(dfData, dfLabels, numberFeaturesReduction) if featureReductionFunction is not None else dfData, dfLabels
        
        dfTrain, dfTest, dfTargetTrain, dfTargetTest = utils.train_test_split(dfData, dfLabels, test_size=0.3, random_state=42)
        classifierFunction = optionsFeatureClassifier[classifierOption]
        dfTargetTest, dfPredictions = classifierFunction(dfTrain, dfTest, dfTargetTrain, dfTargetTest)
        
        evaluation.main(
            str(numberFeaturesSelection),
            featureSelectionFunction.__name__ if featureSelectionFunction is not None else "No selection",
            str(numberFeaturesReduction),
            featureReductionFunction.__name__ if featureReductionFunction is not None else "No reduction",
            classifierFunction.__name__,
            dfTargetTest,
            dfPredictions
        )


if __name__ == "__main__":
    main()