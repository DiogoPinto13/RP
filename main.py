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
        removeCorrelated,
        classifierOption,
        criterionPCAOption
    ) = utils.showMenu()

    optionsFeatureSelection = {
        1: featureSelectionReduction.featureSelectionKsTest,
        2: featureSelectionReduction.featureSelectionRocCurve,
        3: None
    }
    optionsFeatureReduction = {
        1: featureSelectionReduction.featureReductionPCA,
        2: featureSelectionReduction.featureReductionLDA,
        3: None
    }
    optionsFeatureClassifier = {
        1: classifiers.fisherLDA,
        2: classifiers.eucludeanMinimumDistanceClassifier,
        3: classifiers.mahalanobisMinimumDistanceClassifier,
        4: classifiers.svmClassifier,
        5: classifiers.KNNClassifier,
        6: classifiers.naiveBayesClassifier
    }

    if optionTrain == 1:
        automaticTestings.generateDimensionalityCurve(
            dict(list(optionsFeatureSelection.items())[:-1]), 
            dict(list(optionsFeatureReduction.items())[:-1]), 
            optionsFeatureClassifier, 
            dfData, 
            dfLabels
        )
    elif optionTrain == 2:
        automaticTestings.trainConfidenceInterval(
            dict(list(optionsFeatureSelection.items())[:-1]),
            dict(list(optionsFeatureReduction.items())[:-1]), 
            optionsFeatureClassifier, 
            dfData, 
            dfLabels
        )
    elif optionTrain == 4:
        #automaticTestings.parametersCombinationSVM(dfData, dfLabels)
        automaticTestings.parametersCombinationKNN(dfData, dfLabels)
    elif optionTrain == 5:
        automaticTestings.featureSelectionRocCurveResults(dfData, dfLabels)
    elif optionTrain == 6:
        automaticTestings.featureSelectionKsResults(dfData, dfLabels)
    elif optionTrain == 7:
        automaticTestings.featureCorrelationResults(dfData)
    else:
        featureSelectionFunction = optionsFeatureSelection[featureSelectionOption]
        dfData, dfLabels = featureSelectionFunction(dfData, dfLabels, removeCorrelated, numberFeaturesSelection) if featureSelectionFunction is not None else dfData, dfLabels
        
        featureReductionFunction = optionsFeatureReduction[featureReductionOption]
        dfData, dfLabels = featureReductionFunction(dfData, dfLabels, numberFeaturesReduction, criterionPCAOption) if featureReductionFunction is not None else dfData, dfLabels
        
        dfTrain, dfTest, dfTargetTrain, dfTargetTest = utils.train_test_split(dfData, dfLabels, test_size=0.3, random_state=42)
        classifierFunction = optionsFeatureClassifier[classifierOption]
        dfTargetTest, dfPredictions = classifierFunction({
            "dfTrain": dfTrain,
            "dfTest": dfTest,
            "dfTargetTrain": dfTargetTrain,
            "dfTargetTest": dfTargetTest
        })
        
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