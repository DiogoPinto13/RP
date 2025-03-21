import utils
import featureSelectionReduction
import evaluation

def trainConfidenceInterval(optionsFeatureSelection, optionsFeatureReductionSelection, optionsFeatureClassifier, dfData, dfLabels):
  dimensionalities = [40, 30, 20, 10]
  
  for i in range(30):
    for dimensionality in dimensionalities:
      for keySelection, selection in optionsFeatureSelection.items():
        if selection != None:
          dfData = featureSelectionReduction.featureSelectionKsTest(dfData, dfLabels, dimensionality)
        
        for keyReduction, reduction in optionsFeatureReductionSelection.items():
          if reduction != None:
            dfData = reduction(dfData, dfLabels)
          for keyClassifier, classifier in optionsFeatureClassifier.items():
            dfTrain, dfTest, dfTargetTrain, dfTargetTest = utils.train_test_split(dfData, dfLabels, test_size=0.3, random_state=42, stratify=dfLabels)
            dfPredictions, dfTargetTest = classifier(dfTrain, dfTest, dfTargetTrain, dfTargetTest)
            evaluation.main(
              dimensionality,
							selection.__name__,
							dfData.shape[1],
							reduction.__name__,
							classifier.__name__,
							dfTargetTest,
							dfPredictions
            )

def generateDimensionalityCurve(optionsFeatureSelection, optionsFeatureReductionSelection, optionsFeatureClassifier, dfData, dfLabels):
  dimensionalities = [40]
  resultsDict = {classifier.__name__: utils.pd.DataFrame() for classifier in optionsFeatureClassifier}
  
  for dimensionality in dimensionalities:
    for keySelection, selection in list(optionsFeatureSelection.items())[:-1]:
      if selection != None:
        dfData = featureSelectionReduction.featureSelectionKsTest(dfData, dfLabels, dimensionality)

      for keyClassifier, classifier in optionsFeatureClassifier.items():
        dfTrain, dfTest, dfTargetTrain, dfTargetTest = utils.train_test_split(dfData, dfLabels, test_size=0.3, random_state=42, stratify=dfLabels)
        dfPredictions, dfTargetTest = classifier(dfTrain, dfTest, dfTargetTrain, dfTargetTest)
        
        dfResult = evaluation.main(
					dimensionality,
					selection.__name__,
					"None",
					"No reduction",
					classifier.__name__,
					dfTargetTest,
					dfPredictions
        )
        if len(resultsDict[keyClassifier]) == 0:
          resultsDict[keyClassifier].concat(dfResult)
        else:
          resultsDict[keyClassifier].concat(dfResult.iloc[1:])
  plotCurveDimensionalities(resultsDict)

def plotCurveDimensionalities(resultsDict):
	outputDir = "outputs/dimensionalityCurves"
	utils.os.makedirs(outputDir, exist_ok=True) 
	print(resultsDict)
	for classifier, dfResults in resultsDict.items():
    
		x = dfResults["numberFeaturesForSelection"]
		y = dfResults["f-score"]

		utils.plt.figure(figsize=(6, 4))
		utils.plt.plot(x, y, color='red', linewidth=2) 
		utils.plt.xlabel("Dimensionality")
		utils.plt.ylabel("Performance (F-Score)")
		utils.plt.title(f"Performance vs Dimensionality - {classifier}")

		output_path = utils.os.path.join(outputDir, f"{classifier}.png")
		utils.plt.savefig(output_path, bbox_inches="tight")
		utils.plt.close()
