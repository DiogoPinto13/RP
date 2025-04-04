import utils
import featureSelectionReduction
import evaluation

def trainConfidenceInterval(optionsFeatureSelection, optionsFeatureReduction, optionsFeatureClassifier, dfData, dfLabels):
  dimensionalityPerClassifier = {
    "fisherLDA": 50,
    "eucludeanMinimumDistanceClassifier": 12,
    "mahalanobisMinimumDistanceClassifier": 50
  }
  resultsDict = {
    classifier.__name__: { 
      reduction.__name__: [] for key, reduction in list(optionsFeatureReduction.items()) 
    } for key, classifier in optionsFeatureClassifier.items()
  }
  columnNames = []

  for keyClassifier, classifier in optionsFeatureClassifier.items():
    classifierDimensionality = dimensionalityPerClassifier[classifier.__name__]
    for k, selection in list(optionsFeatureSelection.items()):
      dfDataSelected = featureSelectionReduction.featureSelectionKsTest(dfData, dfLabels, classifierDimensionality)
    
      for keyReduction, reduction in list(optionsFeatureReduction.items()):
        dfDataReducted = reduction(dfDataSelected, dfLabels, None, 1)
        
        for seed in range(30):
          dfTrain, dfTest, dfTargetTrain, dfTargetTest = utils.train_test_split(dfDataReducted, dfLabels, test_size=0.3, random_state=seed, stratify=dfLabels)
          dfPredictions, dfTargetTest = classifier(dfTrain, dfTest, dfTargetTrain, dfTargetTest)
          dfResult = evaluation.main(
            classifierDimensionality,
            selection.__name__,
            dfDataReducted.shape[1],
            reduction.__name__,
            classifier.__name__,
            dfTargetTest,
            dfPredictions,
            False
          )
          resultsDict[classifier.__name__][reduction.__name__] += dfResult.values.tolist()
          columnNames = dfResult.columns.values
    
  plot_boxplots(resultsDict, columnNames)

def plot_boxplots(resultsDict, columnNames):
  outputDir = "outputs/comparisonClassifiers"
  utils.os.makedirs(outputDir, exist_ok=True) 
  
  classifiers = list(resultsDict.keys())
  reductions = list(resultsDict[classifiers[0]].keys())
  for reduction in reductions:
    fScoreValues = []
    for classifier in resultsDict.keys():
      dfResults = utils.pd.DataFrame(resultsDict[classifier][reduction], columns=columnNames)
      output_path = utils.os.path.join(outputDir, f"{utils.getReductionLabel(reduction)}_{utils.getClassifierLabel(classifier)}.csv")
      dfResults.to_csv(output_path, index=False)

      classifierFScoreValues = dfResults["fScore"]
      fScoreValues.append(classifierFScoreValues)

    classifiersLabels = [utils.getClassifierLabel(classifier) for classifier in classifiers]
    utils.plt.boxplot(fScoreValues, patch_artist=True, boxprops=dict(color="blue"), medianprops=dict(color="red"))
    utils.plt.xticks(range(1, len(classifiersLabels) + 1), classifiersLabels, fontsize=12)
    utils.plt.ylabel('F-Score', fontsize=12)
    utils.plt.title(f'Classifiers Comparison ({utils.getReductionLabel(reduction)})', fontsize=14)

    output_path = utils.os.path.join(outputDir, f"{utils.getReductionLabel(reduction)}.png")
    utils.plt.savefig(output_path, bbox_inches="tight")
    utils.plt.close()

def generateDimensionalityCurve(optionsFeatureSelection, optionsFeatureReductionSelection, optionsFeatureClassifier, dfData, dfLabels):
  dimensionalities = list(range(1, dfData.shape[1]+1))
  resultsDict = {classifier.__name__: [] for key, classifier in optionsFeatureClassifier.items()}
  columnNames = []
  
  for dimensionality in dimensionalities:
    for keySelection, selection in list(optionsFeatureSelection.items()):
      dfDataSelected = featureSelectionReduction.featureSelectionKsTest(dfData, dfLabels, dimensionality)

      for keyClassifier, classifier in optionsFeatureClassifier.items():
        dfTrain, dfTest, dfTargetTrain, dfTargetTest = utils.train_test_split(dfDataSelected, dfLabels, test_size=0.3, random_state=42, stratify=dfLabels)
        dfPredictions, dfTargetTest = classifier(dfTrain, dfTest, dfTargetTrain, dfTargetTest)
        
        dfResult = evaluation.main(
					dimensionality,
					selection.__name__,
					"None",
					"No reduction",
					classifier.__name__,
					dfTargetTest,
					dfPredictions,
          False
        )
        resultsDict[classifier.__name__] += dfResult.values.tolist()
        columnNames = dfResult.columns.values
        
  plotCurveDimensionalities(resultsDict, columnNames)

def plotCurveDimensionalities(resultsDict, columnsName):
	outputDir = "outputs/cfd"
	utils.os.makedirs(outputDir, exist_ok=True) 
	for classifier, dfResults in resultsDict.items():
		dfResults = utils.pd.DataFrame(dfResults, columns=columnsName)
		output_path = utils.os.path.join(outputDir, f"{utils.getClassifierLabel(classifier)}.csv")
		dfResults.to_csv(output_path, index=False)

		x = dfResults["numberFeaturesSelection"]
		y = dfResults["fScore"]

		utils.plt.figure(figsize=(6, 4))
		utils.plt.plot(x, y, color="red", linewidth=2) 
		utils.plt.xlabel("Dimensionality")
		utils.plt.ylabel("Performance (F-Score)")
		utils.plt.title(f"Critical Feature Dimension (CFD)")

		output_path = utils.os.path.join(outputDir, f"{utils.getClassifierLabel(classifier)}.png")
		utils.plt.savefig(output_path, bbox_inches="tight")
		utils.plt.close()
