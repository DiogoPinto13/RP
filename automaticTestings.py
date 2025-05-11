import preProcess
import utils
import featureSelectionReduction
import evaluation
import classifiers

def trainConfidenceInterval(optionsFeatureSelection, optionsFeatureReduction, optionsFeatureClassifier, dfData, dfLabels):
  dimensionalityPerClassifier = {
    "fisherLDA": 50,
    "eucludeanMinimumDistanceClassifier": 12,
    "mahalanobisMinimumDistanceClassifier": 50,
    "svmClassifier" : 50, 
    "KNNClassifier": 50, 
    "naiveBayesClassifier": 50 
  }
  c = 0.1
  gamma = 0.1
  k = 1

  resultsDict = {
    classifier.__name__: { 
      reduction.__name__: [] for key, reduction in list(optionsFeatureReduction.items()) 
    } for key, classifier in optionsFeatureClassifier.items()
  }
  columnNames = []

  for keyClassifier, classifier in optionsFeatureClassifier.items():
    classifierDimensionality = dimensionalityPerClassifier[classifier.__name__]
    for k, selection in list(optionsFeatureSelection.items())[:1]:
      dfDataSelected = featureSelectionReduction.featureSelectionKsTest(dfData, dfLabels, classifierDimensionality)
    
      for keyReduction, reduction in list(optionsFeatureReduction.items()):
        dfDataReducted = reduction(dfDataSelected, dfLabels, None, 1)
        
        for seed in range(30):
          dfTrain, dfTest, dfTargetTrain, dfTargetTest = utils.train_test_split(dfDataReducted, dfLabels, test_size=0.3, random_state=seed, stratify=dfLabels)
          classifier_args = {
            "dfTrain": dfTrain,
            "dfTest": dfTest,
            "dfTargetTrain": dfTargetTrain,
            "dfTargetTest":dfTargetTest,
            "c": c,
            "gamma": gamma,
            "k": k
          }
          dfTargetTest, dfPredictions = classifier(classifier_args)
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
      outputPath = utils.os.path.join(outputDir, f"{utils.getReductionLabel(reduction)} {utils.getClassifierLabel(classifier)}.csv")
      dfResults.to_csv(outputPath, index=False)

      classifierFScoreValues = dfResults["fScore"]
      fScoreValues.append(classifierFScoreValues)

    classifiersLabels = [utils.getClassifierLabel(classifier) for classifier in classifiers]
    
    utils.plt.figure(figsize=(12, 6))
    utils.plt.boxplot(fScoreValues, patch_artist=True, boxprops=dict(color="blue"), medianprops=dict(color="red"))

    utils.plt.xticks(range(1, len(classifiersLabels) + 1), classifiersLabels, fontsize=11, rotation=30, ha='right')
    utils.plt.ylabel('F-Score', fontsize=12)
    utils.plt.title(f'Classifiers Comparison ({utils.getReductionLabel(reduction)})', fontsize=14)

    utils.plt.tight_layout()
    
    outputPath = utils.os.path.join(outputDir, f"{utils.getReductionLabel(reduction)}.png")
    utils.plt.savefig(outputPath, bbox_inches="tight")
    utils.plt.close()

def generateDimensionalityCurve(optionsFeatureSelection, optionsFeatureReductionSelection, optionsFeatureClassifier, dfData, dfLabels):
  dimensionalities = list(range(1, dfData.shape[1]+1))
  resultsDict = {classifier.__name__: [] for key, classifier in optionsFeatureClassifier.items()}
  columnNames = []
  
  for dimensionality in dimensionalities:
    for keySelection, selection in list(optionsFeatureSelection.items())[:1]:
      dfDataSelected = featureSelectionReduction.featureSelectionKsTest(dfData, dfLabels, dimensionality)

      for keyClassifier, classifier in optionsFeatureClassifier.items():
        if keyClassifier in [1,2,3]:
          continue

        dfTrain, dfTest, dfTargetTrain, dfTargetTest = utils.train_test_split(dfDataSelected, dfLabels, test_size=0.3, random_state=42, stratify=dfLabels)
        dfTargetTest, dfPredictions = classifier({
          "dfTrain": dfTrain,
          "dfTest": dfTest,
          "dfTargetTrain": dfTargetTrain,
          "dfTargetTest": dfTargetTest,
          "c": 0.1,
          "gamma": 0.1,
          "k": 1
        })
        
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
  for classifier, dfResults in resultsDict.items():
    classifierLabel = utils.getClassifierLabel(classifier)
    outputDir = f"outputs/classifiers/{classifierLabel}/Critical Feature Dimension"
    utils.os.makedirs(outputDir, exist_ok=True)

    csvPath = utils.os.path.join(outputDir, "Critical Feature Dimension.csv")
    dfResults = utils.pd.DataFrame(dfResults, columns=columnsName)
    dfResults.to_csv(csvPath, index=False)

    x = dfResults["numberFeaturesSelection"]
    y = dfResults["fScore"]

    utils.plt.figure(figsize=(6, 4))
    utils.plt.plot(x, y, color="red", linewidth=2)
    utils.plt.xlabel("Dimensionality")
    utils.plt.ylabel("Performance (F-Score)")
    utils.plt.title("Critical Feature Dimension (CFD)")
    utils.plt.tight_layout()
    plotPath = utils.os.path.join(outputDir,"Critical Feature Dimension.png")
    utils.plt.savefig(plotPath, bbox_inches="tight")
    utils.plt.close()

def featureCorrelationResults(dfData):
  outputDir = utils.Path("outputs/featureSelections/featureCorrelation")
  outputDir.mkdir(parents=True, exist_ok=True)

  df = utils.pd.DataFrame(dfData, columns=dfData.columns.values)
  correlationMatrix = df.corr()

  utils.plt.figure(figsize=(16, 14))
  utils.sns.heatmap(correlationMatrix, annot=False, cmap="coolwarm", xticklabels=True, yticklabels=True)
  utils.plt.title("Feature Correlation Heatmap")
  utils.plt.tight_layout()
  utils.plt.savefig(outputDir / "correlationHeatmap.png")
  utils.plt.close()

  correlatedPairs = []
  for i in range(len(correlationMatrix.columns)):
    for j in range(i + 1, len(correlationMatrix.columns)):
      corr = correlationMatrix.iloc[i, j]
      if abs(corr) > 0.9:
        featureA = correlationMatrix.columns[i]
        featureB = correlationMatrix.columns[j]
        correlatedPairs.append((featureA, featureB, corr))

  with open(outputDir / "highlyCorrelatedFeatures.txt", "w") as f:
    f.write("Highly correlated feature pairs (|correlation| > 0.9):\n\n")
    for a, b, corr in correlatedPairs:
      f.write(f"{a} ↔ {b} → correlation = {corr:.3f}\n")

def featureSelectionRocCurveResults(dfData, dfLabels):
  outputDir = utils.Path("outputs/featureSelections/featureSelectionRocCurve")
  outputDir.mkdir(parents=True, exist_ok=True)

  aucScores = featureSelectionReduction.featureSelectionRocCurve(
    dfData, 
    dfLabels,   
    None, 
    False,
    True
  )

  # features ranking
  rankingPath = outputDir / "featuresRanking.txt"
  with open(rankingPath, "w") as f:
    f.write("AUC ranking:\n\n")
    for feature, score in aucScores:
      f.write(f"{feature}-->{score:.3f}\n")
  
  # best and worst feature AUC curve
  bestFeature, _ = aucScores[0]
  worstFeature, _ = aucScores[-1]
  fprBest, tprBest, _ = utils.roc_curve(dfLabels, dfData[bestFeature])
  fprWorst, tprWorst, _ = utils.roc_curve(dfLabels, dfData[worstFeature])

  utils.plt.figure()
  utils.plt.plot(fprBest, tprBest, color="blue", lw=1.5, label=f"{bestFeature}")
  utils.plt.plot(fprWorst, tprWorst, color="red", lw=1.5, label=f"{worstFeature}")
  utils.plt.title("Best vs Worst Feature ROC Curve", fontsize=12, weight="bold")
  utils.plt.xlabel("1-SP", fontsize=10)
  utils.plt.ylabel("SS", fontsize=10)
  utils.plt.xlim([0, 1])
  utils.plt.ylim([0, 1])
  utils.plt.legend(loc="lower right")
  utils.plt.tight_layout()
  utils.plt.savefig(outputDir / "rocCurve_BestVsWorst.png")
  utils.plt.close()

def featureSelectionKsResults(dfData, dfLabels):
  outputDir = utils.Path("outputs/featureSelections/featureSelectionKs")
  outputDir.mkdir(parents=True, exist_ok=True)

  Hs = featureSelectionReduction.featureSelectionKsTest(
    dfData, 
    dfLabels,   
    None, 
    False, 
    True
  )

  # features ranking
  rankingPath = outputDir / "featuresRanking.txt"
  with open(rankingPath, "w") as f:
    f.write("Kruskal-Wallis ranking:\n\n")
    for feature, score in Hs:
      f.write(f"{feature}-->{score:.3f}\n")

