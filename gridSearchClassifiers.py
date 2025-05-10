import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path

import classifiers
import preProcess
import utils

def plotCombinationsBoxplots(classifierFn, metric, basePath):
  combinationDirs = sorted(
    [d for d in basePath.iterdir() if d.is_dir() and d.name.startswith("combination")],
    key=lambda d: int(d.name.replace("combination", ""))
  )
  
  allFscores = []
  combinationLabels = []
  for i, combDir in enumerate(combinationDirs):
    resultsPath = combDir / "combinationResults.csv"
    if resultsPath.exists():
      df = pd.read_csv(resultsPath)
      allFscores.append(df[metric].tolist())
      combinationLabels.append(f"comb{i+1}")
  
  plt.figure(figsize=(12, 6))
  plt.boxplot(allFscores, tick_labels=combinationLabels, showfliers=True, patch_artist=True, boxprops=dict(color="blue"), medianprops=dict(color="red"))
  plt.xticks(rotation=45)
  plt.xlabel("Combination")
  plt.ylabel("F1 Score")
  plt.title("Hyperparamater Combinations Results Boxplot")
  plt.grid(axis='y')
  plt.tight_layout()
  plt.savefig(basePath / "combinationsResultsPlot.png")
  plt.close()

def generateResultsDict(dfTargetTest, dfPredictions):
  tn, fp, fn, tp = utils.confusion_matrix(dfTargetTest, dfPredictions).ravel()

  accuracy = utils.accuracy_score(dfTargetTest, dfPredictions)
  precision = utils.precision_score(dfTargetTest, dfPredictions)
  recall = utils.recall_score(dfTargetTest, dfPredictions)
  fScore = utils.f1_score(dfTargetTest, dfPredictions)

  return {
    "TP": tp,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "fScore": fScore
  }

def hyperparamGridSearch(
  classifierFn,
  paramGrid,
  dfData,
  dfLabels,
  nRepeats=30,
  metric="fScore"
):
  seeds = list(range(nRepeats))
  combinations = list(product(*paramGrid.values()))
  paramNames = list(paramGrid.keys())

  basePath = Path(f"outputs/classifiers/{classifierFn.__name__}/hyperparamGridSearch")
  basePath.mkdir(parents=True, exist_ok=True)

  combinationsResultsMean = []
  combinationsResultsStd = []
  for i, comb in enumerate(combinations):
    combDict = dict(zip(paramNames, comb))
    combName = f"combination{i+1}"
    combPath = basePath / combName
    combPath.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([combDict]).to_csv(combPath / "hyperparams.csv", index=False)

    combinationResults = []
    for seed in seeds:
      dfTrain, dfTest, dfTargetTrain, dfTargetTest = utils.train_test_split(dfData, dfLabels, test_size=0.3, random_state=seed, stratify=dfLabels)
      args = {
        **combDict,
        "dfTrain": dfTrain,
        "dfTest": dfTest,
        "dfTargetTrain": dfTargetTrain,
        "dfTargetTest": dfTargetTest
      }
      dfTargetTest, dfPredictions = classifierFn(args)
      resultDict = generateResultsDict(dfTargetTest, dfPredictions)

      combinationResults.append({
        "seed": seed,
        **resultDict
      })

    combinationResultsDf = pd.DataFrame(combinationResults)
    combinationResultsDf.to_csv(combPath / "combinationResults.csv", index=False)
    combinationResultsMeanDf = pd.DataFrame([combinationResultsDf.drop(columns=["seed"]).mean(numeric_only=True)])
    combinationResultsStdDf = pd.DataFrame([combinationResultsDf.drop(columns=["seed"]).std(numeric_only=True)])
    combinationResultsMeanDf.to_csv(combPath / "combinationResultsMean.csv", index=False)
    combinationResultsStdDf.to_csv(combPath / "combinationResultsStd.csv", index=False)

    combinationResultsMeanDf.insert(0, "combination", i + 1)
    combinationResultsStdDf.insert(0, "combination", i + 1)
    combinationsResultsMean.append(combinationResultsMeanDf)
    combinationsResultsStd.append(combinationResultsStdDf)

  pd.concat(combinationsResultsMean, ignore_index=True).to_csv(basePath / "combinationsResultsMean.csv", index=False)
  pd.concat(combinationsResultsStd, ignore_index=True).to_csv(basePath / "combinationsResultsStd.csv", index=False)

  plotCombinationsBoxplots(classifierFn, metric, basePath)

dfData, dfLabels = preProcess.preProcessDataset()

dfData = dfData.iloc[:100].copy()
dfLabels = dfLabels.iloc[:100].copy()

paramGrid = {
  "k": list(range(1, 11))
}

hyperparamGridSearch(
  classifierFn=classifiers.KNNClassifier,
  paramGrid=paramGrid,
  dfData=dfData,
  dfLabels=dfLabels,
  nRepeats=30,
  metric="fScore"
)