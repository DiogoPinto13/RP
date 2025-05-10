import classifiers
import preProcess
import utils

def performanceTestDataset(
  classifierFn,
  classifierParams=None
):
  if classifierParams is None:
    classifierParams = {}

  dfDev, dfTest, dfTargetDev, dfTargetTest = preProcess.preProcessDataset(True)

  outputDir = f"outputs/classifiers/{utils.getClassifierLabel(classifierFn.__name__)}/Performance Test Dataset"
  utils.os.makedirs(outputDir, exist_ok=True)

  kf = utils.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  f1ScoresKfold = []

  for trainIndex, valIndex in kf.split(dfDev, dfTargetDev):
    dfTrain, dfVal = dfDev.iloc[trainIndex], dfDev.iloc[valIndex]
    dfTargetTrain, dfTargetVal = dfTargetDev.iloc[trainIndex], dfTargetDev.iloc[valIndex]

    yTrue, yPred = classifierFn({
      "dfTrain": dfTrain,
      "dfTest": dfVal,
      "dfTargetTrain": dfTargetTrain,
      "dfTargetTest": dfTargetVal,
      **classifierParams
    })

    f1 = utils.f1_score(yTrue, yPred)
    f1ScoresKfold.append(f1)

  f1ScoresKfoldMean = utils.pd.Series(f1ScoresKfold).mean()
  f1ScoresKfoldStd = utils.pd.Series(f1ScoresKfold).std()

  yTrueTest, yPredTest = classifierFn({
    "dfTrain": dfDev,
    "dfTest": dfTest,
    "dfTargetTrain": dfTargetDev,
    "dfTargetTest": dfTargetTest,
    **classifierParams
  })

  f1ScoreTest = utils.f1_score(yTrueTest, yPredTest)

  metricsDf = utils.pd.DataFrame({
    "f1ScoresKfoldMean": [f1ScoresKfoldMean],
    "f1ScoresKfoldStd": [f1ScoresKfoldStd],
    "f1ScoreTest": [f1ScoreTest]
  })
  metricsDf.to_csv(utils.os.path.join(outputDir, "metrics.csv"), index=False)

  utils.plt.figure()
  utils.plt.boxplot(
    f1ScoresKfold,
    patch_artist=True,
    boxprops=dict(color="blue"),
    medianprops=dict(color="red"),
    tick_labels=["F1 Score (K-Folds)"]
  )
  utils.plt.scatter(1, f1ScoreTest, color="black", zorder=3, label="Test F1 Score")
  utils.plt.title("Performance Test Dataset", fontsize=13)
  utils.plt.ylabel("F1-Score", fontsize=12)
  utils.plt.legend()
  utils.plt.savefig(utils.os.path.join(outputDir, "performanceBoxplot.png"), bbox_inches='tight')
  utils.plt.close()


# naiveBayesClassifier, KNNClassifier
performanceTestDataset(
  classifierFn=classifiers.KNNClassifier,
  classifierParams={
    "k": 1
  }
)