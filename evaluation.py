import utils

def main(
  numberFeaturesSelection,
  featureSelection,
  numberFeaturesReduction,
  featureReduction,
  classifier,
  dfTargetTest,
  dfPredictions,
  generateCSV=True
):
  # get confusion matrix
  tn, fp, fn, tp = utils.confusion_matrix(dfTargetTest, dfPredictions).ravel()

  # get metrics
  accuracy = utils.accuracy_score(dfTargetTest, dfPredictions)
  precision = utils.precision_score(dfTargetTest, dfPredictions)
  recall = utils.recall_score(dfTargetTest, dfPredictions)
  fScore = utils.f1_score(dfTargetTest, dfPredictions)

  # format result df
  dfResult = utils.pd.DataFrame([{
    "featureSelection": featureSelection,
    "numberFeaturesSelection": numberFeaturesSelection,
    "featureReduction": featureReduction,
    "numberFeaturesReduction": numberFeaturesReduction,
    "classifier": classifier,
    "TP": tp,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "fScore": fScore
  }])

  if generateCSV:
    # save result df as csv
    timestamp = utils.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"outputs/output_{timestamp}.csv"
    dfResult.to_csv(path, index=False)

  return dfResult


def parametersCombinationTest(args):
  classifier = args["classifier"]
  dfTargetTest = args["dfTargetTest"]
  dfPredictions = args["dfPredictions"]
  generateCSV = bool(args["generateCSV"])
  # get confusion matrix
  tn, fp, fn, tp = utils.confusion_matrix(dfTargetTest, dfPredictions).ravel()

  # get metrics
  accuracy = utils.accuracy_score(dfTargetTest, dfPredictions)
  precision = utils.precision_score(dfTargetTest, dfPredictions)
  recall = utils.recall_score(dfTargetTest, dfPredictions)
  fScore = utils.f1_score(dfTargetTest, dfPredictions)

  # format result df
  resultDict = {
    "classifier": classifier,
    "TP": tp,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "fScore": fScore
  }
  for optionalParam in ["c", "gamma", "k"]:
    if optionalParam in args:
      resultDict[optionalParam] = args[optionalParam]

  dfResult = utils.pd.DataFrame([resultDict])
  if generateCSV:
    # save result df as csv
    timestamp = utils.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"outputs/parameters_combination_test/{classifier}/output_1.csv"
    dfResult.to_csv(path, mode='a', index=False, header=not utils.os.path.exists(path))

  return dfResult
