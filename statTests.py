import utils

def logStatsMsg(logFile, msg, reset=False):
  print(msg)
  utils.Path(logFile).parent.mkdir(parents=True, exist_ok=True)
  mode = "w" if reset else "a"
  with open(logFile, mode) as f:
    f.write(msg + "\n")

def checkCombinationsNormality(classifier, testType, metric):
  basePath = utils.Path(f"outputs/classifiers/{classifier.__name__}/{testType}")
  logFile = utils.os.path.join(basePath, "statisticalTestResult.txt")

  logStatsMsg(
    logFile,
    f"1. Shapiro-Wilk normality test results for classifier '{classifier.__name__}' and test '{testType}':\n",
    True
  )

  combinationDirs = sorted(
    [d for d in basePath.iterdir() if d.is_dir() and d.name.startswith("combination")],
    key=lambda d: int(d.name.replace("combination", ""))
  )
  
  allParametric = True
  for i, combDir in enumerate(combinationDirs, start=1):
    resultsPath = combDir / "combinationResults.csv"
    df = utils.pd.read_csv(resultsPath)
    combinationValues = df[metric].tolist()

    _, p = utils.test_normal_sw(combinationValues)
    logStatsMsg(logFile, f"Combination {i}: p = {p:.4f} → {'Normal' if p > 0.05 else 'Not normal'}")

    if p <= 0.05:
      allParametric = False

  logStatsMsg(
    logFile,
    "\nResult: All combinations are parametric." if allParametric else "\nResult: At least one combination is not parametric."
  )
  return allParametric

def checkNonParametricCombinationsDifference(classifier, testType, metric):
  basePath = utils.Path(f"outputs/classifiers/{classifier.__name__}/{testType}")
  logFile = basePath / "statisticalTestResult.txt"

  logStatsMsg(
    logFile,
    f"\n2. Due to the normality test indicating non-parametric data, performing Kruskal-Wallis test for differences between parameter combinations of classifier '{classifier.__name__}' and test '{testType}':"
  )

  combinationDirs = sorted(
    [d for d in basePath.iterdir() if d.is_dir() and d.name.startswith("combination")],
    key=lambda d: int(d.name.replace("combination", ""))
  )

  allCombinationValues = []
  for combDir in combinationDirs:
    resultsPath = combDir / "combinationResults.csv"
    df = utils.pd.read_csv(resultsPath)
    combinationValues = df[metric].tolist()
    allCombinationValues.append(combinationValues)

  _, p = utils.kruskal_wallis(allCombinationValues)

  logStatsMsg(logFile, f"Result: p = {p:.4f}")
  logStatsMsg(logFile, "Conclusion: No significant difference between combinations." if p > 0.05 else "Conclusion: At least one combination differs significantly from the others.")
  return p <= 0.05


def checkPairwiseNonParametricComparisons(classifier, testType, metric):
  basePath = utils.Path(f"outputs/classifiers/{classifier.__name__}/{testType}")
  logFile = basePath / "statisticalTestResult.txt"

  logStatsMsg(
    logFile,
    f"\n3. Pairwise Mann-Whitney U tests for combinations of classifier '{classifier.__name__}' and test '{testType}':"
  )

  combinationDirs = sorted(
    [d for d in basePath.iterdir() if d.is_dir() and d.name.startswith("combination")],
    key=lambda d: int(d.name.replace("combination", ""))
  )

  allCombinationValues = []
  for combDir in combinationDirs:
    resultsPath = combDir / "combinationResults.csv"
    df = utils.pd.read_csv(resultsPath)
    combinationValues = df[metric].tolist()
    allCombinationValues.append(combinationValues)

  n = len(allCombinationValues)
  wins = [0] * n

  for (i, data1), (j, data2) in utils.combinations(enumerate(allCombinationValues), 2):
    _, p = utils.mann_whitney(data1, data2)
    mean1 = utils.np.mean(data1)
    mean2 = utils.np.mean(data2)

    if p <= 0.05:
      if mean1 > mean2:
        wins[i] += 1
        result = f"Combination {i+1} beats Combination {j+1} (p = {p:.4f})"
      else:
        wins[j] += 1
        result = f"Combination {j+1} beats Combination {i+1} (p = {p:.4f})"
    else:
      result = f"Combination {i+1} vs Combination {j+1}: Not significant (p = {p:.4f})"

    logStatsMsg(logFile, result)

  ranked = sorted(enumerate(wins), key=lambda x: x[1], reverse=True)
  logStatsMsg(logFile, "\nRanking based on number of pairwise wins:")
  for rank, (idx, winCount) in enumerate(ranked, start=1):
    logStatsMsg(logFile, f"{rank}. Combination {idx+1} — {winCount} wins")

def baseStatTest(classifier, testType, metric="fScore"):
  isParametric = checkCombinationsNormality(classifier, testType, metric)

  if not isParametric:
    isDifferent = checkNonParametricCombinationsDifference(classifier, testType, metric)
    if isDifferent:
      checkPairwiseNonParametricComparisons(classifier, testType, metric)