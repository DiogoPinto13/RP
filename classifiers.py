import utils

def fisherLDA(dfTrain, dfTest, dfTargetTrain, dfTargetTest):
	lda = utils.LinearDiscriminantAnalysis()
	lda.fit(dfTrain, dfTargetTrain)

	predictions = lda.predict(dfTest)

	print("Accuracy:", utils.accuracy_score(dfTargetTest, predictions))
	print("\nConfusion Matrix:")
	print(utils.confusion_matrix(dfTargetTest, predictions))
	print("\nClassification Report:")
	print(utils.classification_report(dfTargetTest, predictions))

def eucludianMinimumDistanceClassifier(dfTrain, dfTest, dfTargetTrain, dfTargetTest):
	# get means for each class
	means = []
	sortedClasses = utils.np.sort(dfTargetTrain.unique())
	for classLabel in sortedClasses:
		classSamples = dfTrain[dfTargetTrain == classLabel]
		mean = classSamples.mean(axis=0).values
		means.append(mean)
	means = utils.np.array(means)
	
	# set g function for each class
	gFunctions = []
	for mean in means:
		def gFunction(x, mean=mean): 
			return mean.T @ x - 0.5 * mean @ mean.T
		
		gFunctions.append(gFunction)
	
	# set d function for two first classes
	def d(x):
		# if d >= 0 then isNotReliable else isReliable
		return gFunctions[0](x) - gFunctions[1](x)

	# apply d function to dfTest and get predictions
	dfResults = dfTest.apply(d, axis=1)
	dfPredictions = dfResults.apply(lambda result: 0 if result >= 0 else 1)

	print("Accuracy:", utils.accuracy_score(dfTargetTest, dfPredictions))
	print("\nConfusion Matrix:")
	print(utils.confusion_matrix(dfTargetTest, dfPredictions))
	print("\nClassification Report:")
	print(utils.classification_report(dfTargetTest, dfPredictions))

def mahalanobisMinimumDistanceClassifier(dfTrain, dfTest, dfTargetTrain, dfTargetTest):
	# get means for each class
	means = []
	sortedClasses = utils.np.sort(dfTargetTrain.unique())
	for class_label in sortedClasses:
		classSamples = dfTrain[dfTargetTrain == class_label]
		mean = classSamples.mean(axis=0).values
		means.append(mean)
	means = utils.np.array(means)

	# get covariance matrix for each class
	covarianceMatrices = []
	for classLabel in sortedClasses:
		classSamples = dfTrain[dfTargetTrain == classLabel]
		covarianceMatrix = utils.np.cov(classSamples, rowvar=False)
		covarianceMatrices.append(covarianceMatrix)
	
	# get inversed pooled covariance matrix
	pooledCovariance = sum(covarianceMatrices) / len(sortedClasses)
	inversedPooledCovariance = utils.inv(pooledCovariance)
	
	# set g function for each class
	gFunctions = []
	for mean in means:
		def gFunction(x, mean=mean): 
			return mean.T @ inversedPooledCovariance @ x - 0.5 * mean.T @ inversedPooledCovariance @ mean
		
		gFunctions.append(gFunction)
	
	# set d function for two first classes
	def d(x):
		# if d > 0 then isNotReliable else isReliable
		return gFunctions[0](x) - gFunctions[1](x)

	# apply d function to dfTest and get predictions
	dfResults = dfTest.apply(d, axis=1)
	dfPredictions = dfResults.apply(lambda result: 0 if result >= 0 else 1)

	print("Accuracy:", utils.accuracy_score(dfTargetTest, dfPredictions))
	print("\nConfusion Matrix:")
	print(utils.confusion_matrix(dfTargetTest, dfPredictions))
	print("\nClassification Report:")
	print(utils.classification_report(dfTargetTest, dfPredictions))



	
	


