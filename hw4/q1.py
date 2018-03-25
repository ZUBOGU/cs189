import scipy.io as sio
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

if __name__ == '__main__':
	#read input
	housingData = sio.loadmat('./data/housing_data.mat')

	#parta Train the model
	Xvalidate = housingData['Xvalidate']
	Yvalidate = housingData['Yvalidate']
	Xtrain = housingData['Xtrain']
	Ytrain = housingData['Ytrain'] #19440 *

	def ridgeregressionModel(Xtrain, Ytrain, Lambda): 
		Xplus = Xtrain.T.dot(Xtrain) + Lambda * np.identity(len(Xtrain[0]))
		Xplus = inv(Xplus)
		W = Xplus.dot(Xtrain.T).dot(Ytrain)
		w0 = np.mean(Ytrain)
		return W, w0

	def ridgeregressionFit(Xvalidate, W, w0):
		return Xvalidate.dot(W) + w0

	def calculateRSS(expect, actual, W, lamb):
		return np.sum(np.square(expect - actual)) + lamb * W.T.dot(W)

	#partb.ii 10-fold cross validation training to find optimal lambda
	lambdavalue = 0.00001 # change lanbdavalue to test best lambda
	Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
	RSS = 0
	print("10-fold cross-validation training on 10000 samples")
	for i in range(10):
		newXtrain = np.concatenate((Xtrain[:i * 1944], Xtrain[(i + 1) * 1944:]), axis = 0)
		newYtrain = np.concatenate((Ytrain[:i * 1944], Ytrain[(i + 1) * 1944:]), axis = 0)
		W, w0 = ridgeregressionModel(newXtrain, newYtrain, lambdavalue)
		expect = ridgeregressionFit(Xtrain[i * 1944:(i + 1) * 1944] , W, w0)
		RSS += calculateRSS(expect, Ytrain[i * 1944:(i + 1) * 1944], W, lambdavalue)

	#partb.ii the RSS
	W, w0 = ridgeregressionModel(Xtrain, Ytrain, lambdavalue)
	expect = ridgeregressionFit(Xvalidate, W, w0)
	RSS = calculateRSS(expect, Yvalidate, W, lambdavalue)
	print("The RSS is", RSS)

	# part2b.ii plot w
	x_label = [1, 2, 3, 4, 5, 6, 7, 8]
	plt.title('Problem 1.b.iii the regression coefficients W \n \
		lambdavalue is 0.00001')
	plt.xlabel('The Indexs')
	plt.ylabel('coefficients of W')
	plt.plot(x_label, W, 'bo')
	plt.show()


