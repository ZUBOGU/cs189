import scipy.io as sio
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
from scipy.stats import multivariate_normal
from sklearn.utils import shuffle
import csv

if __name__ == '__main__':
	#read input
	trainData = sio.loadmat('./data/train.mat')
	trainImages = trainData['train_images']

	x_dim = len(trainImages)
	y_dim = len(trainImages[0])
	image_index = len(trainImages[0][0])
	trainImages = trainImages.transpose((2, 0, 1))
	trainImages = trainImages.reshape(image_index, x_dim * y_dim)
	actualLabels = np.transpose(trainData['train_labels'],(1, 0))[0]
	trainImages, actualLabels = shuffle(trainImages, actualLabels)
	trainImages, actualLabels = shuffle(trainImages, actualLabels)

	trainImages = preprocessing.normalize(trainImages.astype("float"))

	# part a
	data = {}
	for i in range(10):
		data[i] = []
	for i in range(len(trainImages)):
		key = actualLabels[i]
		value = trainImages[i]
		data[key].append(value)
	means = {}
	for i in range(10):
		means[i] = np.sum(data[i], axis=0) / len(data[i])

	print(means)

	covariances = {}
	for i in range(10):
		arrays = np.array(data[i])
		mu = np.array(means[i])
		matrix = (arrays - mu).T.dot((arrays - mu))
		matrix = matrix / len(arrays)
		covariances[i] = matrix
	print(covariances)

	# part b
	prior = []
	for i in range(10):
		prior += [len(data[i]) / len(trainImages)]
		print("The prior probability for ", i, "is", prior[i])

	# part c
	plt.matshow(covariances[2])
	plt.title('Problem 2. Confusion matrix for covariance matrix')
	plt.colorbar()
	plt.show()

	#part d.i

	def calculateAccuracy(expect, actual):
		same = [i for i in range(len(expect)) if expect[i] == actual[i]]
		return len(same) / len(actual)

	def errorrate(trainImages, actualLabels, validateX, validateY):
		data = {}
		for i in range(10):
			data[i] = []
		for i in range(len(trainImages)):
			key = actualLabels[i]
			value = trainImages[i]
			data[key].append(value)

		means = {}
		for i in range(10):
			means[i] = np.sum(data[i], axis=0) / len(data[i])

		covariances = {}
		for i in range(10):
			arrays = np.array(data[i])
			mu = np.array(means[i])
			matrix = (arrays - mu).T.dot((arrays - mu))
			matrix = matrix / len(arrays)
			covariances[i] = matrix

		prior = []
		for i in range(10):
			prior += [len(data[i]) / len(trainImages)]

		avgCovariance = 0
		alpha = 0.0008

		for i in range(10):
			avgCovariance += covariances[i] + alpha * np.identity(784)
		avgCovariance = avgCovariance / 10

		predict = []
		predict1 =[]
		var = [multivariate_normal(means[i], avgCovariance) for i in range(10)]
		var1 = [multivariate_normal(means[i], covariances[i] +\
				 alpha * np.identity(784)) for i in range(10)]
		for i in range(len(validateX)):
			if i % 1000 == 0:
				print(i)
			pdfs = [var[j].logpdf(validateX[i]) + np.log(prior[j]) for j in range(10)]
			predict += [np.argmax(pdfs)]

			pdfs1 = [var1[j].logpdf(validateX[i]) + np.log(prior[j]) for j in range(10)]
			predict1 += [np.argmax(pdfs1)]

		accuracy = calculateAccuracy(predict, validateY)
		accuracy1 = calculateAccuracy(predict1, validateY)
		return 1 - accuracy, 1 - accuracy1

	samplesSize = [100, 200, 500, 10000, 2000, 50000, 10000, 30000, 50000]
	errorrates1 = []
	errorrates2 = []
	for size in samplesSize:
		print("samplesSize is", size)
		error1, error2 = errorrate(trainImages[:size], actualLabels[:size], \
			trainImages[50000:], actualLabels[50000:])
		errorrates1 += [error1]
		errorrates2 += [error2]

	plt.title('Problem 7.d.i The error rates on validation set \
		versus training example sizes\n average covariance matrix')
	plt.xlabel('Traning sample size')
	plt.ylabel('Error rate')
	plt.plot(samplesSize, errorrates1, 'bo')
	plt.show()

	plt.title('Problem 7.d.ii The error rates on validation set versus\
	 training example sizes\n corresponding covariances matrix')
	plt.xlabel('Traning sample size')
	plt.ylabel('Error rate')
	plt.plot(samplesSize, errorrates2, 'bo')
	plt.show()







