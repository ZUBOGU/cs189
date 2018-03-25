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
	testData = sio.loadmat('./data/test.mat')
	trainData = sio.loadmat('./data/train.mat')

	trainImages = trainData['train_images']
	testImages = testData['test_images']
	testLabel = testData['test_labels']
	testLabels = np.transpose(testLabel,(1, 0))[0]

	x_dim = len(trainImages)
	y_dim = len(trainImages[0])
	image_index = len(trainImages[0][0])
	trainImages = trainImages.transpose((2, 0, 1))
	trainImages = trainImages.reshape(image_index, x_dim * y_dim)
	actualLabels = np.transpose(trainData['train_labels'],(1, 0))[0]
	trainImages, actualLabels = shuffle(trainImages, actualLabels)
	trainImages, actualLabels = shuffle(trainImages, actualLabels)

	#rotate test data to correct way
	for i in range(len(testImages)):
		testImages[i] = np.fliplr(np.rot90(np.rot90(np.rot90(testImages[i].reshape(28,28))))).reshape(28 * 28)

	testImages = preprocessing.normalize(testImages.astype("float"))
	trainImages = preprocessing.normalize(trainImages.astype("float"))

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

		alpha = 0.0008

		predict =[]
		var = [multivariate_normal(means[i], covariances[i] + alpha * np.identity(784)) for i in range(10)]
		for i in range(len(validateX)):

			pdfs = [var[j].logpdf(validateX[i]) + np.log(prior[j]) for j in range(10)]
			predict += [np.argmax(pdfs)]
		accuracy = calculateAccuracy(predict, validateY)

		# creat Kaggle submission file
		predict_labels = predict
		indexs = [i for i in range(1, 10001)]
		data = []
		data += [indexs]
		data += [predict_labels]
		data = np.transpose(data, (1, 0)).tolist()
		first_row = [['Id', 'Category']]
		with open('digitpredict.csv', 'w') as f:
		    a = csv.writer(f)
		    a.writerows(first_row)
		    a.writerows(data)

		return 1 - accuracy

	print(errorrate(trainImages[:60000], actualLabels[:60000], testImages, testLabels))

