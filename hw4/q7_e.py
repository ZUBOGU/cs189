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
	#input data
	trainData = sio.loadmat('./data/spam_data.mat')
	trainImages = trainData['training_data']
	actualLabels = trainData['training_labels'][0]
	testImages = trainData['test_data']

	trainImages, actualLabels = shuffle(trainImages, actualLabels)
	trainImages, actualLabels = shuffle(trainImages, actualLabels)

	trainImages = preprocessing.normalize(trainImages.astype("float"))

	def predict(trainImages, actualLabels, testImages):
		data = {}
		for i in range(2):
			data[i] = []
		for i in range(len(trainImages)):
			key = actualLabels[i]
			value = trainImages[i]
			data[key].append(value)

		means = {}
		for i in range(2):
			means[i] = np.sum(data[i], axis=0) / len(data[i])

		covariances = {}
		for i in range(2):
			arrays = np.array(data[i])
			mu = np.array(means[i])
			matrix = (arrays - mu).T.dot((arrays - mu))
			matrix = matrix / len(arrays)
			covariances[i] = matrix

		prior = []
		for i in range(2):
			prior += [len(data[i]) / len(trainImages)]

		alpha = 0.0008

		predict =[]
		var = [multivariate_normal(means[i], covariances[i] + alpha * np.identity(32)) for i in range(2)]
		for i in range(len(testImages)):
			pdfs = [var[j].logpdf(testImages[i]) + np.log(prior[j]) for j in range(2)]
			predict += [np.argmax(pdfs)]

		# creat Kaggle submission file
		predict_labels = predict
		indexs = [i for i in range(1, len(testImages) + 1)]
		data = []
		data += [indexs]
		data += [predict_labels]
		data = np.transpose(data, (1, 0)).tolist()
		first_row = [['Id', 'Category']]
		with open('spampredict.csv', 'w') as f:
		    a = csv.writer(f)
		    a.writerows(first_row)
		    a.writerows(data)

	predict(trainImages, actualLabels, testImages)
