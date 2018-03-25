import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
import csv
import time

#read input
testData = sio.loadmat('./dataset/test.mat')
trainData = sio.loadmat('./dataset/train.mat')

trainImages = trainData['train_images']   # 60000 datapoint
testImages = testData['test_images']

x_dim = len(trainImages)
y_dim = len(trainImages[0])
image_index = len(trainImages[0][0])
trainImages = trainImages.transpose((2, 0, 1))
trainImages = trainImages.reshape(image_index, x_dim * y_dim)
actualLabels = np.transpose(trainData['train_labels'],(1, 0))[0]
trainImages, actualLabels = shuffle(trainImages, actualLabels)
trainImages, actualLabels = shuffle(trainImages, actualLabels)

x_dim = len(testImages)
y_dim = len(testImages[0])
image_index = len(testImages[0][0])
testImages = testImages.transpose((2, 0, 1))
testImages = testImages.reshape(image_index, x_dim * y_dim)

trainImages = np.insert(trainImages, 784, 1, axis = 1)
testImages = np.insert(testImages, 784, 1, axis = 1)

testImages = preprocessing.normalize(testImages.astype("float"), norm='l2', axis=0)
trainImages = preprocessing.normalize(trainImages.astype("float"), norm='l2', axis=0)

validationset = trainImages[50000:]
validationsetLabels = actualLabels[50000:]

def createY(label):
	y = np.zeros(10)
	y[label] = 1.0
	return y

def calculateAccuracy(expect, actual):
	same = [i for i in range(len(expect)) if expect[i] == actual[i]]
	return len(same) / len(actual)



## cost function 1: the mean squared error.
##			2:the cross-entropy error.
def trainNeuralNetwork(images, labels, learningRate, costfuction):
	W1 = np.random.randn(785, 200)
	W2 = np.random.randn(201, 10)
	t0 = time.clock()
	iterations = 0
	trainerrors = []
	validaaccuracies = []
	while True:

		# if (iterations % 1000000 == 0):

		# 	# write csv
		# 	i = iterations / 1000000
		# 	predict = predictNeuralNetwork(W1, W2, images)
		# 	print("training accuracy",calculateAccuracy(predict, labels))
		# 	print(time.clock() -t0, "seconds , write", i)
		# 	predict_labels = predictNeuralNetwork(W1, W2, testImages)
		# 	indexs = [i for i in range(1, len(testImages) + 1)]
		# 	data = []
		# 	data += [indexs]
		# 	data += [predict_labels]
		# 	data = np.transpose(data, (1, 0)).tolist()
		# 	first_row = [['Id', 'Category']]
		# 	with open('digitpredict' + str(i) + '.csv', 'w') as f:
		# 	    a = csv.writer(f)
		# 	    a.writerows(first_row)
		# 	    a.writerows(data)

		# if (iterations % 10000 == 0):
		# 	predict = predictNeuralNetwork(W1, W2, validationset)
		# 	print("validationset accuracy",calculateAccuracy(predict, validationsetLabels))

		##for plot purpose
		if (iterations % 1000 == 0):
			print(iterations)
			predict = predictNeuralNetwork(W1, W2, validationset)
			trainerrors += [1 - calculateAccuracy(predict, validationsetLabels)]
			predict = predictNeuralNetwork(W1, W2, images)
			validaaccuracies += [calculateAccuracy(predict, labels)]

		if (iterations  % 100000 == 0):
			x_label = [i * 1000 for i in range(len(trainerrors))]
			plt.title('total training error vs iterations')
			plt.xlabel('The iterations')
			plt.ylabel('The training error rate')
			plt.plot(x_label, trainerrors, 'bo')
			plt.show()

			plt.title('classification accuracy vs iterations')
			plt.xlabel('The iterations')
			plt.ylabel('The training error rate')
			plt.plot(x_label, validaaccuracies, 'bo')
			plt.show()

		# get data point
		index = np.random.randint(len(images))
		x = images[index]
		label = labels[index]
		y = createY(label)

		#forward
		z1 = x.dot(W1)
		y1 = np.tanh(z1)
		y1b = np.insert(y1, 200, 1, axis = 0)
		z2 = y1b.dot(W2)
		y2 = 1 / (1 + np.exp(-z2))

		#backword
		if costfuction == 1:
			dirz2 = np.diag(y2).dot(np.diag(1 - y2)).dot(y2 - y).T
			deltaW2 = np.outer(y1b, dirz2)

			nobaisW2 = np.delete(W2, 200, axis = 0)
			deltaW1 = np.outer(x, (np.diag(1 - np.power(y1, 2)).dot(nobaisW2).dot(dirz2).T))

			W2 = 0.7 * W2 + 0.3 * (W2 - learningRate * deltaW2)
			W1 = W1 - learningRate * deltaW1
		else:
			for i in range(len(y2)):
				if y2[i] == 1:
					y2[i] = 0.999999999999
				if y2[i] == 0:
					y2[i] = 0.000000000001
			dirz2 = np.diag(y2).dot(np.diag(1 - y2)).dot(np.divide(1 - y, 1 - y2) - np.divide(y, y2)).T
			deltaW2 = np.outer(y1b, dirz2)

			nobaisW2 = np.delete(W2, 200, axis = 0)
			deltaW1 = np.outer(x, (np.diag(1 - np.power(y1, 2)).dot(nobaisW2).dot(dirz2).T))

			W2 = 0.7 * W2 + 0.3 * (W2 - learningRate * deltaW2)
			W1 = W1 - learningRate * deltaW1
		iterations += 1
	return W1, W2


def predictNeuralNetwork(W1, W2, images):
	predict = []
	for x in images:
		z1 = W1.T.dot(x)
		y1 = np.tanh(z1)
		y1b = np.insert(y1, 200, 1, axis = 0)
		z2 = W2.T.dot(y1b)
		y2 = 1 / (1 + np.exp(-z2))
		label = np.argmax(y2)
		predict += [label]
	return predict

trainNeuralNetwork(trainImages[:50000], actualLabels[:50000], 2, 1)

