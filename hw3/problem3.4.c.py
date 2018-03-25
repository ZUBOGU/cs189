import scipy.io as sio
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
import random

def calculateRisk(Y, alpha, sum1):
	fx = sum1.dot(alpha)
	power = - np.diag(Y).dot(fx)
	power = power.clip(-99, 99)
	return np.sum(np.log(1 + np.exp(power)))

if __name__ == '__main__':
	#read input
	spamData = sio.loadmat('./data/spam.mat')

	Xtest = spamData['Xtest']
	Xtrain = spamData['Xtrain']
	Ytrain = spamData['Ytrain']
	Ytrain = Ytrain.T[0]

	#standardize each column, so they each have mean 0 and unit variance
	#add bias
	Xtrain = np.insert(Xtrain, 57, 1, axis = 1)

	#transform the feature
	Xtransform = np.log(Xtrain + 0.1)
	# print(Xtransform)

	#partition samples
	samples_indexs = [i for i in range(3450)]
	random.shuffle(samples_indexs)
	validation_sets_index = samples_indexs[:1150]
	validation_sets = []
	validation_sets_labels = []
	samples_sets = []
	samples_sets_labels = []
	for index in validation_sets_index:
		validation_sets += [Xtransform[index]]
		validation_sets_labels += [Ytrain[index]]
	samples_indexs = samples_indexs[1150:3450]
	for index in samples_indexs:
		samples_sets += [Xtransform[index]]
		samples_sets_labels += [Ytrain[index]]

	validation_sets = np.array(validation_sets)
	validation_sets_labels = np.array(validation_sets_labels)
	samples_sets = np.array(samples_sets)
	samples_sets_labels = np.array(samples_sets_labels)		

	#part c
	alpha = np.zeros((2300, 1))
	gama = 0.00000001
	ylabel1 = []
	ylabel2 = []
	sum1 = samples_sets.dot(samples_sets.T) + 1 # for trianng set
	sum2 = validation_sets.dot(samples_sets.T) + 1 # for validation set
	for i in range(10000):
		if i % 100 == 0:
			validationRisk = calculateRisk(validation_sets_labels, alpha, sum2)
			ylabel2 += [validationRisk]
			print('The ', i, 'th risk is ', validationRisk)
		index = np.random.randint(0, 2300)	
		xi = np.reshape(samples_sets[index], (58,1))
		yi = samples_sets_labels[index]
		zi = yi * alpha.T.dot(samples_sets.dot(xi) + 1)
		zi = zi.clip(-99, 99)
		alpha[index] = alpha[index] + 0.00001 * yi / (1 + np.exp(zi))
		alpha = (1 - gama) * alpha
		risk = calculateRisk(samples_sets_labels, alpha, sum1)
		ylabel1 += [risk]
		# print('The ', i, 'th risk is ', risk)
	x_label1 = [i for i in range(len(ylabel1))]
	x_label2 = [i * 100 for i in range(len(ylabel2))]


	pred = []
	for i in range(1150):
		xi = np.reshape(validation_sets[i], (58,1))
		fx = alpha.T.dot(np.square(samples_sets.dot(xi) + 1))
		probbe1 = 1 / (1 + np.exp(-fx))
		if probbe1 > 0.5:
			pred += [1]
		else:
			pred += [-1]

	same = [i for i in range(1150) if pred[i] == validation_sets_labels[i]]

	print('the accuracy is', len(same)/1150)

	plt.title('Problem 3.4.c the training risk vs. the number of iterations \n linear kernel feature ii) gama = 0.00000001 setpsize = 0.00001')
	plt.xlabel('number of iteration')
	plt.ylabel('the risk')
	plt.plot(x_label1, ylabel1, 'bo-')
	plt.show()

	plt.title('Problem 3.4.c the validation risk risk vs. the number of iterations \n linear kernel feature ii) gama = 0.0000001 setpsize = 0.00001')
	plt.xlabel('number of iteration')
	plt.ylabel('the risk')
	plt.plot(x_label2, ylabel2, 'bo-')
	plt.show()