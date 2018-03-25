import scipy.io as sio
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing


def calculateRisk(X, Y, w):
	power = -np.diag(Y).dot(X).dot(w)
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
	Xstandardize = preprocessing.scale(Xtrain)
	# print(Xstandardize)

	#add bias
	Xtrain = np.insert(Xtrain, 57, 1, axis = 1)

	#transform the feature
	Xtransform = np.log(Xtrain + 0.1)
	# print(Xtransform)

	#Binarize the feature
	binarizer = preprocessing.Binarizer().fit(Xtrain)
	Xbinarize = binarizer.transform(Xtrain)
	# print(Xbinarize)


	#q1
	# #using Xstandardize
	w = np.zeros((57, 1))
	ylabel1 = []
	Q = np.diag(Ytrain).dot(Xstandardize)
	QT = Q.T
	for i in range(1000):
		QW = Q.dot(w)
		QW = QW.clip(-99, 99)
		derivateWRTw = - QT.dot(1 / (1 + np.exp(QW)))
		w = w - 0.001 * derivateWRTw
		risk = calculateRisk(Xstandardize, Ytrain, w)
		ylabel1 += [risk]
		print('The ', i, 'th risk is ', risk)
	x_label1 = [i for i in range(len(ylabel1))]

	plt.title('Problem 3.1 the training risk vs. the number of iterations \n feature i) setpsize = 0.001')
	plt.xlabel('number of iteration')
	plt.ylabel('the risk')
	plt.plot(x_label1, ylabel1, 'bo-')
	plt.show()

	# #using Xtransform
	w = np.zeros((58, 1))
	ylabel2 = []
	Q = np.diag(Ytrain).dot(Xtransform)
	QT = Q.T
	for i in range(1000):
		QW = Q.dot(w)
		QW = QW.clip(-99, 99)
		derivateWRTw = - QT.dot(1 / (1 + np.exp(QW)))
		w = w - 0.00001 * derivateWRTw
		risk = calculateRisk(Xtransform, Ytrain, w)
		ylabel2 += [risk]
		print('The ', i, 'th risk is ', risk)
	x_label2 = [i for i in range(len(ylabel2))]

	plt.title('Problem 3.1 the training risk vs. the number of iterations \n feature ii) setpsize = 0.00001')
	plt.xlabel('number of iteration')
	plt.ylabel('the risk')
	plt.plot(x_label2, ylabel2, 'bo-')
	plt.show()


	# #using Xbinarize
	w = np.zeros((58, 1))
	ylabel3 = []
	Q = np.diag(Ytrain).dot(Xbinarize)
	QT = Q.T
	for i in range(1000):
		QW = Q.dot(w)
		QW = QW.clip(-99, 99)
		derivateWRTw = - QT.dot(1 / (1 + np.exp(QW)))
		w = w - 0.00001 * derivateWRTw
		risk = calculateRisk(Xbinarize, Ytrain, w)
		ylabel3 += [risk]
		print('The ', i, 'th risk is ', risk)
	x_label3 = [i for i in range(len(ylabel3))]

	plt.title('Problem 3.1 the training risk vs. the number of iterations \n feature iii) setpsize = 0.00001')
	plt.xlabel('number of iteration')
	plt.ylabel('the risk')
	plt.plot(x_label3, ylabel3, 'bo-')
	plt.show()
