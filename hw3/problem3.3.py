import scipy.io as sio
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing

def calculateRisk(Q, w):
	power = -Q.dot(w)
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

	#q2
	# using Xstandardize
	w = np.zeros((57, 1))
	ylabel1 = []
	Q = np.diag(Ytrain).dot(Xstandardize)
	for i in range(10000):
		index = np.random.randint(0, 3450)	
		xi = np.reshape(Xstandardize[index], (57,1))
		yi = Ytrain[index]
		zi = yi * xi.T.dot(w)
		w = w + yi * xi / (1 + np.exp(zi)) / (i + 1)
		risk = calculateRisk(Q, w)
		ylabel1 += [risk]
		print('The ', i, 'th risk is ', risk)
	x_label1 = [i for i in range(len(ylabel1))]

	plt.title('Problem 3.3 the training risk vs. the number of iterations \n feature i) setpsize = 1/t t:iteration number')
	plt.xlabel('number of iteration')
	plt.ylabel('the risk')
	plt.plot(x_label1, ylabel1, 'bo-')
	plt.show()

	#using Xtransform
	w = np.zeros((58, 1))
	ylabel2 = []
	Q = np.diag(Ytrain).dot(Xtransform)
	for i in range(10000):
		index = np.random.randint(0, 3450)	
		xi = np.reshape(Xtransform[index], (58,1))
		yi = Ytrain[index]
		zi = yi * xi.T.dot(w)
		w = w + yi * xi / (1 + np.exp(zi)) / (i + 1)
		risk = calculateRisk(Q, w)
		ylabel2 += [risk]
		print('The ', i, 'th risk is ', risk)
	x_label2 = [i for i in range(len(ylabel2))]
	plt.title('Problem 3.3 the training risk vs. the number of iterations \n feature ii) setpsize = 1/t t:iteration number')
	plt.xlabel('number of iteration')
	plt.ylabel('the risk')
	plt.plot(x_label2, ylabel2, 'bo-')
	plt.show()


	# #using Xbinarize
	w = np.zeros((58, 1))
	ylabel3 = []
	Q = np.diag(Ytrain).dot(Xbinarize)
	for i in range(10000):
		index = np.random.randint(0, 3450)	
		xi = np.reshape(Xbinarize[index], (58,1))
		yi = Ytrain[index]
		zi = yi * xi.T.dot(w)
		w = w + yi * xi / (1 + np.exp(zi)) / (i + 1)
		risk = calculateRisk(Q, w)
		ylabel3 += [risk]
		print('The ', i, 'th risk is ', risk)
	x_label3 = [i for i in range(len(ylabel3))]
	plt.title('Problem 3.3 the training risk vs. the number of iterations \n feature iii) setpsize = 1/t t:iteration number')
	plt.xlabel('number of iteration')
	plt.ylabel('the risk')
	plt.plot(x_label3, ylabel3, 'bo-')
	plt.show()

