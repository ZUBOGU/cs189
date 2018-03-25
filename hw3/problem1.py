import scipy.io as sio
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

if __name__ == '__main__':
	#read input
	housingData = sio.loadmat('./data/housing_data.mat')
	# print(housingData)

	#part1 Train the model, get W
	Xvalidate = housingData['Xvalidate']
	Yvalidate = housingData['Yvalidate']
	Xtrain = housingData['Xtrain']
	Ytrain = housingData['Ytrain']
	# print(len(Xvalidate)) #1200
	# print(len(Xvalidate[0])) #8
	# print(len(Yvalidate)) #1200
	# print(len(Xtrain)) #19440
	# print(len(Xtrain[0])) #8

	X = np.insert(Xtrain, 8, 1, axis = 1)
	Xplus = X.T.dot(X)
	Xplus = inv(Xplus)
	W = Xplus.dot(X.T).dot(Ytrain)

	#part2 get the Residual sum of Squares
	Xvalidate1 = np.insert(Xvalidate, 8, 1, axis = 1)
	expect = Xvalidate1.dot(W)
	diff = 0

	sub = expect - Yvalidate
	RSS = 0
	for i in range(len(Yvalidate)):
		RSS += np.square(sub[i])
	print('The RSS is ', RSS)

	print('The predicted value is ', np.min(expect), 'to', np.max(expect))

	print('The range is ', np.max(expect) - np.min(expect))

	print('The exact value is ', np.min(Yvalidate), 'to', np.max(Yvalidate))

	print('The range is ', np.max(Yvalidate) - np.min(Yvalidate))

	#part3 plot W
	W = W[:-1]
	x_label = [0, 1, 2, 3, 4, 5, 6, 7]
	plt.title('Problem 1.3 the regression coefficients W')
	plt.xlabel('The Indexs')
	plt.ylabel('coefficients of W')
	plt.plot(x_label, W, 'bo')
	plt.show()

	#part4 plosy residuals (f(x) - y)
	plt.title('Problem 1.4 Histogram of the residuals')
	plt.ylabel('residuals')
	plt.hist(sub, bins = len(sub)/10)
	plt.show()