import scipy.io as sio
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import operator
import csv

JokeData = sio.loadmat('./joke_data/joke_train.mat')
Images = JokeData['train']

data = [[int(i) for i in (line.strip().split(','))] for line in open("./joke_data/validation.txt", 'r')]
data = np.array(data)
ValidationSet = data[:, :2]
ValidationLabels = data[:, 2]

query = [[int(i) for i in (line.strip().split(','))] for line in open("./joke_data/query.txt", 'r')]
query = np.array(query)
query = query[:, 1:3]

#2.3.3
def calculateAccuracy(expect, actual):
	same = [i for i in range(len(actual)) if expect[i] == actual[i]]
	return len(same) / len(actual)

def predictWithUV(Dataset, vectorU, vectorV):
	predict = []
	for data in Dataset:
		estimate = vectorU[data[0] - 1].dot(vectorV[:, data[1] - 1])
		predict += [1] if estimate > 0 else [0]
	return predict

def calculateLoss(Images, U, V, lamb):
	loss = 0
	for i in range(len(Images)):
		for j in range(len(Images[0])):
			if not np.isnan(Images[i][j]):
				loss += np.square(np.linalg.norm(U[i].T.dot(V[ : , j]) - Images[i][j]))\
						+ lamb * np.square(np.linalg.norm(U[i]))\
						+ lamb * np.square(np.linalg.norm(V[ : , j]))
	return loss

def findUV(Images, d, lamb):
	U = np.random.randn(len(Images), d)
	V = np.random.randn(d, len(Images[0]))
	# Changed = True
	iterantions = 0
	while iterantions < 100:
		# Changed = False
		iterantions += 1
		newU = []
		# orignLoss = calculateLoss(Images, U, V, lamb)
		for i in range(len(Images)):
			left = lamb * np.identity(d)
			right = np.zeros(d)
			for j in range(len(Images[0])):
				if not np.isnan(Images[i][j]):
					left += np.outer(V[ : , j], V[ : , j])
					right += int(Images[i][j]) * V[ : , j].T
			newU.append(right.dot(np.linalg.inv(left)))
		newU = np.array(newU)
		U = newU

		newV = []
		for j in range(len(Images[0])):
			left = lamb * np.identity(d)
			right = np.zeros(d)
			for i in range(len(Images)):
				if not np.isnan(Images[i][j]):
					left += np.outer(U[i].T, U[i].T)
					right += int(Images[i][j]) * U[i].T
			newV.append(right.dot(np.linalg.inv(left)))
		newV = np.array(newV).T
		V = newV

		# newLoss = calculateLoss(Images, U, V, lamb)
		# diff = orignLoss - newLoss
		# print('diff is', diff)
		# print('MSE is ', newLoss)
		# predict = predictWithUV(ValidationSet, U, V)
		# accuracy = calculateAccuracy(predict, ValidationLabels)
		# print('Validation Set accuracy is ', accuracy, 'for d = ', d)
		# if abs(diff) > 3:
		# 	Changed = True
	# creat Kaggle submission file
	predict_labels = predictWithUV(query, U, V)
	indexs = [i for i in range(1, len(query) + 1)]
	data = []
	data += [indexs]
	data += [predict_labels]
	data = np.transpose(data, (1, 0)).tolist()
	first_row = [['Id', 'Category']]
	with open('3spampredict d = ' + str(d) + '.csv', 'w') as f:
	    a = csv.writer(f)
	    a.writerows(first_row)
	    a.writerows(data)
	return U, V


lamb = 0.1
Ds = [2, 5, 10, 20]

for d in Ds:
	print('begin train with d = ', d)
	U, V = findUV(Images, d, lamb)
	MSE = calculateLoss(Images, U, V, lamb)
	print('MSE is', MSE, 'for d = ', d)
	predict = predictWithUV(ValidationSet, U, V)
	accuracy = calculateAccuracy(predict, ValidationLabels)
	print('Validation Set accuracy is ', accuracy, 'for d = ', d)
	print('done train with d = ', d)


