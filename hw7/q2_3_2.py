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

def calculateAccuracy(expect, actual):
	same = [i for i in range(len(actual)) if expect[i] == actual[i]]
	return len(same) / len(actual)

def predictWithUV(Dataset, vectorU, vectorV):
	predict = []
	for data in Dataset:
		estimate = vectorU[data[0] - 1].dot(vectorV[:, data[1] - 1])
		predict += [1] if estimate > 0 else [0]
	return predict

# newImages replace Nan with 0
newImages = Images
newImages[np.isnan(newImages)] = 0
# newImages = preprocessing.normalize(newImages.astype("float"), norm='l2', axis=0)

# 2.3.1 compute U and V
U, s, V = np.linalg.svd(newImages, full_matrices=False)
U = U.dot(np.diag(s))

#2.3.2 with d vary compute MSE and validation set accuracies
Ds = [2, 5, 10, 20]
for d in Ds:
	newU = U[:, :d+1]
	newV = V[:d+1, :]
	newR = newU.dot(newV)
	MSE = np.square(np.linalg.norm(newR - newImages))
	print('MSE is', MSE, 'for d = ', d)
	predict = predictWithUV(ValidationSet, newU, newV)
	accuracy = calculateAccuracy(predict, ValidationLabels)
	print('Validation Set accuracy is ', accuracy, 'for d = ', d)

	# creat Kaggle submission file
	predict_labels = predictWithUV(query, newU, newV)
	indexs = [i for i in range(1, len(query) + 1)]
	data = []
	data += [indexs]
	data += [predict_labels]
	data = np.transpose(data, (1, 0)).tolist()
	first_row = [['Id', 'Category']]
	with open('2spampredict d = ' + str(d) + '.csv', 'w') as f:
	    a = csv.writer(f)
	    a.writerows(first_row)
	    a.writerows(data)

