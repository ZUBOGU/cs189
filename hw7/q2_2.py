import scipy.io as sio
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import Imputer
import operator

JokeData = sio.loadmat('./joke_data/joke_train.mat')
Images = JokeData['train']

data = [[int(i) for i in (line.strip().split(','))] for line in open("./joke_data/validation.txt", 'r')]
data = np.array(data)
ValidationSet = data[:, :2]
ValidationLabels = data[:, 2]

def calculateAccuracy(expect, actual):
	same = [i for i in range(len(actual)) if expect[i] == actual[i]]
	return len(same) / len(actual)

#get nearst Neighbors with increase in distance
def findNearestNeighbors(Images, k, point):
	distances = []
	for i in range(len(Images)):
		dist = np.linalg.norm(point - Images[i])
		distances.append((Images[i], dist))
	distances = sorted(distances, key=operator.itemgetter(1))
	neighbors = []
	for i in range(1, k + 1):
		neighbors.append(distances[i][0])
	return np.array(neighbors)

#2.2 Warm-up predict with average rating
averageRateValue = np.nanmean(Images, axis=0)
averageRatePredict = [ 1 if value > 0 else 0 for value in averageRateValue]
predict = []
for data in ValidationSet:
	index =  data[1]
	predict.append(averageRatePredict[index - 1])

accuracy = calculateAccuracy(predict, ValidationLabels)
print("Rating by its average rating, the validation set accuracy is", accuracy)

#2.2 Warm-up with predict the k nearest neighbors 

#replace Nan with 0
Ks = [10, 100, 1000]
Images[np.isnan(Images)] = 0
accuracy = 0
for k in Ks:
	print('k =', k)
	predict = []
	userId = 0
	averageRatePredict = []
	for data in ValidationSet:
		newuserId = data[0]
		if not (newuserId == userId):
			NearestNeighbors = findNearestNeighbors(Images, k, Images[newuserId - 1])
			averageRateValue = np.mean(NearestNeighbors, axis=0)
			averageRatePredict = [1 if value > 0 else 0 for value in averageRateValue]
			predict.append(averageRatePredict[data[1] - 1])
			userId = newuserId
		else:
			predict.append(averageRatePredict[data[1] - 1])
	accuracy = calculateAccuracy(predict, ValidationLabels)
	print('the validation set accuracy is ', accuracy)


