import scipy.io as sio
from sklearn import svm
import random
import numpy as np
import csv


#Use to write CSV format file for kaggle_digit submisiion with required SVM model
if __name__ == '__main__':
	#train SVM model
	trainData = sio.loadmat('./data/digit-dataset/train.mat')
	trainImages = trainData['train_images']

	x_dim = len(trainImages)
	y_dim = len(trainImages[0])
	image_index = len(trainImages[0][0])
	trainImages = trainImages.transpose((2, 0, 1))
	trainImages = trainImages.reshape(image_index, x_dim * y_dim).tolist()

	actualLabels = np.transpose(trainData['train_labels'],(1, 0)).tolist()[0]

	samples_indexs = [i for i in range(60000)]
	random.shuffle(samples_indexs)
	random.shuffle(samples_indexs)
	sample_sets_index = samples_indexs[:60000]
	sample_sets = []
	sample_sets_labels = []
	for index in sample_sets_index:
		sample_sets += [trainImages[index]]
		sample_sets_labels += [actualLabels[index]]

	#training with optimal value C
	clf = svm.LinearSVC(C = 0.0000001)
	clf.fit(sample_sets, sample_sets_labels)

	#predit result
	testData = sio.loadmat('./data/digit-dataset/test.mat')
	testImages = testData['test_images']
	x_dim = len(testImages)
	y_dim = len(testImages[0])
	image_index = len(testImages[0][0])
	testImages = testImages.transpose((2, 0, 1))
	testImages = testImages.reshape(image_index, x_dim * y_dim).tolist()

	predict_labels = clf.predict(testImages)
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

