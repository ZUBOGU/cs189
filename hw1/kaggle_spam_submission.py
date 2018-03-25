import scipy.io as sio
from sklearn import svm
import random
import numpy as np
import csv


#Use to write CSV format file for kaggle_digit submisiion with required SVM model
if __name__ == '__main__':
	#train SVM model
	trainData = sio.loadmat('./data/spam-dataset/spam_data.mat')
	trainImages = trainData['training_data'].tolist()
	actualLabels = trainData['training_labels'].tolist()[0]

	#chose random 10000 image as training samples
	samples_indexs = [i for i in range(5172)]
	random.shuffle(samples_indexs)
	sample_sets_index = samples_indexs[:5172]
	sample_sets = []
	sample_sets_labels = []
	for index in sample_sets_index:
		sample_sets += [trainImages[index]]
		sample_sets_labels += [actualLabels[index]]

	clf = svm.LinearSVC(C = 40)
	clf.fit(sample_sets, sample_sets_labels)

	#predit result
	testImages = trainData['test_data']

	predict_labels = clf.predict(testImages)
	indexs = [i for i in range(1, 5858)]
	data = []
	data += [indexs]
	data += [predict_labels]
	data = np.transpose(data, (1, 0)).tolist()
	first_row = [['Id', 'Category']]
	with open('spampredict.csv', 'w') as f:
	    a = csv.writer(f)
	    a.writerows(first_row)
	    a.writerows(data)

