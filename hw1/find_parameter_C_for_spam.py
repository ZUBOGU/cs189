import scipy.io as sio
from sklearn import svm
import random
import numpy as np

if __name__ == '__main__':
	#read input
	trainData = sio.loadmat('./data/spam-dataset/spam_data.mat')
	trainImages = trainData['training_data'].tolist()
	actualLabels = trainData['training_labels'].tolist()[0]

	#chose random 5172 data as training samples
	samples_indexs = [i for i in range(5172)]
	random.shuffle(samples_indexs)
	sample_sets_index = samples_indexs[:5172]
	sample_sets = []
	sample_sets_labels = []
	for index in sample_sets_index:
		sample_sets += [trainImages[index]]
		sample_sets_labels += [actualLabels[index]]

	sample_sets = np.reshape(sample_sets, (6, 862, 32)).tolist()
	sample_sets_labels = np.reshape(sample_sets_labels, (6, 862)).tolist()

	#start 6-fold cross validation training
	accuracies = 0
	clf = svm.LinearSVC(C = 40)
	print("6-fold cross-validation training on 5172 samples")
	print("C = 40", )
	for i in range(6):
		print("iteration", i)
		for j in range(6):
			if j != i:
				clf.fit(sample_sets[j], sample_sets_labels[j])
		predict_labels = clf.predict(sample_sets[i])
		accuracy_indexs = [j for j in range(862) if predict_labels[j] == sample_sets_labels[i][j]]
		accuracy = len(accuracy_indexs) / 862
		accuracies += accuracy
	average_accuracy = accuracies / 6
	print('The accuracy rate is', average_accuracy)
