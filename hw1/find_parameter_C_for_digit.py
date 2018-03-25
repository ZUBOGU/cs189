import scipy.io as sio
from sklearn import svm
import random
import numpy as np

if __name__ == '__main__':
	#read input
	trainData = sio.loadmat('./data/digit-dataset/train.mat')
	trainImages = trainData['train_images']

	#covert to array of 60,000's 28 x 28 arrays
	x_dim = len(trainImages)
	y_dim = len(trainImages[0])
	image_index = len(trainImages[0][0])
	trainImages = trainImages.transpose((2, 0, 1))
	trainImages = trainImages.reshape(image_index, x_dim * y_dim).tolist()

	actualLabels = np.transpose(trainData['train_labels'],(1, 0)).tolist()[0]

	#chose random 10000 image as training samples
	samples_indexs = [i for i in range(60000)]
	random.shuffle(samples_indexs)
	sample_sets_index = samples_indexs[:10000]
	sample_sets = []
	sample_sets_labels = []
	for index in sample_sets_index:
		sample_sets += [trainImages[index]]
		sample_sets_labels += [actualLabels[index]]

	sample_sets = np.reshape(sample_sets, (10, 1000, x_dim * y_dim)).tolist()
	sample_sets_labels = np.reshape(sample_sets_labels, (10, 1000)).tolist()

	#start 10-fold cross validation training
	accuracies = 0
	clf = svm.LinearSVC(C = 0.00000025)
	print("10-fold cross-validation training on 10000 samples")
	print("C = 0.00000025", )
	for i in range(10):
		print("iteration", i)
		for j in range(10):
			if j != i:
				clf.fit(sample_sets[j], sample_sets_labels[j])
		predict_labels = clf.predict(sample_sets[i])
		accuracy_indexs = [j for j in range(1000) if predict_labels[j] == sample_sets_labels[i][j]]
		accuracy = len(accuracy_indexs) / 1000
		accuracies += accuracy
	average_accuracy = accuracies / 10
	print('The accuracy rate is', average_accuracy)
