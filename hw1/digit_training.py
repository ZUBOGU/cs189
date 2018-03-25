import scipy.io as sio
from sklearn import svm
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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

	#partition samples
	samples_indexs = [i for i in range(60000)]
	random.shuffle(samples_indexs)
	validation_sets_index = samples_indexs[:10000]
	validation_sets = []
	validation_sets_labels = []
	for index in validation_sets_index:
		validation_sets += [trainImages[index]]
		validation_sets_labels += [actualLabels[index]]
	samples_indexs = samples_indexs[10000:60000]

	error_rates = []
	#train classifler with 100 sample
	clf = svm.LinearSVC()
	print("Train 100 samples")
	random.shuffle(samples_indexs)
	small_samples_indexs = samples_indexs[:100]
	samples_sets = []
	samples_sets_labels = []
	for index in small_samples_indexs:
		samples_sets += [trainImages[index]]
		samples_sets_labels += [actualLabels[index]]

	clf.fit(samples_sets, samples_sets_labels)
	predict_labels = clf.predict(validation_sets)
	accuracy_indexs = [i for i in range(10000) if predict_labels[i] == validation_sets_labels[i]]
	accuracy = len(accuracy_indexs) / 10000
	print('The error rate is', 1 - accuracy)
	error_rates += [1 - accuracy]
	cm1 = confusion_matrix(validation_sets_labels, predict_labels)

	#train classifler with 200 sample
	clf = svm.LinearSVC()
	print("Train 200 samples")
	random.shuffle(samples_indexs)
	small_samples_indexs = samples_indexs[:200]
	samples_sets = []
	samples_sets_labels = []
	for index in small_samples_indexs:
		samples_sets += [trainImages[index]]
		samples_sets_labels += [actualLabels[index]]

	clf.fit(samples_sets, samples_sets_labels)
	predict_labels = clf.predict(validation_sets)
	accuracy_indexs = [i for i in range(10000) if predict_labels[i] == validation_sets_labels[i]]
	accuracy = len(accuracy_indexs) / 10000
	print('The error rate is', 1 - accuracy)
	error_rates += [1 - accuracy]
	cm2 = confusion_matrix(validation_sets_labels, predict_labels)

	#train classifler with 500 sample
	clf = svm.LinearSVC()
	print("Train 500 samples")
	random.shuffle(samples_indexs)
	small_samples_indexs = samples_indexs[:500]
	samples_sets = []
	samples_sets_labels = []
	for index in small_samples_indexs:
		samples_sets += [trainImages[index]]
		samples_sets_labels += [actualLabels[index]]

	clf.fit(samples_sets, samples_sets_labels)
	predict_labels = clf.predict(validation_sets)
	accuracy_indexs = [i for i in range(10000) if predict_labels[i] == validation_sets_labels[i]]
	accuracy = len(accuracy_indexs) / 10000
	print('The error rate is', 1 - accuracy)
	error_rates += [1 - accuracy]
	cm3 = confusion_matrix(validation_sets_labels, predict_labels)

	#train classifler with 1000 sample
	clf = svm.LinearSVC()
	print("Train 1000 samples")
	random.shuffle(samples_indexs)
	small_samples_indexs = samples_indexs[:1000]
	samples_sets = []
	samples_sets_labels = []
	for index in small_samples_indexs:
		samples_sets += [trainImages[index]]
		samples_sets_labels += [actualLabels[index]]

	clf.fit(samples_sets, samples_sets_labels)
	predict_labels = clf.predict(validation_sets)
	accuracy_indexs = [i for i in range(10000) if predict_labels[i] == validation_sets_labels[i]]
	accuracy = len(accuracy_indexs) / 10000
	print('The error rate is', 1 - accuracy)
	error_rates += [1 - accuracy]
	cm4 = confusion_matrix(validation_sets_labels, predict_labels)

	#train classifler with 2000 sample
	clf = svm.LinearSVC()
	print("Train 2000 samples")
	random.shuffle(samples_indexs)
	small_samples_indexs = samples_indexs[:2000]
	samples_sets = []
	samples_sets_labels = []
	for index in small_samples_indexs:
		samples_sets += [trainImages[index]]
		samples_sets_labels += [actualLabels[index]]

	clf.fit(samples_sets, samples_sets_labels)
	predict_labels = clf.predict(validation_sets)
	accuracy_indexs = [i for i in range(10000) if predict_labels[i] == validation_sets_labels[i]]
	accuracy = len(accuracy_indexs) / 10000
	print('The error rate is', 1 - accuracy)
	error_rates += [1 - accuracy]
	cm5 = confusion_matrix(validation_sets_labels, predict_labels)

	#train classifler with 5000 sample
	clf = svm.LinearSVC()
	print("Train 5000 samples")
	random.shuffle(samples_indexs)
	small_samples_indexs = samples_indexs[:5000]
	samples_sets = []
	samples_sets_labels = []
	for index in small_samples_indexs:
		samples_sets += [trainImages[index]]
		samples_sets_labels += [actualLabels[index]]

	clf.fit(samples_sets, samples_sets_labels)
	predict_labels = clf.predict(validation_sets)
	accuracy_indexs = [i for i in range(10000) if predict_labels[i] == validation_sets_labels[i]]
	accuracy = len(accuracy_indexs) / 10000
	print('The error rate is', 1 - accuracy)
	error_rates += [1 - accuracy]
	cm6 = confusion_matrix(validation_sets_labels, predict_labels)

	#train classifler with 10000 sample
	clf = svm.LinearSVC()
	print("Train 10000 samples")
	random.shuffle(samples_indexs)
	small_samples_indexs = samples_indexs[:10000]
	samples_sets = []
	samples_sets_labels = []
	for index in small_samples_indexs:
		samples_sets += [trainImages[index]]
		samples_sets_labels += [actualLabels[index]]

	clf.fit(samples_sets, samples_sets_labels)
	predict_labels = clf.predict(validation_sets)
	accuracy_indexs = [i for i in range(10000) if predict_labels[i] == validation_sets_labels[i]]
	accuracy = len(accuracy_indexs) / 10000
	print('The error rate is', 1 - accuracy)
	error_rates += [1 - accuracy]
	cm7 = confusion_matrix(validation_sets_labels, predict_labels)

	#plot graph for problem 1
	x_label = [100,200,500,1000,2000,5000,10000]
	plt.title('Problem 1. The error rates on validation set versus training example sizes')
	plt.xlabel('Traning sample size')
	plt.ylabel('Error rate')
	plt.plot(x_label, error_rates, 'bo-')
	plt.show()

	#plot confusion matrices for problem 2
	plt.matshow(cm1)
	plt.title('Problem 2. Confusion matrix for 100 examples')
	plt.colorbar()
	plt.ylabel('True digit')
	plt.xlabel('Predicted digit')
	plt.show()

	plt.matshow(cm2)
	plt.title('Problem 2. Confusion matrix for 200 examples')
	plt.colorbar()
	plt.ylabel('True digit')
	plt.xlabel('Predicted digit')
	plt.show()

	plt.matshow(cm3)
	plt.title('Problem 2. Confusion matrix for 500 examples')
	plt.colorbar()
	plt.ylabel('True digit')
	plt.xlabel('Predicted digit')
	plt.show()

	plt.matshow(cm4)
	plt.title('Problem 2. Confusion matrix for 1000 examples')
	plt.colorbar()
	plt.ylabel('True digit')
	plt.xlabel('Predicted digit')
	plt.show()

	plt.matshow(cm5)
	plt.title('Problem 2. Confusion matrix for 2000 examples')
	plt.colorbar()
	plt.ylabel('True digit')
	plt.xlabel('Predicted digit')
	plt.show()

	plt.matshow(cm6)
	plt.title('Problem 2. Confusion matrix for 5000 examples')
	plt.colorbar()
	plt.ylabel('True digit')
	plt.xlabel('Predicted digit')
	plt.show()

	plt.matshow(cm7)
	plt.title('Problem 2. Confusion matrix for 10000 examples')
	plt.colorbar()
	plt.ylabel('True digit')
	plt.xlabel('Predicted digit')
	plt.show()

	#problem 3. train 10000 samples with parameter C=0.00000025.
	#Use 10-fold cross validation training
	clf = svm.LinearSVC(C = 0.00000025)
	print("Train 10000 samples with parameter C = 0.00000025")
	random.shuffle(samples_indexs)
	small_samples_indexs = samples_indexs[:10000]
	samples_sets = []
	samples_sets_labels = []
	for index in small_samples_indexs:
		samples_sets += [trainImages[index]]
		samples_sets_labels += [actualLabels[index]]

	clf.fit(samples_sets, samples_sets_labels)
	predict_labels = clf.predict(validation_sets)
	accuracy_indexs = [i for i in range(10000) if predict_labels[i] == validation_sets_labels[i]]
	accuracy = len(accuracy_indexs) / 10000
	print('The error rate is', 1 - accuracy)
