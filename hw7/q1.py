import scipy.io as sio
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


MnistData = sio.loadmat('./mnist_data/images.mat')
Images = MnistData['images']


x_dim = len(Images)
y_dim = len(Images[0])
image_index = len(Images[0][0])
Images = Images.transpose((2, 0, 1))
Images = Images.reshape(image_index, x_dim * y_dim)

def Kmean(images, k):
	#initial clusters and centers
	centers = np.random.randn(k, images[0].size)
	clusters = [[] for i in range(k)]
	for point in images:
		squres = [np.square(np.linalg.norm(center - point)) for center in centers]
		index = np.argmin(squres)
		clusters[index] += [point]
	centers = [np.mean(data, axis=0) for data in clusters]

	#repeat assign point to new centers until no change
	changed = True
	iteration = 0
	while changed:
		print(iteration)
		changed = False
		newclusters = [[] for i in range(k)]
		for i in range(k):
			cluster = clusters[i]
			for point in cluster:
				squres = [np.square(np.linalg.norm(center - point)) for center in centers]
				index = np.argmin(squres)
				newclusters[index] += [point]
				if i != index:
					changed = True
		clusters = newclusters
		centers = [np.mean(data, axis=0)for data in clusters]
		iteration += 1
	return centers

k = 5
centers = Kmean(Images, k)
print("done for k = 5")
for i in range(k):
    plt.matshow(np.reshape(centers[i], (28, 28)))
    plt.show()

k = 10
centers = Kmean(Images, k)
print("done for k = 10")
for i in range(k):
    plt.matshow(np.reshape(centers[i], (28, 28)))
    plt.show()

k = 20
centers = Kmean(Images, k)
print("done for k = 20")
for i in range(k):
    plt.matshow(np.reshape(centers[i], (28, 28)))
    plt.show()
