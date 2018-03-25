import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

	#part a
	x1 = np.random.normal(3, 3, 100)
	N = np.random.normal(4, 2, 100)
	x2 = x1/2 + N

	print("part.a. mean of X1 is", np.mean(x1))
	print("part.a. mean of X2 is", np.mean(x2))

	print("part.a. variance of X1 is", np.var(x1))
	print("part.a. variance of X2 is", np.var(x2))

	#part b
	convaranceMatrix = np.cov(x1, x2)
	print("part.b. convaranceMatrix is \n", convaranceMatrix)

	#part c
	print("part.c.")
	eigenvalues, eigenvectors = np.linalg.eig(convaranceMatrix)
	eigenvectors = eigenvectors.T
	for i in range(len(eigenvalues)):
		print("eigenvalue is ", eigenvalues[i], "eigenvector is ", eigenvectors[i])

	#part d
	print("part.d.")
	plt.quiver(np.mean(x1), np.mean(x2), eigenvectors[0][0], eigenvectors[0][1], scale = eigenvalues[1])
	plt.quiver(np.mean(x1), np.mean(x2), eigenvectors[1][0], eigenvectors[1][1], scale = eigenvalues[0])
	plt.title('Problem 5.d All N=100 data point')
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.plot(x1, x2, 'bo')
	plt.xlim(-15, 15)
	plt.ylim(-15, 15)
	plt.show()

	#part e
	print("part.e.")
	UT = eigenvectors.T
	newX = UT.dot(np.array([x1 - np.mean(x1), x2 - np.mean(x2)]))
	newx1 = newX[0]
	newx2 = newX[1]
	plt.title('Problem 5.e ratated data point')
	plt.xlabel('newX1')
	plt.ylabel('newX2')
	plt.plot(newx1, newx2, 'bo')
	plt.xlim(-15, 15)
	plt.ylim(-15, 15)
	plt.show()