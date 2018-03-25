import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.mlab import bivariate_normal
import matplotlib.pyplot as plt

if __name__ == '__main__':
	def plot(eigen_value, cov_matrix):
		x = np.arange(-8.0, 8.0, 0.01)
		y = np.arange(-8.0, 8.0, 0.01)
		X, Y = np.meshgrid(x, y)

		Z = bivariate_normal(X, Y, cov_matrix[0][0], cov_matrix[1][1], eigen_value[0], eigen_value[1], cov_matrix[0][1])
		plt.contour(X,Y,Z)
		plt.show()

	#part a
	cov_matrix = [[2,0],
                  [0,1]]
	eigen_value = [1, 1]
	plot(eigen_value, cov_matrix)

	#part b
	cov_matrix = [[3,1],
                  [1,2]]
	eigen_value = [-1, 2]
	plot(eigen_value, cov_matrix)	


	def plot1(eigen_value, cov_matrix, eigen_value1, cov_matrix1):
		x = np.arange(-10.0, 10.0, 0.01)
		y = np.arange(-10.0, 10.0, 0.01)
		X, Y = np.meshgrid(x, y)

		Z = bivariate_normal(X, Y, cov_matrix[0][0], cov_matrix[1][1], eigen_value[0], eigen_value[1], cov_matrix[0][1])
		Z1 = bivariate_normal(X, Y, cov_matrix1[0][0], cov_matrix1[1][1], eigen_value1[0], eigen_value1[1], cov_matrix1[0][1])
		plt.contour(X,Y, Z - Z1, 20)
		plt.show()


	# part c
	cov_matrix = [[1,1],
                  [1,2]]
	eigen_value = [0, 2]

	cov_matrix1 = [[1,1],
                  [1,2]]
	eigen_value1 = [2, 0]
	plot1(eigen_value, cov_matrix, eigen_value1, cov_matrix1)	
	

	# part d
	cov_matrix = [[1,1],
                  [1,2]]
	eigen_value = [0, 2]

	cov_matrix1 = [[3,1],
                  [1,2]]
	eigen_value1 = [2, 0]
	plot1(eigen_value, cov_matrix, eigen_value1, cov_matrix1)	

	#part e
	cov_matrix = [[1,0],
                  [0,2]]
	eigen_value = [1, 1]

	cov_matrix1 = [[2,1],
                  [1,2]]
	eigen_value1 = [-1, -1]
	plot1(eigen_value, cov_matrix, eigen_value1, cov_matrix1)	



