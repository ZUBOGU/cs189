import numpy as np
from numpy.linalg import inv

if __name__ == '__main__':
	X = np.array([[0, 3, 1], [1, 3, 1], [0, 1, 1], [1, 1, 1]])
	y = np.array([1, 1, -1, -1])
	w0 = np.array([-2, 1, 0])
	n = 1

	u0 = 1 / (1 + np.exp(-X.dot(w0.T)))
	print('u0 is', u0)

	Q = np.diag(y).dot(X)
	QW = Q.dot(w0)
	derivateWRTw0 = - Q.T.dot(1 / (1 + np.exp(QW)))
	w1 = w0 - n * derivateWRTw0
	print('w1 is ', w1)

	u1 = 1 / (1 + np.exp(-X.dot(w1.T)))
	print('u1 is', u1)

	QW = Q.dot(w1)
	derivateWRTw1 = - Q.T.dot(1 / (1 + np.exp(QW)))
	w2 = w1 - n * derivateWRTw1
	print('w2 is ', w2)
