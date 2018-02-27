import numpy as np

# Sigmoid non-linearity
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

# Initializes weights and biases with zeros
def init_with_zeros(dimension):

	w = np.zeros(shape = (dimension, 1))
	b = 0

	return w,b

def propogate(w, b, X, Y):

	# w - weights
	# b - bias
	# X - input label (px * px * 3, number of examples)
	# Y - output label

	m = X.shape(1) # number of examples

	A = sigmoid(np.dot(w.T, X) + b) # compute activation
	cost = (-1 / m) * np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A)))

	dw = (1 / m) * np.dot(X, (A - Y).T)
	db = (1 / m) * np.sum(A - Y)

	# saving local gradients for backpropagation
	grads = {"dw" : dw,
			 "db" : db}

	return grads, cost		 

# Optimizes w and b by running gradient descent
def optimize (w, b, X, Y, number_iterations, learning_rate, print_cost = False):	

	costs = []

	for i in range(number_iterations):

		grads, cost = propogate(w, b, X, Y)

		dw = grads["dw"]
		db = grads["db"]

		# gradient descent
		w = w - learning_rate * dw
		b = b - learning_rate * db

		if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

	params = {"w" : w,
			  "b" : b}

	grads = {"dw" : dw,
	  		 "db" : db}
