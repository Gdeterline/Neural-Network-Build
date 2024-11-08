# Building a neural network

import numpy as np

##### Initialization of the neural network

# nb_layers is the number of layers in the neural network
# nb_neurons is a list of the number of neurons in each layer

# W is a list of matrices, each matrix is the weights of a layer
# b is a list of vectors, each vector is the biases of a layer

def init_layers(nb_layers, nb_neurons):
    
    if type(nb_layers) != int:
        raise Exception("Number of layers must be integer")
    if len(nb_neurons) != nb_layers:
        raise Exception("The nb_neurons is per layer - The nb_neurons list length must match nb_layers")
    if type(nb_neurons) != list:
        raise Exception("nb_neurons parameter is supposed to be a list")

    W = []
    b = [0]
    
    for i in range(0, nb_layers-1):
        W.append(np.random.randn(nb_neurons[i+1], nb_neurons[i]))
        b.append(np.random.randn(nb_neurons[i+1], 1))
    return W, b


##### Activation functions

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0,x)

def leaky_relu(x):
    return np.maximum(0.01*x,x)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)


###### Feed forward
# X refers to the input values. X must be of the following shape: (nb_features, nb_samples). X must be a numpy array
# nb_neurons of the input layer must match the number of features
# g refers to the activation function. Default is sigmoid

def feed_forward_test(nb_layers, X, W, b, g=sigmoid):
    
    # Initialize the input layer
    A = [0 for i in range(nb_layers)]
    A[0] = X.T
    Z = [0 for i in range(0, nb_layers)]
    Z[0] = A[0]
    
    # Neuron values updated with the forward pass
    for i in range(1, len(W)+1):
        Z[i] = W[i-1] @ A[i-1] + b[i-1]
        A[i] = g(Z[i])
    return A, Z, A[-1]


def feed_forward(nb_layers, nb_neurons, X, g=sigmoid):
    
    
    
    # Initialize the input layer
    A = [0 for i in range(nb_layers)]
    A[0] = X.T
    Z = [0 for i in range(0, nb_layers)]
    Z[0] = A[0]
    W, b = init_layers(nb_layers, nb_neurons)

    print(W)
    print("----------------")
    print(b)
    print(len(b))
    print("----------------")
    
    # Neuron values updated with the forward pass
    for i in range(1, len(W)):
        Z[i] = W[i-1] @ A[i-1] + b[i]
        A[i] = g(Z[i])
    return A, Z, A[-1]


#print(feed_forward(3, [2, 2, 1], np.array([[1, 2], [3, 4]]), g=sigmoid))

W = [np.array([[1.], [2.]]), np.array([[1., 2.]])]
b = [np.array([[1.], [2.]]), np.array([[3.]])]

L=3
    # n = [1, 2, 1] number of neurons
X = np.array([[1.]])
    
A, Z, output = feed_forward_test(L, X, W, b)

print(Z)
