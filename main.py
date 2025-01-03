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
    b = [np.zeros((nb_neurons[0], 1))]
    
    for i in range(0, nb_layers-1):
        W.append(np.random.randn(nb_neurons[i+1], nb_neurons[i])*np.sqrt(2/nb_neurons[i])) # 
        b.append(np.zeros((nb_neurons[i+1], 1)))
    #print("Expected number of features: ", len(W[0][0]))
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

###### Derivatives of the activation functions

def sigmoid_derivative(x, sigmoid=sigmoid):
    return sigmoid(x)*(1-sigmoid(x))

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu_derivative(x, relu=relu):
    return np.where(relu(x) <= 0, 0, 1)

def leaky_relu_derivative(x):
    return np.where(x <= 0, 0.01, 1)


###### Feed forward function
# X refers to the input values. X must be of the following shape: (nb_features, nb_samples). X must be a numpy array
# nb_neurons of the input layer must match the number of features
# g refers to the activation function. Default is sigmoid

def feed_forward(nb_layers, X, W, b, g=sigmoid):
    #print("Assertion going on")
    assert len(X[0]) == len(W[0][0]), "Expected shape of X: number of samples in line, number of features in column"
    #print("Assertation value ok")
    # Initialize the input layer
    A = [0 for i in range(nb_layers)]
    A[0] = X.T
    Z = [0 for i in range(0, nb_layers)]
    Z[0] = A[0]
    
    # Neuron values updated with the forward pass
    for i in range(1, nb_layers):
        
        #print(W[i-1].shape) # debugging purposes
        #print(A[i-1].shape) # debugging purposes
        #print(A[i-1]) # debugging purposes
        
        Z[i] = W[i-1] @ A[i-1] + b[i]
        A[i] = g(Z[i])
    return A, Z, A[-1]


##### Loss Functions 

def binary_cross_entropy(y, y_hat, nb_samples):
    return -(1/nb_samples)*np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

def mean_squared_error(y, y_hat, nb_samples):
    return 1/nb_samples*np.sum((y - y_hat)**2)

##### Loss derivative

def binary_cross_entropy_derivative(y, y_hat):
    return -y/y_hat + (1-y)/(1-y_hat)

def mean_squared_error_derivative(y, y_hat):
    return y_hat - y


##### Backpropagation

def backpropagation(nb_layers, X, y, W, b, A, Z, y_hat, g=sigmoid, gder=sigmoid_derivative, lossder=mean_squared_error_derivative):
        
        # Initialize the gradients
        grad_W = [np.zeros_like(w) for w in W]
        grad_b = [np.zeros_like(bias) for bias in b]

        # Number of samples
        nb_samples = X.shape[1]
        
        # Compute the loss
        dL_dA = lossder(y, y_hat)
        dA_dZ = gder(Z[-1])
        delta = dL_dA * dA_dZ
        
        # Backpropagate through layers
        for i in range(nb_layers-1, 0, -1):
            grad_W[i-1] = delta @ A[i-1].T / nb_samples
            grad_b[i] = np.sum(delta, axis=1, keepdims=True) / nb_samples
            
            if i > 1:
                delta = (W[i-1].T @ delta) * gder(Z[i-1])
                
        return grad_W, grad_b
    
##### Update the weights and biases

def update_parameters(W, b, grad_W, grad_b, learning_rate):
    for i in range(len(W)):
        W[i] = W[i] - learning_rate * grad_W[i]
        b[i+1] = b[i+1] - learning_rate * grad_b[i+1]
    return W, b

def train_nn(nb_layers, X, y, nb_neurons, learning_rate, epochs, g=sigmoid, gder=sigmoid_derivative, loss=mean_squared_error, lossder=mean_squared_error_derivative):
    
    Epochs = []
    Losses = []
    nb_samples = len(X)

    #### Improvement suggestions - automatise the number of neurons in the input layer based on the number of features of the input data
    W, b = init_layers(nb_layers, nb_neurons)
    
    for i in range(epochs):
        A, Z, y_hat = feed_forward(nb_layers, X, W, b, g)
        grad_W, grad_b = backpropagation(nb_layers, X, y, W, b, A, Z, y_hat, g, gder, lossder)
        W, b = update_parameters(W, b, grad_W, grad_b, learning_rate)
        Epochs.append(i)
        Losses.append(loss(y, y_hat, nb_samples))
                
        if i % 500 == 0:
            print(f"Epoch {i} - Loss: {loss(y, y_hat, nb_samples)}")
            Epochs.append(i)
            Losses.append(loss(y, y_hat, nb_samples))

    return W, b, Epochs, Losses

def predict(X, W, b, g=sigmoid):
    A, Z, y_hat = feed_forward(len(b), X, W, b, g)
    return y_hat