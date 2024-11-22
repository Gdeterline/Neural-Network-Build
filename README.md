# Building a Neural Network

This project aims at creating a neural network from scratch, to apprehend not only the theoretical part but also the practical part.

## Table of Contents

1. [Introduction](#introduction)
    
    1.1. [Why build a Neural Network from scratch?](#why-building-a-neural-network-from-scratch)
    
    1.2. [What is the goal of this project?](#what-is-the-goal-of-this-project)

2. [Neural Network notions](#notions)

    2.1. [Neural Network](#neural-network)

    2.2. [Notations](#notations)
    
    2.3. [Activation Function](#activation-function)
    
    2.4. [Feed-forward](#feed-forward)
    
    2.5. [Loss Function](#loss-function)
    
    2.6. [Backpropagation](#backpropagation)

3. [Implementation](#implementation)

4. [First batch of results](#results)

5. [Optimization](#optimization)

6. [Second batch of results](#results)

7. [Conclusion](#conclusion)


## Introduction <a name="introduction"></a>

### Why build a Neural Network from scratch? <a name="why-building-a-neural-network-from-scratch"></a>

I'm really into data and AI, so I chose to work on data science projects on the side. Among the algorithms to learn are neural networks.
I believe that building a neural network from scratch is a great way to understand how it works. It allows you to apprehend the theoretical part but also the practical part. It is a great way to understand the different components of a neural network and how they interact with each other.
Then, explaining the code to someone else is a great way to consolidate the knowledge acquired. So I hope the documentation will be clear enough for you to understand how to build a NN. That would mean I'm not too bad at it! ^^

### What is the goal of this project? <a name="what-is-the-goal-of-this-project"></a>

The goal of this project is to build a neural network from scratch, that meets the following expectations:

<ins>**Requirements:**</ins>

The neural network must be able to:
- take any number of inputs
- have any number of hidden layers
- have any number of neurons in each hidden layer
- have any number of outputs
- use any activation function
- use any loss function

<ins>**Constraints:**</ins>

The neural network must use the following algorithm:
- backpropagation algorithm

To summarise the goal of this project, I want to build a neural network that is as modular as possible, while using the most common algorithms in the field of neural networks.

## Neural Network notions <a name="notions"></a>

In this section, I will explain the different notions that are important to understand when building a neural network.

### Neural Network <a name="neural-network"></a>

A neural network is a set of algorithms, modeled after the human brain, that is designed to recognize patterns. 
It is composed of layers of neurons, each of which applies an activation function to the weighted sum of its inputs.
These layers are divided into three types:
- the input layer, which receives the input data
- the hidden layers, which process the data
- the output layer, which produces the output data
The idea is to do a forward pass, where the input data is passed through the network, and the output is predicted.
Then, the neural network uses a loss function to measure the difference between the predicted output and the actual output. The goal is to minimize this difference by adjusting the weights of the neurons.
This is done through the backpropagation algorithm, which calculates the gradient of the loss function with respect to the weights and biaises of the neurons, and updates the weights/biaises accordingly.
Finally, we can train the neural network by repeating the forward pass and the backpropagation algorithm until the loss function is minimized.

The training process is usually divided into several steps:
- The first step consists in splitting the data into a training set and a test set.
- The second step consists in normalizing the data, to make it easier for the neural network to learn.
- The third step consists in initializing the weights and biaises of the neurons.
- The fourth step consists in training the neural network on the training set, by doing a forward pass and a backpropagation algorithm.
- The fifth step consists in evaluating the neural network on the testing set, to see how well it generalizes to new data.
- The sixth step consists in tuning the hyperparameters of the neural network, to improve its performance.

### Notations <a name="notations"></a>

In order to build a neural network, we need to define the following notations:

- **L**: the number of layers of the neural network. 

- **n**: the list of the number of neurons in each layer of the neural network, which is a list of integers of length L. The value of n[0] is the number of features of the input data X.

- **g**: the activation function of the neural network.

- **X**: the input data, which is a matrix of shape (n, m), where n is the number of features and m is the number of samples.

- **W**: the list of weight matrices of the neural network, which is a list of matrices of shape (n[i], n[i-1]), where n[i] is the number of neurons in the i-th layer of the neural network, and n[i-1] is the number of neurons in the (i-1)-th layer of the neural network.

- **b**: the list of bias vectors of the neural network, which is a list of vectors of shape (n[i], 1), where n[i] is the number of neurons in the i-th layer of the neural network.

- **Z**: the list of weighted sums of the neural network (to which are added the biaises), which is a list of matrices of shape (n[i], m), where n[i] is the number of neurons in the i-th layer of the neural network, and m is the number of samples.

- **A**: the list of output matrices of the neural network, which is a list of matrices of shape (n[i], m), where n[i] is the number of neurons in the i-th layer of the neural network, and m is the number of samples. The value of A[0] is the input data X. The difference between A and Z is that A is the output of the activation function applied to Z.

- **$\hat{Y}$**: the prediction of the neural network, which is the output of the output layer of the neural network.



_**The list needs to be completed along the way, as we will need to define other matrices and vectors to build the neural network.**_


### Activation Function <a name="activation-function"></a>

An activation function is a mathematical function that is applied to the weighted sum of the inputs of a neuron, to introduce non-linearity into the neural network.

There are several activation functions that can be used in a neural network. The most common ones are:
- **the Sigmoid Function**

    <ins>Use Case:</ins> Output layer of a binary classification model.

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$


- **the Tanh Function**

    <ins>Use Case:</ins> Often preferred in hidden layers when inputs can be negative.

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$


- **the ReLU Function**

    <ins>Use Case:</ins> Very commonly used in hidden layers for deep networks.

$$
f(x) = \max(0, x)
$$


- **the Leaky ReLU Function**

    <ins>Use Case:</ins> A variant of the ReLU function that allows a small gradient when the input is negative.

$$
f(x) = \max(0.01x, x)
$$


- **the Softmax Function**

    <ins>Use Case:</ins> Output layer of a multi-class classification model.

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

In order to build a neural network, we have to choose an activation function. First we will build it using the sigmoid function. It is set as default function, but the idea is for the code to be modular, so in the end it should be possible to choose the activation function.


### Feed Forward <a name="feed-forward"></a>

The feed-forward algorithm is the process of passing the input data through the neural network, from the input layer to the output layer, to make a prediction.

The feed-forward algorithm goes like follows:
- The input data is passed through the input layer, and the output of the input layer is passed to the first hidden layer.
- The output of the first hidden layer is passed to the second hidden layer, and so on, until the output layer.
- The output of the output layer is the prediction of the neural network.

The feed-forward algorithm is the first step of the training process of a neural network. It allows the neural network to make a prediction, which can be compared to the actual output to calculate the loss function.

Our input data is a matrix of shape (n, m), where n is the number of features and m is the number of samples. This matrix will be defined as the X matrix.

The value of A[0] is the input data X.
We define the value of Z[i] as the weighted sum of the inputs of the i-th layer of the neural network, to which we add the bias of the i-th layer. The value of Z[i] is calculated as follows:

$$ Z[i] = W[i-1] \cdot A[i-1] + b[i] $$

Where:
- W[i-1] is the weight matrix of the links between the (i-1)-th and the i-th layer of the neural network.
- A[i-1] is the output of the (i-1)-th layer of the neural network.
- b[i] is the bias vector of the i-th layer of the neural network.

The value of A[i] is the output of the i-th layer of the neural network, after applying the activation function to the value of Z[i]. The value of A[i] is calculated as follows:

$$ A[i] = g(Z[i]) $$

Where:

- g is the activation function of the i-th layer of the neural network.

The value of A[i] is the output of the i-th layer of the neural network.

Finally, the value of $\hat{Y}$ is the output of the output layer of the neural network, which is the prediction of the neural network. The value of $\hat{Y}$ is calculated as follows:

$$ \hat{Y} = A[L] $$

Where:

- L is the number of layers of the neural network.
- A[L] is the output of the output layer of the neural network.


To build the feed-forward algorithm, we need to define the following functions:
- the **init_layers** function, which initializes the weights and biaises of the neural network. It takes as input the number of required layers of the neural network, and the number of neurons in each layer of the neural network, and returns both the lists of weight matrices and bias vectors of the neural network.
- the **feed_forward** function, which applies the feed-forward algorithm to the neural network. It takes as input the input data X, the number of layers required, the number of neurons in each layer of the neural network. This enables the function to call the init_layers function to initialize the weights and biaises of the neural network. The function also takes as input the activation function to be used in the neural network. The default activation function is the sigmoid function. Finally, the function returns the list of A matrices, Z matrices, and the prediction of the neural network. Theoretically, the A and Z lists should not be returned, but they are useful for debugging purposes.

These functions are tested in the nnTesting.ipynb notebook.
They are first tested on specific examples, and then generalized through random examples.

The feed-forward algorithm is the first step of the training process of a neural network. It allows the neural network to make a prediction, which can be compared to the actual output to calculate the loss function.


### Loss Function <a name="loss-function"></a>

Non convex functions - local minimums can be a problem
binary cross entropy loss function

### Backpropagation <a name="backpropagation"></a>

## Implementation <a name="implementation"></a>

## First batch of results <a name="results"></a>

## Optimization <a name="optimization"></a>

## Second batch of results <a name="results"></a>

## Conclusion <a name="conclusion"></a>







