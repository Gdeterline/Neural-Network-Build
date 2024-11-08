import numpy as np

# 1. create network architecture
L = 3
n = [2, 3, 3, 1]

# 2. create weights and biases
W1 = np.random.randn(n[1], n[0])
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)

# 3. create training data and labels
def prepare_data():
  X = np.array([
      [150, 70],
      [254, 73],
      [312, 68],
      [120, 60],
      [154, 61],
      [212, 65],
      [216, 67],
      [145, 67],
      [184, 64],
      [130, 69]
  ])
  y = np.array([0,1,1,0,0,1,1,0,1,0])
  m = 10
  A0 = X.T
  Y = y.reshape(n[L], m)

  return A0, Y

# 4. create activation function
def sigmoid(arr):
  return 1 / (1 + np.exp(-1 * arr))

# 5. create feed forward process
def feed_forward(A0):

  # layer 1 calculations
  Z1 = W1 @ A0 + b1
  A1 = sigmoid(Z1)

  # layer 2 calculations
  Z2 = W2 @ A1 + b2
  A2 = sigmoid(Z2)

  # layer 3 calculations
  Z3 = W3 @ A2 + b3
  A3 = sigmoid(Z3)
  
  y_hat = A3
  return y_hat

A0, Y = prepare_data()
y_hat = feed_forward(A0)