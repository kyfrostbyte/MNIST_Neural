import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load data from CSV
data = pd.read_csv('data/mnist_train.csv')

# Convert data into a numpy array, get dimensions, and shuffle
# m = amount of rows (images), n = pixels + 1, extra one is due to label
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# Flip data so each column is an image, instead of each row
data_dev = data[0:1000].T

# After transformation, the first row now contains the labels for all images, and is saved into the Y_dev variable
# The remaining data is the pixel values for each image, and is saved into X_dev
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

# Prepare training data
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

# Y_train = labels, X_train = pixel value, remove shape to show pixel values of first image
print(Y_train)
print(X_train[:, 0].shape)


def init_params():
    # Set up initial weights and biases between input and hidden layer.
    # All values between 0 and 1, then subtract .5 to get values to be between -0.5 and 0.5
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5

    # Set up initial weights and biases between hidden and output layer
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


# Forward prop starts with an input, and returns a prediction
# After each prediction is made, backwards prop is used to refine the network for the next forward prop
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 # Unactivated hidden layer
    A1 = ReLU(Z1) # Activation function
    Z2 = W2.dot(A1) + b2 # Activated hidden layer
    A2 = softmax(Z2) # Output prediction
    return Z1, A1, Z2, A2


# Backward prop starts with prediction, and works backwards towards the input
# It analyzes how far the prediction was from the label (correct answer), and then we find out how much the previous weights
# and biases contributed to that deviation
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y) # Represents the correct label
    dZ2 = A2 - one_hot_Y # Represents how much the prediction was from the label (error of 2nd layer)
    dW2 = 1 / m * dZ2.dot(A1.T) # How much did the previous weight contribute to that error? Derivative of loss with respect to weights in layer 2
    db2 = 1 / m * np.sum(dZ2) # Average of the absolute error
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) # Applies error from second layer, to work backwards to get to errors in first layer
    dW1 = 1 / m * dZ1.dot(X.T) # Same thing as dw2 and db2, but for first layer
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Figures out how much to adjust weights and biases based off of information gained during the backwards prop
# The updates are applied before each iteration, so the network should improve a little bit after each iteration
# Alpha is the learning rate. I have it set to .1, so the edits to the weights and biases are not too drastic
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


# This is the optimization function
# Its job is to figure out where the network is at, and what can be done to minimize loss
# Here is where we run the loop for the training of the network
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params() # Start with initial params
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X) # Do forward prop
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y) # Do backward prop, and analyze where improvements can be made
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha) # Implement those improvements, on a very small scale
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)

            print(f"Accuracy: {accuracy * 100:.2f}%")
    return W1, b1, W2, b2 # Returns the final weights and biases of the trained network


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


# Returns label
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


# Activation function
def ReLU(Z):
    return np.maximum(Z, 0)


def ReLU_deriv(Z):
    return Z > 0


def softmax(Z):
    # Preserves amount of columns, collapses amount of rows with np.sum
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    print("")

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 350)


print("\n----------------------------------------")
print("Testing of the network:")
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(5, W1, b1, W2, b2)
test_prediction(10, W1, b1, W2, b2)