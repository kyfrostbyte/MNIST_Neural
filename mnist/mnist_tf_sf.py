import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Get MNIST Data
mnist = tf.keras.datasets.mnist

# Unpack data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize Data (Represents all values between 0 and 1)
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
x_train = x_train.reshape(-1, 28, 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28).astype('float32') / 255.0

# Establish architecture of neural network
# This model will have 4 layers. An input layer, two hidden layers, and an output layer
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # Hidden Layer 1 (128 neurons, tf.nn.rellu is a default activation function)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # Hidden Layer 2
model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)) # Output Layer (10 possible outputs (one for each number), softmax is used for probability distribution)

# Parameters for training of model
# Optimizer: Control the method of cost optimization. Adam is default. Could use something like gradient descent as well
# Lost: Basically, this is the difference between the target value, and score of the network. The goal is to minimize the amount of lost, or "cost"
# Metrics: Tells what metrics you want to keep track of as the model is trained
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
# First two parameters are what you want to train
# Epochs: Defines how many cycles, or generations, will occur. If epochs=3, the network will be show each of the 60,000 images 3 times.
model.fit(x_train, y_train, epochs=3)

# Evaluate the network
val_loss, val_acc = model.evaluate(x_test, y_test)
print(f"Loss: {val_loss:.3f}\nAccuracy: {val_acc:.3f}")
predictions = model.predict([x_test])

run_loop = True
while run_loop:
    user_input = int(input("Pick and number between 1 and 60000 (Press 0 to exit): "))
    if user_input == 0:
        run_loop = False

    print(f"Prediction: {np.argmax(predictions[user_input])}")
    plt.imshow(x_test[user_input])
    plt.show()