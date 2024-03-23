import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class Mnist:
    def __init__(self):
        # Load MNIST data
        self.mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()

        # Normalize data
        self.x_train = tf.keras.utils.normalize(self.x_train, axis=1)
        self.x_test = tf.keras.utils.normalize(self.x_test, axis=1)

        # Establish model architecture
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),  # Input layer
            tf.keras.layers.Dense(128, activation=tf.nn.relu),  # Hidden layer 1
            tf.keras.layers.Dense(128, activation=tf.nn.relu),  # Hidden layer 2
            tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)  # Output layer
        ])

        # Compile the model
        self.model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs):
        # Train the model
        self.model.fit(self.x_train, self.y_train, epochs=epochs)

    def evaluate(self):
        # Evaluate the model
        val_loss, val_acc = self.model.evaluate(self.x_test, self.y_test)
        print(f"Loss: {val_loss:.3f}\nAccuracy: {val_acc:.3f}")

    def predict_and_display(self, index):
        # Make predictions
        predictions = self.model.predict([self.x_test])
        prediction = np.argmax(predictions[index])

        # Display prediction and image
        print(f"Prediction: {prediction}")
        plt.imshow(self.x_test[index])
        plt.show()

    def show_menu(self):
        print("1: Train model")
        print("2: Evaluate model")
        print("3: Identify number")
        user_input = input('Select a number')
        match user_input:
            case 1:
                user_input = int(input("How many epochs: "))
                self.model.train(user_input)
            case 2:
                self.model.evaluate()
            case 3:
                user_input = int(input("Select a number between 1-60,000 to pass into neural network: "))
                self.model.predict_and_display(user_input - 1)

