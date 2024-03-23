# Overview

With AI's explosion in popularity over these past few years, I have been interested to get my feet wet with some sort of machine learning.
I set out to train a neural network to be able to recognize a number from an image of a handwritten number. I chose this, mainly because I had no experience 
with neural networks, and this was said to be a common first project. The MNIST dataset provides 60,000 images of individual numbers, which
I used to train the neural network.

I followed some guides and was pretty quickly able to get a network functional using the tensorflow library. That library actually makes it incredibly easy,
and could be done in less than 15-20 lines or so. However, I wanted to understand what was happening under the hood so to speak.
So, I then began to research how to program neural networks without any libraries. This involves doing all the math to adjust the weights and 
biases manually, instead of using pre-made functions. Luckily, there is quite a bit of documentation on the MNIST data set, so I had lots of resources.

The final result of the project is 2 networks, one using libraries, and one that does it from scratch. By doing it from scratch,
I was able to have a much better understanding over what was being done in each step. If you are familiar with calculus, the concepts
are not terribly complex, but it is still difficult to keep track of so much information. Using libraries does not expose you to the math, so
it is harder to fully grasp what is happening. The from-scratch version includes a lot of comments that I wrote, mainly so that I could drill the concepts into my own head.


[Software Demo Video](https://youtu.be/f7VtDW6oKnQ)

# Development Environment

I chose to use PyCharm as the IDE for this project. The mnist_tf_sf.py file relies predominantly on the tensorflow library, which has
keras built in. Tensorflow allows you to use tensors, which are essentially n-dimensional matrices, that make it much easier to lay out the
structure of the network. Tensorflow also allows you to utilize built in optimization, loss, and activation functions, which simplifies things quite a bit.

The mnist_from_scratch version really just needs numpy and pandas to work, as all the computations are done in house. Both these libraries
provides methods needed to organize the data, and perform some unique mathematical operations on them.

Libraries:
* tensorflow
* matplotlib
* keras
* numpy
* pandas
* matplotlib

# Useful Websites

- [TensorFlow](https://www.tensorflow.org/datasets/catalog/mnist)
- [Wiki for backprop](https://en.wikipedia.org/wiki/Backpropagation#:~:text=Essentially%2C%20backpropagation%20evaluates%20the%20expression,%22backwards%20propagated%20error%22).)
- [MNIST CSV download](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download&select=mnist_train.csv)
- [NEAT](https://neat-python.readthedocs.io/en/latest/neat_overview.html)

# Future Work

There is not a ton left to do with this specific network, as it has a very specific role, and does it well. I could work on optimizing it more to get the accuracy to be higher after
all the training is done, but there isn't a ton left that I could do. Instead of refining this project, I would like to train other networks all together.
I plan on training a network on the CIFAR-10 dataset, which is used for full on image recognition, not just numbers.
I also want to learn more about using the NEAT system of training networks to see what other applications it has.  