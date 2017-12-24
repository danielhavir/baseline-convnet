import numpy as np
import tensorflow_model.train as tf_train

LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
epochs = 10
batch_size = 128
keep_probability = 0.5
learning_rate = 3e-4

# TensorFlow
tf_train.run(epochs, batch_size, learning_rate, keep_probability)
