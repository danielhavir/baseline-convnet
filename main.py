import tensorflow_model.train as tf_train
from pytorch_model.train import BaselineConvnet

LABEL_NAMES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
epochs = 25
batch_size = 128
keep_probability = 0.5
learning_rate = 3e-4

# TensorFlow
# tf_train.run(epochs, batch_size, learning_rate, keep_probability)

# PyTorch
baseline_convnet = BaselineConvnet(batch_size, keep_probability, learning_rate)
baseline_convnet.run(epochs, LABEL_NAMES)
