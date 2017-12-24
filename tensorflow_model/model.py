import tensorflow as tf
import numpy as np


def placeholders(input_shape, num_classes):
	"""
	Create placeholders.

	:param image_shape: Shape of the input data
	:param num_classes: Number of classes
	: return: Tensors placeholders
	"""

	inputs = tf.placeholder(tf.float32, [None, *input_shape], name='inputs')
	targets = tf.placeholder(tf.float32, [None, num_classes], name='targets')
	is_training = tf.placeholder(tf.bool, name='is_training')
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	lr = tf.placeholder(tf.float32, name='learning_rate')

	return (inputs, targets, is_training, keep_prob, lr)


def batch_norm(layer, is_training, epsilon=1e-4):
	"""
	Create a convolutional layer with the given layer as input.

	:param layer: TensorFlow tensor
	:param is_training: bool or Tensor
		Indicates whether or not the network is currently training, which tells the batch normalization
		layer whether or not it should update or use its population statistics.
	:returns Normalized tensor
	"""

	# Check if we normalize convolutional layer
	if "conv" in layer.name.lower():
		num_units = layer.get_shape().as_list()[3]
		moments = [0,1,2]
	else:
		num_units = layer.get_shape().as_list()[1]
		moments = [0]

	gamma = tf.Variable(tf.ones([num_units]))
	beta = tf.Variable(tf.zeros([num_units]))

	pop_mean = tf.Variable( tf.ones([num_units]), trainable=False )
	pop_variance = tf.Variable( tf.zeros([num_units]), trainable=False )

	def batch_norm_training():
		batch_mean, batch_variance = tf.nn.moments(layer, moments)

		decay = 0.99
		train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
		train_variance = tf.assign(pop_variance, pop_variance * decay + batch_variance * (1 - decay))

		with tf.control_dependencies([train_mean, train_variance]):
			return tf.nn.batch_normalization(layer, batch_mean, batch_variance, beta, gamma, epsilon)

	def batch_norm_inference():
		return tf.nn.batch_normalization(layer, pop_mean, pop_variance, beta, gamma, epsilon)

	return tf.cond(is_training, batch_norm_training, batch_norm_inference)


def conv2d(inputs, conv_num_outputs, conv_ksize, conv_strides,
			is_training, activation_fn=tf.nn.relu,
			use_bias=False):
	"""
	:param inputs: TensorFlow Tensor
	:param conv_num_outputs: Number of outputs for the convolutional layer
	:param conv_ksize: kernal size 2-D Tuple for the convolutional layer
	:param conv_strides: Stride 2-D Tuple for convolution
	:param is_training: Boolean passed into Batch Norm to control dependencies
	: return: A tensor that represents convolution of inputs
	"""

	# Weight initialization reference: https://arxiv.org/abs/1502.01852
	# E.g.: n = 32 * 3 * 3 (32 filters of size 3x3)
	n = conv_num_outputs * np.prod(conv_ksize)
	weights = tf.Variable(
				tf.truncated_normal(
					[*conv_ksize, int(inputs.shape[-1]), conv_num_outputs],
					stddev=np.sqrt(2.0/n)
				)
			)

	conv = tf.nn.conv2d(inputs, weights, strides=[1, *conv_strides, 1], padding='SAME')

	if use_bias:
		bias = tf.Variable( tf.zeros([conv_num_outputs]) )
		conv = tf.nn.bias_add(conv, bias)

	conv = batch_norm(conv, is_training)
	conv = activation_fn(conv)

	return conv


def pool2d(inputs, pool_ksize=(2,2), pool_strides=(2,2)):
	"""
	:param inputs: TensorFlow Tensor
	:param pool_ksize: kernal size 2-D Tuple for pool
	:param pool_strides: Stride 2-D Tuple for pool
	: return: A tensor after max pooling of inputs
	"""

	pool = tf.nn.max_pool(inputs, ksize=[1, *pool_ksize, 1], strides=[1, *pool_strides, 1], padding='SAME')

	return pool


def flatten(tensor):
	"""
	Flatten tensor to (Batch Size, Flattened Image Size)
	:param tensor: A tensor
	: return: Flattened tensor
	"""

	shape = tensor.shape.as_list()[1:]
	tensor = tf.reshape(tensor, [-1, np.prod(shape)])
	return tensor


def fully_connected(inputs, num_outputs, activation_fn=None):
	"""
	Apply a fully connected layer to x_tensor using weight and bias
	:param inputs: A 2-D tensor where the first dimension is batch size.
	:param num_outputs: The number of output that the new tensor should be.
	: return: A 2-D tensor where the second dimension is num_outputs.
	"""

	n = int(inputs.shape[1])
	weights = tf.Variable( tf.truncated_normal([n, num_outputs], stddev=np.sqrt(2.0 / n)) )
	bias = tf.Variable( tf.zeros([num_outputs]) )

	fc = tf.add(tf.matmul(inputs, weights), bias)

	if activation_fn is not None:
		fc = activation_fn(fc)
	return fc


def baseline_model(inputs, num_classes, keep_prob, is_training):
	"""
	Build the feed forward convolutional neural network model
	:param x: Placeholder tensor that holds image data.
	:param num_classes: Number of classes.
	:param keep_prob: Placeholder tensor that hold dropout keep probability.
	: return: Tensor that represents logits
	"""

	conv1 = conv2d(inputs, 16, (3,3), (1,1), is_training)
	pool1 = pool2d(conv1)
	conv2 = conv2d(pool1, 32, (3,3), (1,1), is_training)
	pool2 = pool2d(conv2)
	conv3 = conv2d(pool2, 64, (3,3), (1,1), is_training)
	pool3 = pool2d(conv3)

	flat = flatten(pool3)

	drop1 = tf.nn.dropout(flat, keep_prob*2)
	fc1 = fully_connected(drop1, 256, activation_fn=tf.nn.relu)
	drop2 = tf.nn.dropout(fc1, keep_prob)

	logits = fully_connected(drop2, num_classes)

	return tf.identity(logits, name="logits")


def baseline_convnet(input_shape, num_classes):
	"""
	Build the convolutional neural network
	:param input_shape: Shape of input data.
	:param num_classes: Number of classes.
	"""

	# Create placeholders
	inputs, targets, is_training, keep_prob, lr = placeholders(input_shape, num_classes)

	# Create the model
	logits = baseline_model(inputs, num_classes, keep_prob, is_training)

	# Loss and optimizer
	loss = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(
					logits=logits, labels=targets) )
	optimizer = tf.train.AdamOptimizer().minimize(loss)

	# Accuracy
	correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(targets, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

	return (inputs, targets, is_training, keep_prob, lr,
			logits, loss, optimizer, accuracy)
