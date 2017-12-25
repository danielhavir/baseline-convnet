import tensorflow as tf
import numpy as np
import pickle
import os
from datetime import datetime
import tensorflow_model.model as tf_model

CIFAR10_DATASET_FOLDER_PATH = os.path.join('data', 'cifar-10-batches-py')
SAVE_MODEL_PATH = 'tensorflow_model/checkpoints/'

MEAN = np.array([125.306918046875, 122.950394140625, 113.86538318359375])
STD = np.array([62.99321927813685, 62.088707640014405, 66.70489964063101])

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def gen_batches(features, labels, batch_size: int):
	for start in range(0, features.shape[0], batch_size):
		end = min(start+batch_size, features.shape[0])
		yield features[start:end], labels[start:end]


def cifar_loader(train: bool, batch_id=None):
	if train:
		# Cifar comes with 5 data_batches
		file_path = 'data_batch_{}'.format(batch_id)
	else:
		file_path = 'test_batch'
	data_batch = unpickle(os.path.join(CIFAR10_DATASET_FOLDER_PATH, file_path))
	features = data_batch[b'data'].reshape((len(data_batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
	# One-hot encode labels
	labels = np.eye(10)[data_batch[b'labels']]
	# Normalize features
	features = (features - MEAN) / STD
	return features, labels


def run(epochs, batch_size, learning_rate, keep_probability):
	print(8*'#', 'Run started'.upper(), 8*'#')
	(inputs, targets, is_training, keep_prob, lr,
	 logits, loss, optimizer, accuracy) = tf_model.baseline_convnet((32, 32, 3), 10)
	print(8*'#', 'Model built'.upper(), 8*'#')

	with tf.Session() as sess:
		# Initializing the variables
		sess.run(tf.global_variables_initializer())

		# Training cycle
		for epoch in range(epochs):
			print('Training...')

			train_loss = 0; train_accuracy = 0
			# Loop over all batches
			i = 1
			for batch_id in range(1,6):
				features, labels = cifar_loader(True, batch_id)
				for batch_features, batch_labels in gen_batches(features, labels, batch_size):
					feed_dict = {
						inputs: batch_features,
						targets: batch_labels,
						is_training: True,
						keep_prob: keep_probability,
						lr: learning_rate
					}
					_, batch_loss, batch_accuracy = sess.run([optimizer, loss, accuracy], feed_dict=feed_dict)
					train_loss += batch_loss
					train_accuracy += batch_accuracy
					print('Epoch {}, Batch {}, Loss {}, Accuracy {}'.format(epoch + 1, i, train_loss/i, train_accuracy/i))
					i+=1

			print('Evaluating..')
			test_loss = 0; test_accuracy = 0
			features, labels = cifar_loader(False)
			for i, (batch_features, batch_labels) in enumerate(gen_batches(features, labels, batch_size)):
				feed_dict = {
					inputs: batch_features,
					targets: batch_labels,
					is_training: False,
					keep_prob: 1.
				}
				batch_loss, batch_accuracy = sess.run([loss, accuracy], feed_dict=feed_dict)
				test_loss += batch_loss
				test_accuracy += batch_accuracy
			print('Epoch {}, Test Loss {}, Test Accuracy {}'.format(epoch+1, test_loss/i, test_accuracy/i))

		print(8*'#', 'Finished training'.upper(), 8*'#')


		# Save Model
		saver = tf.train.Saver()
		now = str(datetime.now()).replace(" ", "-")
		save_path = saver.save(sess, os.path.join(SAVE_MODEL_PATH, '{}.ckpt'.format(now)))
