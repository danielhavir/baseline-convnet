import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
from datetime import datetime
from pytorch_model.model import BaselineModel

SAVE_MODEL_PATH = 'pytorch_model/checkpoints/'

class BaselineConvnet:
	def __init__(self, batch_size, keep_probability, learning_rate):
		self.use_gpu = torch.cuda.is_available()

		self.transform = transforms.Compose(
			[transforms.ToTensor(),
			 transforms.Normalize([0.49136144,  0.48213023,  0.44652155], [0.2470296 ,  0.24346825,  0.26157242] )
		])

		self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
		self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=2)

		self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
		self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4, shuffle=False, num_workers=2)

		self.net = BaselineModel(10, keep_probability)
		if self.use_gpu:
			self.net = self.net.cuda()

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
		print(8*'#', 'Model built'.upper(), 8*'#')

	def run(self, epochs, classes):
		print(8*'#', 'Run started'.upper(), 8*'#')
		for epoch in range(epochs):
			print('Training...')
			self.net.train()

			train_loss = 0.0
			correct = 0; total = 0
			for i, (inputs, labels) in enumerate(self.trainloader):
				if self.use_gpu:
					inputs, labels = inputs.cuda(), labels.cuda()

				inputs, labels = Variable(inputs), Variable(labels)

				self.optimizer.zero_grad()

				outputs = self.net(inputs)
				loss = self.criterion(outputs, labels)
				loss.backward()
				self.optimizer.step()

				train_loss += loss.data[0]

				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels.data).sum()
				train_accuracy = float(correct) / float(total)

				print('Epoch {}, Batch {}, Loss {}, Accuracy {}'.format(epoch+1, i+1, train_loss/(i+1), train_accuracy))


			print('Evaluating...')
			self.net.eval()

			test_loss = 0.0
			correct = 0; total = 0
			for i, (inputs, labels) in enumerate(self.testloader):
				if self.use_gpu:
					inputs, labels = inputs.cuda(), labels.cuda()

				inputs, labels = Variable(inputs), Variable(labels)

				outputs = self.net(inputs)
				loss = self.criterion(outputs, labels)

				test_loss += loss.data[0]

				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels.data).sum()
				test_accuracy = float(correct) / float(total)

			print('Epoch {}, Test Loss {}, Test Accuracy {}'.format(epoch+1, test_loss/(i+1), test_accuracy))

		now = str(datetime.now()).replace(" ", "-")
		torch.save(self.net, os.path.join(SAVE_MODEL_PATH, '{}.pth.tar'.format(now)))

		class_correct = list(0. for i in range(10))
		class_total = list(0. for i in range(10))
		for inputs, labels in self.testloader:
			if self.use_gpu:
				inputs, labels = inputs.cuda(), labels.cuda()

			outputs = self.net(Variable(inputs))
			_, predicted = torch.max(outputs.data, 1)
			c = (predicted == labels).squeeze()
			for i in range(4):
				label = labels[i]
				class_correct[label] += c[i]
				class_total[label] += 1


		for i in range(10):
			print('Accuracy of %5s : %2d %%' % (
				classes[i], 100 * class_correct[i] / class_total[i]))

