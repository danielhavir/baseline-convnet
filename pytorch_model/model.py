import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaselineModel(nn.Module):
	def __init__(self, num_classes, keep_probability):
		super(BaselineModel, self).__init__()
		self.keep_prob = (1 - keep_probability)
		self.conv1 = nn.Conv2d(3, 16, 3, bias=False, padding=1)
		self.bn1 = nn.BatchNorm2d(16)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(16, 32, 3, bias=False, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, 3, bias=False, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.drop1 = nn.Dropout(p=self.keep_prob*(3/2))
		self.fc1 = nn.Linear(64 * 4**2, 128)
		self.drop2 = nn.Dropout(p=self.keep_prob)
		self.fc2 = nn.Linear(128, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, x):
		# Layer 1
		out = self.conv1(x)
		out = F.relu(self.bn1(out))
		out = self.pool(out)
		# Layer 2
		out = self.conv2(out)
		out = F.relu(self.bn2(out))
		out = self.pool(out)
		# Layer 3
		out = self.conv3(out)
		out = F.relu(self.bn3(out))
		out = self.pool(out)
		# Flatten
		out = out.view(out.size(0), -1)
		# Fully connected
		out = self.drop1(out)
		out = self.fc1(out)
		out = F.relu(out)
		out = self.drop2(out)
		out = self.fc2(out)

		return out
