import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image


# to install pytorch in temp with no admin right
# pip install --install-option="--prefix=/tmp/" --user http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
# pip install --install-option="--prefix=/tmp/" torchvision
	
class OurDataset(Dataset):

	def __init__(self, datapath, targetpath, transform = None):
		self.datapath = datapath
		self.targetpath = targetpath
		self.transform = transform
		self.toTensor = transforms.ToTensor()
		self.images= os.listdir(datapath)

	def __getitem__(self, index):
		data = Image.open(self.datapath + self.images[index])
		data = data.convert('1')
		if self.transform is not None:
			img = self.transform(img)
		target = Image.open(self.targetpath + self.images[index])
		target = target.convert('RGB')
		return self.toTensor(np.array(data).flatten().astype("float")), self.toTensor(np.array(target).flatten().astype("float"))

	def __len__(self):
		return len(self.images)

# loading based on https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

data = OurDataset("./data/input/", "./data/target/") 

batch_size = 1
split = 0.1

total = len(data)
indices = list(range(total))

np.random.seed(50676)
np.random.shuffle(indices)

train_idx, test_idx, valid_idx = indices[:int(total * (1-2*split))], indices[int(total * (1-2*split)):int(total * (1-split))], indices[int(total * (1-split)):]


train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = torch.utils.data.DataLoader( data, batch_size=batch_size, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader( data, batch_size=batch_size, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader( data, batch_size=batch_size, sampler=test_sampler)


class MLP(nn.Module):
	def __init__(self, dimensions):
		super(MLP, self).__init__()
		self.layers =  nn.ModuleList()
		self.length = len(dimensions)-1
		for i in xrange(self.length):
			self.layers.append( nn.Linear(dimensions[i], dimensions[i+1]) )
		self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
		
	def forward(self, image):
		batch_size = image.size()[0]
		x = image.view(batch_size, -1)
		for i in xrange(self.length-1):
			x = F.sigmoid(self.layers[i](x))
		x = F.log_softmax(self.layers[self.length-1](x), dim=0)
		return x



def train(model):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		print(target.shape)
		data, target = Variable(data.cuda()), Variable(target.cuda())
		model.optimizer.zero_grad()
		output = model(data)
		loss = F.mse_loss(output, target)
		loss.backward()
		model.optimizer.step()


def test(model, loader, name):
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in loader:
		data, target = data.cuda(), target.cuda()
		output = model(data)
		test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum u
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max l
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	test_loss /= 10000
	print (name , "set : Average loss:", test_loss, " Accuracy:", correct, "/", 10000, "=", 100. * correct / 10000, "%")
	return test_loss, 100. * correct / 10000


learning_rate = 0.05
architectures = [[320 * 240, 512, 512, 320 * 240 * 3]]

# redirect print to log file : https://stackoverflow.com/questions/2513479
#old_stdout = sys.stdout
#log_file = open("experiences.log","w")
#sys.stdout = log_file



for arc in architectures:
	epochs = 350
	print(arc)
	
	RdN = MLP(arc)
	RdN.cuda()
	
	losses =[]
	
	for epoch in range(1, epochs + 1):
		train(RdN)
		loss, accuracy = test(RdN, valid_loader, 'valid')
		losses.append(loss)
	loss, accuracy = test(RdN, test_loader, 'test')
	
	plt.title("Architecture : " + str(arc) + "  & Learning rate : " + str(learning_rate) + " (accuracy : " + str(accuracy)[:4] + ") ")
	plt.ylabel("Average negative log likelihood")
	plt.xlabel("Epoch")
	plt.plot(losses, label="validation")
	plt.legend()
	plt.savefig(str(arc) + ".png") #plt.show()
	plt.close()

log_file.close()







