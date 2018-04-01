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

data = OurDataset("./dataset/", "./dataset_target/") 

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
		self.optimizer = optim.SGD(self.parameters(), lr=0.01)
		
	def forward(self, image):
		batch_size = image.size()[0]
		x = image.view(batch_size, -1)
		for i in xrange(self.length-1):
			x = F.sigmoid(self.layers[i](x))
		x = F.log_softmax(self.layers[self.length-1](x), dim=0)
		return x

batch_size = 16
RdN = MLP((320 * 240, 512, 512, 320 * 240 * 3))
RdN.cuda()
greyscale = torch.FloatTensor(batch_size, 1, 320, 240)
greyscale =greyscale.cuda()
color = torch.FloatTensor(batch_size, 3, 320, 240)
color = color.cuda()
criterion = nn.BCELoss()
#criterion.cuda()



train_losses = 0.
valid_losses = 0.
test_losses = 0.

for epoch in range(100):
	train_losses = 0.
	for i, (greyscaleImg,colorImg) in enumerate(train_loader, 0):
		RdN.zero_grad()
		#greyscaleImg = greyscaleImg.cuda()
		#colorImg = colorImg.cuda()
		greyscale.resize_as_(greyscaleImg).copy_(greyscaleImg)
		color.resize_as_(colorImg).copy_(colorImg)
		greyscaleVar = Variable(greyscale)
		colorVar = Variable(color/255.)
		#print("greyscaleVar", greyscaleVar.shape)
		#print("colorVar", colorVar.shape)
		output = RdN.forward(greyscaleVar)
		loss = criterion(output, colorVar)
		loss.backward()
		RdN.optimizer.step()
		train_losses += loss
		
	for i, (greyscaleImg,colorImg) in enumerate(valid_loader, 0):
		greyscaleVar = Variable(greyscale)
		colorVar = Variable(color/255.)
		output = RdN.forward(greyscaleVar)
		loss = criterion(output, colorVar)
		valid_losses += loss
	
	print ("train set", epoch,"train err:", train_losses/(total * (1-2 * split)), " valid err:", valid_losses/(total * split))

	
for i, (greyscaleImg,colorImg) in enumerate(test_loader, 0):
		output = RdN.forward(greyscaleImg)
		loss = criterion(output, colorImg/255.)
		test_losses += loss
	
print ("test set", epoch,"test err:", test_losses/(total * split))

import random
for i  in range(10):
	greyscaleImg, colorImg = random.choice(data)
	greyscaleVar = Variable(greyscale)
	colorVar = Variable(color/255.)
	output = RdN.forward(greyscaleVar)
	img = Image.fromarray(np.uint8(output))
	img.save( "test" + str(i) + ".jpg")









