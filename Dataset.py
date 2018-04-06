###################################################
##	gatherDataset.py : utility script for loading a dataset and splitting it into train, valid and test sets
##	@author Luc Courbariaux 2018
###################################################

import os
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import numpy as np
from torchvision import transforms

# takes images from a dataset path, and another to the target path
# assumes that data and target photos have the same name
class CustomDataset(Dataset):

	def __init__(self, datapath, targetpath, transform = None):
		self.datapath = datapath
		self.targetpath = targetpath
		self.transform = transform
		self.toTensor = transforms.ToTensor()
		self.images= os.listdir(datapath)

	def __getitem__(self, index):
		data = Image.open(self.datapath + self.images[index])
		data = data.convert('1') # the input data is in hues of grey, ie 1 byte per pixel
		
		target = Image.open(self.targetpath + self.images[index])
		target = target.convert('RGB') # the target data is in RGB, ie 3 byte per pixel
		
		data = self.toTensor(np.array(data).astype("float"))
		target = self.toTensor(np.array(target).astype("float"))
		
		if self.transform is not None:
			data = self.transform(data)
			target = self.transform(target)
			
		return target, data

	def __len__(self):
		return len(self.images)


def generate(batch_size = 32, split = 0.1, datapath = "./dataset/", targetpath = "./dataset_target/"):
	data = CustomDataset(datapath, targetpath) 

	# loading based on https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
	# creates  a list of indices
	total = len(data)
	indices = list(range(total))

	# shuffles it
	np.random.seed(50676)
	np.random.shuffle(indices)

	# splits it : major part is train, 10% is valid, 10% test
	train_idx, test_idx, valid_idx = indices[:int(total * (1-2*split))], indices[int(total * (1-2*split)):int(total * (1-split))], indices[int(total * (1-split)):]

	# sampler objects that use those indices
	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)
	test_sampler = SubsetRandomSampler(test_idx)

	# the loaders that will be used by the neural network
	train_loader = DataLoader( data, batch_size=batch_size, sampler=train_sampler)
	valid_loader = DataLoader( data, batch_size=batch_size, sampler=valid_sampler)
	test_loader = DataLoader( data, batch_size=batch_size, sampler=test_sampler)
	
	return train_loader, valid_loader, test_loader

# use case : 
# for i, (greyscaleImg,colorImg) in enumerate(train_loader, 0): 
#		<...>







