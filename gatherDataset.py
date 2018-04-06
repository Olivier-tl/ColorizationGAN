###################################################
##	gatherDataset.py : utility script for creating a dataset of jpeg images set to same size
##	@author Luc Courbariaux 2018
###################################################

#import  shutil 
#import random
#import sys

import argparse
import os
from PIL import Image


def resize(image, size):
	# crops image to wanted ratio then resizes it
	# based on https://stackoverflow.com/questions/4744372/

	width  = image.size[0]
	height = image.size[1]
	#print("original size : ", image.size)

	aspect = width / float(height)

	ideal_width = size[0]
	ideal_height = size[1]

	ideal_aspect = ideal_width / float(ideal_height)

	if aspect > ideal_aspect:
		# Then crop the left and right edges:
		new_width = int(ideal_aspect * height)
		offset = (width - new_width) / 2
		resize = (offset, 0, width - offset, height)
	else:
		# ... crop the top and bottom:
		new_height = int(width / ideal_aspect)
		offset = (height - new_height) / 2
		resize = (0, offset, width, height - offset)
	
	croped =  image.crop(resize)
	#print("real crop : ", cr.size)
	
	#image.save("original" + str(resize) + ".jpg", "JPEG")
	#croped.save("sample" + str(resize) + ".jpg", "JPEG")
	
	result = croped.resize((ideal_width, ideal_height), Image.ANTIALIAS)
	#print("result size : ",  result.size)
	
	return result


parser = argparse.ArgumentParser(description='gets all images in INPUT folder, resizes them and puts them in OUTPUT directory.')
parser.add_argument('-i', '--input', nargs='?', default="./", help='folder from which the images')
parser.add_argument('-o', '--output', nargs='?', default="./dataset/", help='folder to store the images')
parser.add_argument('-s', '--size', nargs='?', default="320,240", help='size of output images, format \"width,height\"')
#parser.add_argument('-c', '--create', action='store_true', default=True, help='generates versions with a sobel filter applied')
args = parser.parse_args()
#print(args)


outputDirectory = args.output
if not outputDirectory.endswith("/"):
	outputDirectory += "/"

size = args.size.split(",")
size = (int(size[0]), int(size[1]))
#print(size)

try:
	os.mkdir(outputDirectory)
except:
	pass

# a 
paths = [args.input[:-1] if args.input.endswith("/") else args.input,]


#temp = 1.
while len(paths) !=0:
	
	path = paths.pop()
	
	for file in os.listdir(path):
		#if random.random() < temp:
			
			#temp *= 0.85
			#print(temp)
			#sys.stdout.flush()
			
			filePath = path + "/" + file
			#print(filePath)
			
			if file.endswith(".jpg") or  file.endswith(".png"):
				
				newPath = outputDirectory + path.replace(".","").replace("/","")  + file
				#print(newPath)
				
				#shutil.copy(filePath, newPath)
				img = Image.open(filePath)
				img = resize(img, size)
				#img.save(newPath[:-3] + "jpg", "JPEG")
				img.save(newPath[:-3] + "bmp", "BMP")
			
			else:
				
				if  os.path.isdir(filePath):
					#print("exploring " + filePath)
					paths.append(filePath)


print ( "got a dataset of " + str(len(os.listdir(outputDirectory))))