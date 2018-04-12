import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import torchvision.utils as vutils


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
	
	croped =  croped[resize[0]:resize[2]][resize[1]:resize[3]]
	
	result = cv2.resize(croped,(ideal_width, ideal_height), interpolation = cv2.INTER_CUBIC)
	
	return result
class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.ndf = 64
        self.nc = 3
     
        # input is (nc) x 96 x 160
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.ndf, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (ndf) x 48 x 80
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (ndf*2) x 24 x 40
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (ndf*4) x 12 x 20
        self.layer4 = nn.Sequential(
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (ndf*8) x 6 x 10
        self.layer5 = nn.Sequential(
            nn.Conv2d(self.ndf * 8, self.ndf * 8, 3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        
        # Deconvolution Layers
        # state size. (ndf*8) x 4 x 8
        self.reverseLayer5 = nn.Sequential(
            nn.ConvTranspose2d(self.ndf * 8, self.ndf * 8, 3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.ReLU(True)
        )
        # state size. (ndf*8) x 6 x 10
        self.reverseLayer4 = nn.Sequential(
            nn.ConvTranspose2d(self.ndf * 16, self.ndf * 4, (4, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.ReLU(True)
        )
        # state size. (ndf*4) x 12 x 20
        self.reverseLayer3 = nn.Sequential(
            nn.ConvTranspose2d(self.ndf * 8, self.ndf * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.ReLU(True)
        )
        # state size. (ndf*2) x 16 x 16
        self.reverseLayer2 = nn.Sequential(
            nn.ConvTranspose2d(self.ndf * 4, self.ndf, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.ReLU(True)
        )
        # state size. (ndf) x 32 x 32
        self.reverseLayer1 = nn.Sequential(
            nn.ConvTranspose2d(self.ndf * 2, self.nc, 3, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        resLayer1 = self.layer1(input)
        resLayer2 = self.layer2(resLayer1)
        resLayer3 = self.layer3(resLayer2)
        resLayer4 = self.layer4(resLayer3)
        resLayer5 = self.layer5(resLayer4)
        resReverseLayer5 = self.reverseLayer5(resLayer5)
        resReverseLayer4 = self.reverseLayer4(torch.cat((resLayer4,resReverseLayer5), 1))
        resReverseLayer3 = self.reverseLayer3(torch.cat((resLayer3,resReverseLayer4), 1))
        resReverseLayer2 = self.reverseLayer2(torch.cat((resLayer2,resReverseLayer3), 1))
        output = self.reverseLayer1(torch.cat((resLayer1,resReverseLayer2), 1))
        return output

def take_picture(imgDim):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        w, h = img.shape[:2]
        img = img[:imgDim[1], int(h/2-imgDim[0]/2):int(h/2+imgDim[0]/2)]
        img = cv2.flip(img, 1)
		
        bigImg = resize(img, (178, 218))
        cv2.imshow('webcam', bigImg)
        if cv2.waitKey(1) == 32: # space bar to take picture
            cv2.destroyAllWindows()
            return img 

imgDim = (178, 218)
img = take_picture(imgDim)#cv2.imread("000476.jpg")#take_picture(imgDim)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

model = _netG()
model.load_state_dict(torch.load('netG_epoch_3.pth',map_location='cpu'))

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
loader = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])
greyscale = torch.FloatTensor(1, 1, 218, 178)

pil_im = Image.fromarray(gray)
grayImage = loader(pil_im).float()
grayImage = grayImage.unsqueeze(0)

print("GrayImg shape", grayImage.size())

greyscale.resize_as_(grayImage).copy_(grayImage)
greyscaleVar = Variable(greyscale)

print("Generating fake image ...")
fake = model(greyscaleVar.detach())
print("... Done generating fake image")

fake = cv2.bilateralFilter(fake,9,75,75)

vutils.save_image(fake.data,'img.png',normalize=True)

fakeImg = cv2.imread("img.png")
grey_3_channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

images = np.hstack((img, grey_3_channel, fakeImg[2:-1, 2:-1]))
bigImages = cv2.resize(images, (0,0), fx=2, fy=2) 
cv2.imshow('result', bigImages)
cv2.waitKey() 


