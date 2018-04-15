from __future__ import print_function
import argparse
import os
import os.path
import random
import csv
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable


class ColorBWDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.greyImgsPath = []
        self.colorImgsPath = []
        #self.transform = transforms.ToTensor()
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        dir = os.path.expanduser(root_dir)
        colorDir = os.path.join(dir, "color")
        greyDir = os.path.join(dir, "grey")

        for root, _, fnames in sorted(os.walk(colorDir)):
            for fname in sorted(fnames):
                self.colorImgsPath.append(os.path.join(colorDir, fname))

        for root, _, fnames in sorted(os.walk(greyDir)):
            for fname in sorted(fnames):
                self.greyImgsPath.append(os.path.join(greyDir, fname))

        print("len(self.colorImgsPath) : ", len(self.colorImgsPath))
        print("len(self.greyImgsPath) : ", len(self.greyImgsPath))

    def __len__(self):
        return len(self.colorImgsPath)

    def __getitem__(self, idx):
        colorImage = Image.open(self.colorImgsPath[idx]).convert('RGB')
        greyImage  = Image.open(self.greyImgsPath[idx])
        colorImage = self.transform(colorImage)
        greyImage  = self.transform(greyImage)
        return (colorImage, greyImage)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

# Holds console output
logs = []
with open('log.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow('New experiment')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


dataset = ColorBWDataset(opt.dataroot)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = 1
ngf = int(opt.ngf)
ndf = 64
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        # input is (nc) x 96 x 160
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, ndf, 5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (ndf) x 48 x 80
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (ndf*2) x 24 x 40
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (ndf*4) x 12 x 20
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # state size. (ndf*8) x 6 x 10
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Deconvolution Layers
        # state size. (ndf*8) x 4 x 8
        self.reverseLayer5 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 8, ndf * 8, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True)
        )
        # state size. (ndf*8) x 6 x 10
        self.reverseLayer4 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 16, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True)
        )
        # state size. (ndf*4) x 12 x 20
        self.reverseLayer3 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 8, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True)
        )
        # state size. (ndf*2) x 16 x 16
        self.reverseLayer2 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 4, ndf, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True)
        )
        # state size. (ndf) x 32 x 32
        self.reverseLayer1 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 2, nc, 4, stride=2, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        #     output = self.main(input)
        resLayer1 = self.layer1(input)
        # print("L1 : ",resLayer1.shape)
        resLayer2 = self.layer2(resLayer1)
        # print("L2 : ",resLayer2.shape)
        resLayer3 = self.layer3(resLayer2)
        # print("L3 : ",resLayer3.shape)
        resLayer4 = self.layer4(resLayer3)
        # print("L4 : ",resLayer4.shape)
        resLayer5 = self.layer5(resLayer4)
        # print("L5 : ",resLayer5.shape)
        resReverseLayer5 = self.reverseLayer5(resLayer5)
        # print("RL6 : ",resReverseLayer5.shape)
        resReverseLayer4 = Variable(torch.zeros_like(resLayer3.data))
        resReverseLayer4[:,:,0:-1,:] = self.reverseLayer4(torch.cat((resLayer4,resReverseLayer5), 1))
        # print("RL7 : ",resReverseLayer4.shape)
        resReverseLayer3 = self.reverseLayer3(torch.cat((resLayer3,resReverseLayer4), 1))
        # print("RL8 : ",resReverseLayer3.shape) 
        resReverseLayer2 = self.reverseLayer2(torch.cat((resLayer2,resReverseLayer3), 1))
        # print("RL9 : ",resReverseLayer2.shape)
        output = self.reverseLayer1(torch.cat((resLayer1,resReverseLayer2), 1))
        # print("RL10 : ",output.shape)
        return output


netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.colorPre = nn.Sequential(
            # input is (nc) x 96 x 160
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.greyscalePre = nn.Sequential(
            # input is (nc) x 96 x 160
            nn.Conv2d(1, ndf, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.main = nn.Sequential(
            # state size. (ndf) x 48 x 80
            nn.Conv2d(ndf*2, ndf * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 24 x 40
            nn.Conv2d(ndf * 2, ndf * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 12 x 20
            nn.Conv2d(ndf * 4, ndf * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 6 x 10
            nn.Conv2d(ndf * 8, 1, 3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 1 x 4 x 8
            nn.Conv2d(1,1,(14,12),stride=1,padding=0),
            nn.Sigmoid()
        )

    def forward(self, greyscale, color):
        greyscale = self.greyscalePre(greyscale)
        color = self.colorPre(color)
        input = torch.cat((greyscale, color),1)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        # print(output.shape)
        return output.view(-1, 1).squeeze(1)


netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.L1Loss()

greyscale = torch.FloatTensor(opt.batchSize, 1, 218, 178)
color = torch.FloatTensor(opt.batchSize, 3, 218, 178)
label = torch.FloatTensor(opt.batchSize)


real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    greyscale, color, label = greyscale.cuda(), color.cuda(), label.cuda()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
loss_G = 1
for epoch in range(opt.niter):
    for i, (colorImg,greyscaleImg) in enumerate(dataloader, 0):

        if i == 157:
            break
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        batch_size = greyscaleImg.size(0)
        if opt.cuda:
            greyscaleImg = greyscaleImg.cuda()
            colorImg = colorImg.cuda()
        greyscale.resize_as_(greyscaleImg).copy_(greyscaleImg)
        color.resize_as_(colorImg).copy_(colorImg)
        label.resize_(batch_size).fill_(real_label)
        #label.resize_(batch_size).random_(0.7, 1.0)
        greyscaleVar = Variable(greyscale)
        colorVar = Variable(color)
        labelv = Variable(torch.rand(label.size())*0.2+0.8).cuda()

        output = netD(greyscaleVar, colorVar)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        fake = netG(greyscaleVar.detach())
        labelv = Variable(torch.rand(label.size())*0.0).cuda()
        output = netD(greyscaleVar.detach(),fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
   
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if epoch == 0:
            if i == 0:
                runGen = True
            elif i > 100:
                runGen = True
            else:
                runGen = False
        elif i % 2 == 0:
            runGen = True
        else:
            runGen = False

        if runGen:
            netG.zero_grad()
            labelv = Variable(torch.rand(label.size())*0.2+0.8).cuda()  # fake labels are real for generator cost
            output = netD(greyscaleVar.detach(), fake.detach())
            errG = criterion(output, labelv)
            # GAN + L1 (doesn't work)
            # errG += criterion(fake, colorVar)
            errG.backward()
            D_G_z2 = output.data.mean()
            loss_G = errG.data[0]
            optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

        with open('log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, i, errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2])
        if i % 100 == 0:
            vutils.save_image(color,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(greyscaleVar)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))