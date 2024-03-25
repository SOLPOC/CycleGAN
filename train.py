from option import Option
import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch


from model.generator import ResnetGenerator
from model.discriminator import  Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from dataset import ImageDataset

def train():
    option=Option().get_parser("train")
    # dataset = create_dataset(opt)  # create a dataset given option.dataset_mode and other options
    # dataset_size = len(dataset)  # get the number of images in the dataset.
    netG_A2B=ResnetGenerator(option.in_channels,option.out_channels)
    netG_B2A=ResnetGenerator(option.out_channels,option.in_channels)
    netD_A=Discriminator(option.in_channels)
    netD_B=Discriminator(option.out_channels)
    if(option.cuda):
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    criterion_GAN=torch.nn.MSELoss()
    criterion_cycle=torch.nn.L1loss()
    criterion_identity=torch.nn.L1loss()

    optimizer_G=torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                 lr=option.lr,betas=(0.5,0.999))
    optimizer_D_A=torch.optim.Adam(netD_A.parameters(),lr=option.lr,betas=(0.5,0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=option.lr, betas=(0.5, 0.999))

    lr_scheduler_G=torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=LambdaLR(option.n_epochs, option.epoch, option.decay_epoch).step)
    lr_scheduler_D_A=torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,lr_lambda=LambdaLR(option.n_epochs, option.epoch, option.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(option.n_epochs, option.epoch,option.decay_epoch).step)

    Tensor=torch.cuda.FloatTensor  if option.cuda else torch.Tensor
    input_A=Tensor(option.batchsize,option.in_channels,option.size,option.size)
    input_A=Tensor(option.batchsize,option.out_channels,option.size,option.size)
    target_real=Variable(Tensor(option.batchsize).fill_(1.0),requires_grad=False)
    target_fake = Variable(Tensor(option.batchSize).fill_(0.0), requires_grad=False)

    fake_A_buffer=ReplayBuffer()
    fake_B_buffer=ReplayBuffer()

    transform4img=[
        transforms.Resize(286,Image.BICUBIC),
        transforms.RandomCrop(option.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]
    dataloader=DataLoader(ImageDataset)


if __name__ == "__main__":
    train()