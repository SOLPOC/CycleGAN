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
    input_B=Tensor(option.batchsize,option.out_channels,option.size,option.size)
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
    dataloader=DataLoader(ImageDataset(option=option,
                                       transform4img=transform4img),
                        batch_size=option.batchSize,
                          shuffle=True,num_workers=option.n_cpu)
    logger = Logger(option.n_epochs, len(dataloader))

    for epoch in range(option.epoch,option.n_epochs):
        for i,batch in enumerate(dataloader):
            real_A=Variable(input_A.copy_(batch['A']))
            real_B=Variable(input_B.copy_(batch['B']))

            optimizer_G.zero_grad()

            same_B=netG_A2B(real_B)
            loss_identity_B=criterion_identity(same_B,real_B)*5.0
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            fake_B=netG_A2B(real_A)
            pred_fake_B=netD_B(fake_B)
            loss_GAN_A2B=criterion_GAN(pred_fake_B,target_real)

            fake_A = netG_B2A(real_B)
            pred_fake_A = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real)

            cycle_A=netG_B2A(fake_B)
            loss_cycle_A=criterion_cycle(cycle_A,real_A)*10.0
            cycle_B = netG_A2B(fake_A)
            loss_cycle_B = criterion_cycle(cycle_B, real_B) * 10.0

            loss_G=loss_identity_A+loss_identity_B+loss_GAN_A2B+loss_GAN_B2A+loss_cycle_A+loss_cycle_B
            loss_G.backward()
            optimizer_G.step()

            optimizer_D_A.zero_grad()

            #real loss
            pred_real=netD_A(real_A)
            loss_D_real=criterion_GAN(pred_real,target_real)
            #fake_loss
            fake_A=fake_A_buffer.push_and_pop(fake_A)
            pred_fake=netD_A(fake_A.detach())
            loss_D_fake=criterion_GAN(pred_fake,target_fake)

            loss_D_A=(loss_D_fake+loss_D_real)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()

            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'output/netD_A.pth')
        torch.save(netD_B.state_dict(), 'output/netD_B.pth')


if __name__ == "__main__":
    train()