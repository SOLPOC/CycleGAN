import os

from option import Option
import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import time

from model.generator import ResnetGenerator
from model.discriminator import  Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from dataset import ImageDataset


def train():
    option=Option().get_option("train")

    netG_A2B=ResnetGenerator(option.in_channels,option.out_channels,n_residual_blocks=9)
    netG_B2A=ResnetGenerator(option.out_channels,option.in_channels,n_residual_blocks=9)
    netD_A=Discriminator(option.in_channels)
    netD_B=Discriminator(option.out_channels)

    networks=[netG_A2B,netG_B2A,netD_A,netD_B]
    if(option.gpu):
       for network in networks:
           network.cuda()

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    criterion_GAN=torch.nn.MSELoss()
    criterion_cycle=torch.nn.L1Loss()
    criterion_identity=torch.nn.L1Loss()

    optimizer_G=torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                 lr=option.lr,betas=(0.5,0.999))
    optimizer_D_A=torch.optim.Adam(netD_A.parameters(),lr=option.lr,betas=(0.5,0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=option.lr, betas=(0.5, 0.999))

    lr_scheduler_G=torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=LambdaLR(option.n_epochs, option.epoch_start, option.epoch_decay_start).step)
    lr_scheduler_D_A=torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,lr_lambda=LambdaLR(option.n_epochs,  option.epoch_start, option.epoch_decay_start).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(option.n_epochs,  option.epoch_start,option.epoch_decay_start).step)

    Tensor=torch.cuda.FloatTensor  if option.gpu else torch.Tensor
    input_A=Tensor(option.batch_size,option.in_channels,option.crop_size,option.crop_size)
    input_B=Tensor(option.batch_size,option.out_channels,option.crop_size,option.crop_size)
    target_real=Variable(Tensor(option.batch_size).fill_(1.0),requires_grad=False)
    target_fake = Variable(Tensor(option.batch_size).fill_(0.0), requires_grad=False)

    fake_A_buffer=ReplayBuffer()
    fake_B_buffer=ReplayBuffer()

    transform4img=[
        transforms.Resize(option.load_size,Image.BICUBIC),
        transforms.RandomCrop(option.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]

    # dataset_size = len(dataset)    # get the number of images in the dataset.
    # print('The number of training images = %d' % dataset_size)

    dataloader=DataLoader(ImageDataset(option=option,
                                       transform4img=transform4img),
                        batch_size=option.batch_size,
                          shuffle=True,num_workers=option.n_threads)
    # logger = Logger(option.n_epochs, len(dataloader))

    total_iters = 0  # the total number of training iterations

    for epoch in range(option.epoch_start,option.n_epochs):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        for i,batch in enumerate(dataloader):
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % option.print_freq == 0:
                time_compute= iter_start_time - iter_data_time

            # if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            #     t_comp = (time.time() - iter_start_time) / opt.batch_size
            #
            # if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            #     model.save_networks(save_suffix)



            total_iters += option.batch_size
            epoch_iter += option.batch_size

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


        if epoch % option.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            save_model_dir=option.checkpoints_dir+"/"+option.name+"/"+"epoch_"+str(epoch)+"/"
            os.mkdir(save_model_dir)
            torch.save(netG_A2B.state_dict(), save_model_dir+'netG_A2B.pth')
            torch.save(netG_B2A.state_dict(), save_model_dir+'netG_B2A.pth')
            torch.save(netD_A.state_dict(), save_model_dir+ 'netD_A.pth')
            torch.save(netD_B.state_dict(),  save_model_dir+'netD_B.pth')

def load_networks(self, epoch):
    """Load all the networks from the disk.

    Parameters:
        epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    """
    for name in self.model_names:
        if isinstance(name, str):
            load_filename = '%s_net_%s.pth' % (epoch, name)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            # if you are using PyTorch newer than 0.4 (e.g., built from
            # GitHub source), you can remove str() on self.device
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            # patch InstanceNorm checkpoints prior to 0.4
            for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
            net.load_state_dict(state_dict)

def print_networks(self, verbose):
    """Print the total number of parameters in the network and (if verbose) network architecture

    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    print('---------- Networks initialized -------------')
    for name in self.model_names:
        if isinstance(name, str):
            net = getattr(self, 'net' + name)
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
    print('-----------------------------------------------')

if __name__ == "__main__":
    train()