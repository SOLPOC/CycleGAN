import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,in_channels=3,n_layers=3):
        """
        Get a discriminator of N layers
            The structure is same as the paper.
        :param in_channels: the number of channels in input images
        :param n_layers: the number of conv layers in the discriminator
        :return:
        """
        super().__init__()
        # build a model of 5 layers
        model = [ nn.Conv2d(in_channels=in_channels,out_channels= 64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True) ]

        model += [nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1),
                    nn.InstanceNorm2d(num_features=128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1),
                    nn.InstanceNorm2d(num_features=256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, stride=1,padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [nn.Conv2d(512,1, 4, stride=1, padding=1)]
        self.model=nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

def test():
    x = torch.randn((5, 3, 256, 256))
    print(x)
    model = Discriminator(3,3)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()