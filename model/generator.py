import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        """
        Get residual block
        :param in_features: the number of input and output channels
        """
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
class UnetGenerator(nn.Module):
    pass

class ResnetGenerator(nn.Module):
    def __init__(self,in_channels,out_channels,n_residual_blocks):
        """
        Get a generator of resnet
        :param in_channels:
        :param out_channels:
        :param n_filters:
        :param n_residual_blocks: the number of resnet blocks
        """
        super().__init__()
        # initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels,64,7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(True)]
        # add downsampling layers
        in_features = 64
        out_features = in_features*2
        for i in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(True) ]
            in_features = out_features
            out_features = in_features*2
        # add resnet blocks
        for i in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        # add upsampling layers
        out_features = in_features // 2
        for i in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        # add output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64,out_channels, 7),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = ResnetGenerator(in_channels=3,out_channels=3,n_filters=64,n_residual_blocks=6)
    print(gen(x).shape)


if __name__ == "__main__":
    test()
