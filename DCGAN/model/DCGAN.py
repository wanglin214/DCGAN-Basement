# =============================================================================
# version 2025-04-08.
# =============================================================================
import torch
import torch.nn as nn
import math


# # Define the generator and discriminator
class Generator(nn.Module):
    def __init__(self, nz=200, ngf=64, nc=1):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.nz = nz
        self.nc = nc
        self.main = nn.Sequential(
            # Input noise vector Z, output (ngf*8) x 4 x 4 feature map
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # Input (ngf*8) x 4 x 4 feature map, output (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Input (ngf*4) x 8 x 8, output (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Input (ngf*2) x 16 x 16, output (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Input (ngf) x 32 x 32, output (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=1):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.main = nn.Sequential(
            # Input image size (nc) x 64 x 64, output (ndf) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (ndf) x 32 x 32, output (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (ndf*2) x 16 x 16, output (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (ndf*4) x 8 x 8, output (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Input (ndf*8) x 4 x 4, output 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Traverse the parameters of the generator and discriminator and apply random initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    x = torch.randn(64, 1, 64, 64)
    noise = torch.randn(64, 200, 1, 1)
    netG = Generator()
    netD = Discriminator()
    print(netG(noise).shape, netD(x).shape)
    print(netG.state_dict())
