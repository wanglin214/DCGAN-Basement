import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler
import torch.nn.functional as fun
# from torch.cuda.amp import autocast, GradScaler  # Modules needed for mixed precision training
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from utils import dataload, outGrd
from model import DCGAN
import os
from visdom import Visdom

# Set a random seed for reproducibility
manualSeed = 0
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.benchmark = True

# Basic parameter configuration
# Dataset path
dataroot = "data/real"
# Batch size
batch_size = 128
# Image size
image_size = 64
# Number of channels in the images, output channels of generator and input channels of discriminator
nc = 1
# Size of z latent vector (noise dimension)
nz = 200
# Size of feature maps in generator (base filter count)
ngf = 64
# Size of feature maps in discriminator (base filter count)
ndf = 64
# Loss function
criterion = nn.BCELoss()
# Real and fake labels
real_label = 1.0
fake_label = 0.0
# Number of GPUs to use
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# Create generator and discriminator
netG = DCGAN.Generator().to(device).apply(DCGAN.weights_init)
netD = DCGAN.Discriminator().to(device).apply(DCGAN.weights_init)
# Load data
dataloader = DataLoader(dataload.Basement(dataroot),
                        batch_size=batch_size,
                        shuffle=True,
                        # drop_last=True,
                        pin_memory=True,
                        prefetch_factor=4,
                        num_workers=8)  # Load test data

# Main program
if __name__ == '__main__':
    lr = 1e-3
    beta1 = 0.5
    # scaler = GradScaler()
    # Optimizers for G and D, using Adam
    # Adam learning rate and momentum parameters
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    schedulerD = lr_scheduler.CosineAnnealingLR(optimizerD, T_max=400, eta_min=1e-4, last_epoch=-1)
    schedulerG = lr_scheduler.CosineAnnealingLR(optimizerG, T_max=400, eta_min=1e-8, last_epoch=-1)
    # Loss tracking variables
    G_losses = []
    D_losses = []
    # Initialize Visdom for loss monitoring
    viz_G = Visdom()
    viz_G.line([0.], [0.], win='Generator_loss', opts=dict(title='Generator_loss'))
    viz_D = Visdom()
    viz_D.line([0.], [0.], win='Discriminator_loss', opts=dict(title='Discriminator_loss'))
    # Total epochs
    num_epochs = 800
    # Fixed noise for testing model throughout training
    rnoise = torch.randn(1, nz, 1, 1, device=device)
    # Create directory for model checkpoints
    if not os.path.exists('models'):
        os.mkdir('models')
        print("Starting Training Loop...")

    for epoch in range(num_epochs):
        lossG = 0.0
        lossD = 0.0
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            # print(data.shape)
            # os.system('pause')

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            # Train with real data
            netD.zero_grad()
            real_data = data.to(device)
            label = torch.full((batch_size,), real_label, device=device)
            output = netD(real_data).view(-1)
            # Calculate loss for real images, backpropagate - optimize D to classify real images as real (label 1)
            errD_real = criterion(output, label)
            errD_real.backward()
            # D_x = output.mean().item()

            # Train with fake data
            # Generate latent vectors
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # Generate fake images with G
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            # Calculate loss for fake images, backpropagate - optimize D to classify fake images as fake (label 0)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            # D_G_z1 = output.mean().item()
            # Combine errors and update parameters
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z))) - train G to fool D into classifying fake images as real
            ############################
            netG.zero_grad()
            label.fill_(real_label)  # Set target labels for fake images to real
            # Discriminate fake images again
            output = netD(fake).view(-1)
            # Calculate generator loss and backpropagate
            errG = criterion(output, label)
            errG.backward()
            # D_G_z2 = output.mean().item()
            optimizerG.step()

            # if (i+1) % 2 ==0:
            #     # Print loss for each batch
            print(f'{epoch + 1}-{i + 1}-lossG===>>{errG.item()}')
            print(f'{epoch + 1}-{i + 1}-lossD==>>{errD.item()}')

            # Store losses
            lossG = lossG + errG  # Accumulate batch losses
            lossD = lossD + errD  # Accumulate batch losses

        # schedulerD.step()
        schedulerG.step()
        # Calculate and visualize average loss per epoch
        avg_lossG = lossG / len(dataloader)
        avg_lossD = lossD / len(dataloader)
        G_losses.append(avg_lossG.item())
        D_losses.append(avg_lossD.item())
        viz_G.line([avg_lossG.item()], [epoch + 1], win='Generator_loss', update='append')
        viz_D.line([avg_lossD.item()], [epoch + 1], win='Discriminator_loss', update='append')
        # Save model weights every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpointG = {"model_state_dict": netG.state_dict(),
                           "optimizer_state_dict": optimizerG.state_dict(),
                           "epoch": epoch}
            path_checkpointG = "./models/checkpoint_G_{}_800_lr48.pth".format(epoch+1)
            torch.save(checkpointG, path_checkpointG)  # Save checkpoint file every 10 epochs
            checkpointD = {"model_state_dict": netD.state_dict(),
                           "optimizer_state_dict": optimizerD.state_dict(),
                           "epoch": epoch}
            path_checkpointD = "./models/checkpoint_D_{}_800_lr48.pth".format(epoch+1)
            torch.save(checkpointD, path_checkpointD)  # Save checkpoint file every 10 epochs
            print(' weight save successfully!')

        # Generate and save sample output for each epoch
        fake_epoch = netG(rnoise)
        # Resample to 100x100
        fake_epoch = fun.interpolate(fake_epoch, size=[100, 100], mode='bilinear', align_corners=True)

        depth = 16.0 * fake_epoch.squeeze(0).squeeze(0).detach().cpu().numpy()
        filepath = 'data/out/fake_' + "{:04d}".format(epoch + 1) + '.grd'
        outGrd.outGrd(filepath, depth, np.min(depth), np.max(depth))
        # plt.close()

    # Save final loss values and models
    np.savetxt('Generator_loss800_lr48.txt', G_losses)
    np.savetxt('Discriminator_loss800_lr48.txt', D_losses)
    torch.save(netG.state_dict(), 'models/netG800_lr48.pth')
    torch.save(netD.state_dict(), 'models/netD800_lr48.pth')
