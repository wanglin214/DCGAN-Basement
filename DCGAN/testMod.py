import torch
import torch.nn.functional as fun
from model import DCGAN
import numpy as np
import matplotlib.pyplot as plt
from utils import outGrd
from scipy.ndimage import gaussian_filter

# Create 2D array for grid coordinates
x = np.linspace(0, 400, 100)
y = np.linspace(0, 400, 100)
X, Y = np.meshgrid(x, y)

# Set up GPU if available
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Load model weights
netG = DCGAN.Generator().to(device)
modelpath = 'models/netG400.pth'
# path_check = 'models/checkpoint_G_400_400.pth'
check_point = torch.load(modelpath, map_location=device)
# check_point = torch.load(path_check)
# Load state_dict into netG
netG.load_state_dict(check_point)
# netG.load_state_dict(check_point['model_state_dict'])
# opt_state = check_point['optimizer_state_dict']
# model_state = check_point['model_state_dict']
# print(opt_state)
# print(model_state)
netG.eval()  # Set to inference mode, to switch network layers like dropout and batchnorm between train and val modes
torch.no_grad()  # Disable autograd to speed up computation and save memory

nz = 200  # Noise dimension
sigma = 4  # Gaussian filter parameter

# Generate multiple samples
for i in range(0, 9500):
    # Generate random noise as input
    noise = torch.randn(1, nz, 1, 1, device=device)
    # Generate fake data
    fake = netG(noise)
    # print(fake.shape)
    # Resize to 100x100 using bilinear interpolation
    fake = fun.interpolate(fake, size=[100, 100], mode='bilinear', align_corners=True)
    # Scale output to depth range (0-16 km)
    Z = 16.0 * fake.squeeze(0).squeeze(0).detach().cpu().numpy()
    # print(Z.shape, type(Z))
    
    # # Plot contour map with rainbow colormap
    # plt.contourf(X, Y, Z, levels=20, cmap='rainbow_r')
    # plt.colorbar(label='Value')
    # plt.xlabel('X/km')
    # plt.ylabel('Y/km')
    # plt.title('Basement depth by DCGAN')
    # plt.show()
    # plt.close()
    
    # Clip negative values to zero
    Z[Z < 0] = 0
    # Apply Gaussian smoothing
    Z = gaussian_filter(Z, sigma=sigma)
    
    # Statistics of real data for potential normalization
    mean_real = 1.3794324
    std_real = 1.7700067
    min_real = 0.0
    max_real = 17.43
    
    # Normalization options (commented out)
    # Z = ((Z - np.mean(Z)) / np.std(Z)) * std_real + mean_real
    # Z = (Z - np.min(Z)) * (17.43 - 0.0) / (np.max(Z) - np.min(Z)) + 0.0
    
    # Save generated data to grid file
    filepath = 'data/Generated/SedofBasin_' + "{:04d}".format(i + 1 + 512) + '.grd'
    outGrd.outGrd(filepath, Z, np.min(Z), np.max(Z))
