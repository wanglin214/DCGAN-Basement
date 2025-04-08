import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader
from utils import dataload

# Set the data root path
dataroot = r"D:\Project\DCGAN\data\total"
# Load test data
dataloader = DataLoader(dataload.Basement(dataroot),
                        batch_size=10012,
                        shuffle=False,
                        )
print(len(dataloader))

if __name__ == '__main__':
    for i, data in enumerate(dataloader):
        print(data.shape)
        # Flatten the data into a 1D array
        total = data.view(-1).numpy()
        print(total.shape[0], type(total))
        
        # Calculate statistics: mean, standard deviation, min and max values
        mean_value = np.mean(total)
        std_dev = np.std(total)
        print(mean_value, std_dev, np.min(total), np.max(total))

        # Set font to 'Times New Roman'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

        # Create histogram of the data
        plt.hist(total, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')

        # Fit truncated Gaussian distribution
        # Generate Gaussian distributed data starting from zero
        data = np.random.normal(loc=1.54, scale=0.86, size=1000)
        data = np.clip(data, 0, 16.29)  # Clip values between 0 and 16.29

        # Define Gaussian probability density function for curve fitting
        def gaussian_pdf(x, mu, sigma):
            """
            Gaussian probability density function
            
            Parameters:
            x: independent variable
            mu: mean
            sigma: standard deviation
            
            Returns:
            probability density values
            """
            return norm.pdf(x, mu, sigma)

        # Create x-axis data points and perform fitting
        x = np.linspace(0, np.max(total), 10000)
        fit_params, covariance = curve_fit(gaussian_pdf, x, gaussian_pdf(x, np.mean(data), np.std(data)))

        # Generate the fitted probability density curve
        fit_curve = gaussian_pdf(x, *fit_params)

        # Plot the fitted curve
        plt.plot(x, fit_curve, 'k', linewidth=2)
        
        # Add labels and annotations
        plt.xlabel('Depth/km', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.xlim(0, 17)
        plt.ylim(0, 1)
        plt.text(2.5, 0.9, f'Fit: mean = {fit_params[0]:.2f}, std = {fit_params[1]:.2f}', fontsize=16)
        plt.text(15.0, 0.05, f'(c)', fontsize=20)

        # Set tick label font size
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        
        # Set figure size to fill the canvas
        fig = plt.gcf()
        fig.set_size_inches(4, 4)  # Adjust size as needed

        # Save the figure with 600 dpi resolution
        plt.savefig("real+DCGANdata_distribution.jpeg", bbox_inches='tight', dpi=600)
        
        # Display the figure
        plt.show()
