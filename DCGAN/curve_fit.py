import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

# Generate Gaussian distributed data starting from zero
data = np.random.normal(loc=2, scale=1, size=1000)
data = np.maximum(data, 0)  # Truncate negative values to zero

# Use curve_fit to fit the probability density function of normal distribution
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
x = np.linspace(0, max(data), 1000)
fit_params, covariance = curve_fit(gaussian_pdf, x, gaussian_pdf(x, np.mean(data), np.std(data)))

# Generate the fitted probability density curve
fit_curve = gaussian_pdf(x, *fit_params)

# Plot histogram and fitted curve
plt.hist(data, bins=30, density=True, alpha=0.7, color='b', edgecolor='black', label='Data Histogram')
plt.plot(x, fit_curve, 'k', linewidth=2, label='Fit Curve')

plt.title('Zero-Truncated Gaussian Distribution: Data and Fit')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()
