import numpy as np
from scipy import stats
from scipy.stats import norm

class Analysis:
    """Statistical analysis module for QRNG data."""
    
    def calculate_variance(self, data):
        """Calculate variance of the data."""
        return np.var(data)
    
    def compute_mean(self, data):
        """Compute mean of the data."""
        return np.mean(data)
    
    def compute_std_dev(self, data):
        """Compute standard deviation of the data."""
        return np.std(data)
    
    def calculate_correlation(self, x, y):
        """Calculate correlation between two datasets."""
        return np.corrcoef(x, y)[0, 1]
    
    def compute_poisson_rate(self, data):
        """Compute Poisson rate (events per unit time) from count data."""
        return np.sum(data) / len(data)
    
    def calculate_confidence_interval(self, mean, std, alpha=0.95):
        """Calculate confidence interval for a mean."""
        z = norm.ppf(1-alpha/2)
        return (mean - z*std, mean + z*std)
    
    def compute_poisson_variance(self, data):
        """Compute Poisson variance (variance/mean ratio for overdispersion)."""
        rate = self.compute_poisson_rate(data)
        return np.var(data) / rate
