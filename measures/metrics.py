import numpy as np
from scipy import stats

class MetricCalculator:
    """Calculate various metrics for consciousness emergence analysis."""
    
    @staticmethod
    def calculate_metric(data):
        """Calculate a simple metric (mean) from data."""
        return np.mean(data)
    
    @staticmethod
    def calculate_entropy(data):
        """Calculate Shannon entropy of the data."""
        if len(data) == 0:
            return 0.0
        _, counts = np.unique(data, return_counts=True)
        probs = counts / len(data)
        return -np.sum(probs * np.log2(probs))
    
    @staticmethod
    def calculate_complexity(data):
        """Calculate complexity measure."""
        if len(data) < 2:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return (np.sum((data - mean)**2)) / std**2
    
    @staticmethod
    def calculate_correlation(x, y):
        """Calculate Pearson correlation coefficient."""
        return np.corrcoef(x, y)[0, 1]
