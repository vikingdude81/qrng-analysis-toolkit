import numpy as np

class CuriousModel:
    """Model for curiosity-driven exploration in data analysis."""
    
    def __init__(self, base_score=0.5):
        self.base_score = base_score
    
    def explore_data(self, data):
        """Explore data and return exploration score."""
        if len(data) == 0:
            return {'score': 0.0}
        
        mean = np.mean(data)
        std = np.std(data)
        variance = np.var(data)
        
        # Exploration score based on variance and mean
        score = min(1.0, (variance / (mean**2 + 1e-8)) * self.base_score)
        
        return {'score': float(score)}
    
    def get_exploration_direction(self, data):
        """Get direction of exploration based on data distribution."""
        if len(data) < 2:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        skewness = self._calculate_skewness(data)
        
        # Direction based on skewness and kurtosis
        direction = skewness / (np.abs(skewness) + 1e-8)
        return float(direction)
    
    def _calculate_skewness(self, data):
        """Calculate skewness of the data."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std)**3)
