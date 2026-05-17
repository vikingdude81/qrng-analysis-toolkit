import numpy as np
from scipy import stats
from scipy.special import gamma, erf
from scipy.optimize import brentq
from scipy.integrate import quad
import h5py
import os
import tempfile

# Configuration
SEED = 42
OUTPUT_PATH = "/tmp/helios_integration_test.h5"

def generate_synthetic_spdc_data(seed=SEED):
    """Generate synthetic SPDC data for testing."""
    n_samples = seed * 10000
    np.random.seed(seed)
    
    # Generate initial photon number distribution (Poisson-like)
    n_photons = np.random.poisson(2, size=(n_samples, 3))
    
    # Simulate SPDC: n photons -> m pairs + noise
    # Using Gaussian approximation for simplicity in simulation
    m_pairs = np.random.normal(loc=0.5, scale=1.0, size=n_samples)
    noise = np.random.normal(0, 0.2, size=(n_samples, 3))
    
    # Combine pairs and noise
    n_photons_final = m_pairs + noise
    
    return n_photons_final

def compute_entropy(data):
    """Compute Shannon entropy of data."""
    if len(data) == 0:
        return 0.0
    counts = np.bincount(data, minlength=3)
    # Normalize to probabilities
    probs = counts / (counts.sum() + 1e-10)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def compute_rnyi_entropy(data):
    """Compute Rényi entropy of order alpha."""
    if len(data) == 0:
        return 0.0
    counts = np.bincount(data, minlength=3)
    probs = counts / (counts.sum() + 1e-10)
    # RÃ©nyi entropy is sum(p^alpha * log(p)) for alpha > 0
    rnyi_entropy = -np.sum(probs ** 2 * np.log(probs))
    return rnyi_entropy

def compute_lyapunov_exponents(data):
    """Compute Lyapunov exponents of data."""
    if len(data) == 0:
        return [0.0] * len(data)
    
    # Compute eigenvalues of the covariance matrix
    cov = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = stats.eigvalde(cov)
    
    lyapunov_exponents = []
    for i in range(len(eigenvalues)):
        if abs(eigenvalues[i]) > 1e-6:
            lyapunov_exponents.append(np.log(abs(eigenvalues[i])))
        else:
            lyapunov_exponents.append(0.0)
    
    return np.array(lyapunov_exponents)

def compute_epiplexity(data):
    """Compute epiplexity score."""
    if len(data) == 0:
        return 0.0
    
    # Simple heuristic based on data density and structure
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    
    # Epiplexity is often related to the spread of values relative to mean
    # Using a simplified metric: variance normalized by mean^2
    epiplexity = (std / (mean * 0.5)) ** 2
    
    return epiplexity

def compute_influence_score(data):
    """Compute influence score based on data spread."""
    if len(data) == 0:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    
    # Influence is inversely related to variance (less spread = higher influence)
    # Using a normalized scale factor
    influence_score = max(0, 1 - (std / 5))
    
    return influence_score

def detect_anomalies(data):
    """Detect anomalies in the data."""
    if len(data) == 0:
        return []
    
    # Simple outlier detection using Z-score
    mean = np.mean(data)
    std = np.std(data)
    
    outliers = []
    for i in range(len(data)):
        z_score = (data[i] - mean) / std
        if abs(z_score) > 2.0:
            outliers.append(i)
    
    return outliers

def save_test_output(filepath, data):
    """Save test output to H5."""
    with h5py.File(filepath, 'w') as f:
        # Store metadata
        f.create_dataset('metadata', shape=(1,), data=[SEED])
        
        # Store computed metrics
        for metric_name in ['entropy_shannon', 'rnyi_entropy', 
                           'lyapunov_exponents', 'epiplexity', 
                           'influence_score']:
            if metric_name not in f.data:
                f.create_dataset(metric_name, shape=(1,), data=0.0)
            
            # Store actual values for debugging/verification
            if metric_name == 'entropy_shannon':
                f[metric_name][()] = compute_entropy(data)
            elif metric_name == 'rnyi_entropy':
                f[metric_name][()] = compute_rnyi_entropy(data)
            elif metric_name == 'lyapunov_exponents':
                f[metric_name][()] = compute_lyapunov_exponents(data)
            elif metric_name == 'epiplexity':
                f[metric_name][()] = compute_epiplexity(data)
            elif metric_name == 'influence_score':
                f[metric_name][()] = compute_influence_score(data)
        
        # Store anomaly indices
        f.create_dataset('anomaly_indices', shape=(1,), data=0.0)
        f['anomaly_indices'][()] = detect_anomalies(data)
