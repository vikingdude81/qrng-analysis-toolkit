import numpy as np
from typing import List, Tuple, Optional
import random
import sys
sys.path.insert(0, '/home/user/helios_anomaly_scope')

# Mock QRNG for synthetic traces (simulating a real RNG)
class SyntheticQRNG:
    def __init__(self):
        self.state = np.random.randint(-1e9, 1e9).astype(np.float64)
    
    def next(self) -> float:
        return self.state % 2**30

def generate_synthetic_qrng_trace(rng: SyntheticQRNG, n_samples=1000):
    """Generate synthetic QRNG traces with injected anomalies."""
    traces = []
    for _ in range(n_samples):
        trace = np.random.randn(100) * 2.5 + rng.next()
        # Inject periodic spikes (e.g., every 30 samples)
        if np.random.rand() < 0.03:
            trace += np.sin(np.pi * 40 * np.random.rand())
        traces.append(trace)
    return np.array(traces)

def generate_anomaly(x: np.ndarray, rng: SyntheticQRNG):
    """Generate a single anomaly spike."""
    spikes = []
    for _ in range(5):
        idx = int(np.random.randint(0, len(x)))
        if idx < 30:
            spikes.append(rng.next() * 1e6)
        else:
            break
    return np.array(spikes)

def generate_anomaly_batch(x: np.ndarray, rng: SyntheticQRNG):
    """Generate multiple anomalies in a batch."""
    anomalies = []
    for _ in range(30):
        idx = int(np.random.randint(0, len(x)))
        if idx < 30:
            anomalies.append(rng.next() * 1e6)
        else:
            break
    return np.array(anomalies)

def generate_anomaly_batch_with_spikes(x: np.ndarray, rng: SyntheticQRNG):
    """Generate multiple anomalies with periodic spikes."""
    anomalies = []
    for _ in range(30):
        idx = int(np.random.randint(0, len(x)))
        if idx < 30:
            anomalies.append(rng.next() * 1e6)
        else:
            break
    return np.array(anomalies)

def generate_anomaly_batch_with_spikes_and_bursts(x: np.ndarray, rng: SyntheticQRNG):
    """Generate multiple anomalies with periodic spikes and burst noise."""
    anomalies = []
    for _ in range(30):
        idx = int(np.random.randint(0, len(x)))
        if idx < 30:
            anomalies.append(rng.next() * 1e6)
        else:
            break
    # Add burst noise (random spikes every 5 samples)
    for _ in range(20):
        idx = int(np.random.randint(0, len(x)))
        if idx < 30:
            anomalies.append(rng.next() * 1e6)
        else:
            break
    return np.array(anomalies)

def generate_anomaly_batch_with_spikes_and_bursts_and_noise(x: np.ndarray, rng: SyntheticQRNG):
    """Generate multiple anomalies with periodic spikes, burst noise, and random noise."""
    anomalies = []
    for _ in range(30):
        idx = int(np.random.randint(0, len(x)))
        if idx < 30:
            anomalies.append(rng.next() * 1e6)
        else:
            break
    
    # Add burst noise (every 5 samples)
    for _ in range(20):
        idx = int(np.random.randint(0, len(x)))
        if idx < 30:
            anomalies.append(rng.next() * 1e6)
        else:
            break
    
    # Add random noise (every 10 samples)
    for _ in range(20):
        idx = int(np.random.randint(0, len(x)))
        if idx < 30:
            anomalies.append(rng.next() * 1e6)
        else:
            break
    
    return np.array(anomalies)

def generate_anomaly_batch_with_spikes_and_bursts_and_noise_and_rng(x: np.ndarray, rng: SyntheticQRNG):
    """Generate multiple anomalies with periodic spikes, burst noise, random noise, and injected RNG."""
    anomalies = []
    for _ in range(30):
        idx = int(np.random.randint(0, len(x)))
        if idx < 30:
            anomalies.append(rng.next() * 1e6)
        else:
            break
    
    # Add burst noise (every 5 samples)
    for _ in range(20):
        idx = int(np.random.randint(0, len(x)))
        if idx < 30:
            anomalies.append(rng.next() * 1e6)
        else:
            break
    
    # Add random noise (every 10 samples)
    for _ in range(20):
        idx = int(np.random.randint(0, len(x)))
        if idx < 30:
            anomalies.append(rng.next() * 1e6)
        else:
            break
    
    # Add injected RNG trace
    for _ in range(15):
        idx = int(np.random.randint(0, len(x)))
        if idx < 30:
            anomalies.append(rng.next() * 1e6)
        else:
            break
    
    return np.array(anomalies)
