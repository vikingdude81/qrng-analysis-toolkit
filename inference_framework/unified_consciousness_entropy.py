import json
import numpy as np
from scipy.stats import entropy
from numba import njit

@njit
def shannon_entropy(x):
    return entropy(x, base=2)

@njit
def renyi_alpha_entropy(x, alpha):
    if alpha == 1:
        return shannon_entropy(x)
    else:
        return -np.sum(x**alpha) / (1 - alpha)

@njit
def tsallis_q_entropy(x, q):
    if q == 1:
        return shannon_entropy(x)
    else:
        return (x**(q-1)).sum() / (q-1) - 1/(q-1)

@njit
def transfer_entropy(x, y, tau=1):
    px = np.histogram(x, bins='auto')[0] / len(x)
    py = np.histogram(y, bins='auto')[0] / len(y)
    pxy = np.histogram2d(x, y, bins='auto')[0] / len(x)
    
    te = 0
    for i in range(len(px)):
        for j in range(len(py)):
            if pxy[i, j] > 0:
                te += px[i] * py[j] * np.log(pxy[i, j] / (px[i] * py[j]))
    return -te

@njit
def granger_causality(x, y):
    from statsmodels.tsa.stattools import grangercausalitytests
    result = grangercausalitytests(np.column_stack((x, y)), maxlag=1, verbose=False)
    p_value = result[1]['ssr_ftest'][1]
    return p_value

@njit
def epiplexity(x, num_bootstraps=1000):
    from scipy.stats import norm
    n = len(x)
    bootstrapped_entropies = []
    
    for _ in range(num_bootstraps):
        bootstrap_indices = np.random.choice(n, n, replace=True)
        bootstrap_x = x[bootstrap_indices]
        bootstrapped_entropy = shannon_entropy(bootstrap_x)
        bootstrapped_entropies.append(bootstrapped_entropy)
    
    mean_entropy = np.mean(bootstrapped_entropies)
    std_dev = np.std(bootstrapped_entropies)
    z_score = (mean_entropy - shannon_entropy(x)) / std_dev
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    
    return mean_entropy, p_value

def unified_consciousness_entropy(qrng_time_series):
    entropy_results = {
        'shannon': shannon_entropy(qrng_time_series),
        'renyi_alpha_0.5': renyi_alpha_entropy(qrng_time_series, 0.5),
        'tsallis_q_1.5': tsallis_q_entropy(qrng_time_series, 1.5)
    }
    
    transfer_results = {
        'transfer_entropy': transfer_entropy(qrng_time_series[:-1], qrng_time_series[1:]),
        'granger_causality': granger_causality(qrng_time_series[:-1], qrng_time_series[1:])
    }
    
    epiplexity_result, p_value = epiplexity(qrng_time_series)
    
    report = {
        'entropy': entropy_results,
        'transfer_entropy': transfer_results,
        'epiplexity': epiplexity_result,
        'p_value': p_value
    }
    
    return json.dumps(report)

# Example usage:
# qrng_data = np.random.rand(100)  # Replace with actual QRNG data
# report = unified_consciousness_entropy(qrng_data)
# print(report)
