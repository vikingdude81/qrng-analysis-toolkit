import numpy as np
from scipy.stats import entropy, phi
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
import pandas as pd
import os

# Ensure metrics directory exists
os.makedirs('metrics', exist_ok=True)

def compute_confounded_metrics(state_space_partitioning):
    """Simulate ensemble of surrogates and return entropy/Phi_hat values."""
    n_surrogates = 10_000
    
    # Simulate surrogate states based on partitioning
    state_indices = np.random.choice(n_surrogates, size=n_surrogates)
    
    # Compute metrics for each surrogate
    entropies = []
    phi_hat_values = []
    
    for i in range(n_surrogates):
        entropy_val = entropy(state_indices[i])
        phi_hat_val = np.mean(np.abs(state_indices[i] - state_space_partitioning))
        
        entropies.append(entropy_val)
        phi_hat_values.append(phi_hat_val)
    
    return np.array(entropies), np.array(phi_hat_values)

def compute_consensus_entropy(metrics):
    """Compute consensus entropy using copula-based fusion."""
    # Normalize metrics to [0, 1] range
    m = np.array(metrics) / (np.max(m) + 1e-8)
    
    # Use copulas to fuse distributions
    from scipy.stats import Copula
    
    # Create a joint distribution using copulas
    c = Copula('copula')
    c.fit(m)
    
    # Compute consensus entropy as the minimum of marginals (or weighted sum)
    # For simplicity, use min of marginals which is equivalent to max of copulas
    consensus_entropy = c.compute_marginal_entropy()
    
    return consensus_entropy

def power_analysis(n_surrogates):
    """Simulate power analysis for ensemble surrogates."""
    n_true = 10_000
    
    # Simulate surrogate states based on true partitioning
    state_indices = np.random.choice(n_surrogates, size=n_true)
    
    # Compute metrics
    entropies = []
    phi_hat_values = []
    
    for i in range(n_surrogates):
        entropy_val = entropy(state_indices[i])
        phi_hat_val = np.mean(np.abs(state_indices[i] - state_space_partitioning))
        
        entropies.append(entropy_val)
        phi_hat_values.append(phi_hat_val)
    
    # Compute consensus entropy
    m = np.array(entropies) / (np.max(m) + 1e-8)
    c = Copula('copula')
    c.fit(m)
    consensus_entropy = c.compute_marginal_entropy()
    
    # Calculate power: P(Type I error) = P(consensus < threshold | true is correct)
    # Threshold based on expected consensus entropy
    expected_consensus = np.mean(entropies) / n_surrogates
    
    p_type_i_error = 0.5 * (1 - c.cdf(expected_consensus, 0, 1))
    
    return {
        'n_true': n_true,
        'expected_consensus_entropy': expected_consensus,
        'p_type_i_error': p_type_i_error,
        'consensus_entropy': consensus_entropy
    }

def main():
    # Simulate surrogates
    state_space_partitioning = np.array([0.5, 0.3, 0.2])
    
    entropies, phi_hat_values = compute_confounded_metrics(state_space_partitioning)
    metrics_df = pd.DataFrame({
        'state': state_space_partitioning,
        'entropy': entropies,
        'phi_hat': phi_hat_values
    })
    
    # Compute consensus entropy
    consensus_entropy = compute_consensus_entropy(metrics_df)
    
    # Power analysis
    power_analysis_result = power_analysis(n_surrogates=10_000)
    
    print("Consensus Entropy:", consensus_entropy)
    print(f"Power Analysis (n=10,000): p_type_i_error = {power_analysis_result['p_type_i_error']:.4f}")

if __name__ == "__main__":
    main()