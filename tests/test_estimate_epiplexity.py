import numpy as np
from scipy.signal import correlate

def estimate_epiplexity(signal, lag=1, dim=3, alpha=0.05):
    """Estimate epiplexity of a time series."""
    if len(signal) < 2 * lag:
        raise ValueError("Signal length must be at least 2*lag")
    
    # Embed the signal
    X = np.zeros((dim, len(signal)))
    for i in range(dim):
        X[i] = signal[lag*i : lag*(i+1)]
    
    # Compute mutual information between original and delayed signals
    mi = np.zeros(len(X))
    for j in range(dim):
        mi += correlate(X[j], X[j], mode='full', method='direct')
    
    return mi.mean() - alpha

# Test with Lorenz system (example)
def generate_lorenz():
    import numpy as np
    from scipy.integrate import odeint
    
    def lorenz(x, t, sigma=10, beta=8/3, rho=28):
        return [sigma * (x[1] - x[0]), x[0] * x[2] - beta * x[1], x[0] * x[1] - rho * x[2]]
    
    # Initial condition
    x0 = np.array([1.0, 1.0, 1.0])
    t = np.linspace(0, 10, 1000)
    sol = odeint(lorenz, x0, t)
    return sol[:, 0], sol[:, 1], sol[:, 2]

# Test epiplexity
def test_epiplexity():
    signal, y1, y2 = generate_lorenz()
    lag = 1
    dim = 3
    epiplexity = estimate_epiplexity(signal, lag=lag, dim=dim)
    assert epiplexity > 0.5, "Epiplexity not above 0.5 for deterministic chaos"

test_epiplexity()
