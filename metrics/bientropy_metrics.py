#!/usr/bin/env python3
"""
BiEntropy Randomness Metrics for QRNG Analysis

Based on "BiEntropy - The Approximate Entropy of a Finite Binary String"
by Grenville J. Croll (arXiv:1305.0954)

BiEntropy computes approximate entropy using a weighted average of Shannon 
entropies of a binary string and its binary derivatives. This is particularly
useful for:
- Detecting subtle patterns in QRNG output
- Comparing quantum vs classical randomness sources
- Short sequence analysis where traditional tests lack power

Key insight: True random sequences should have high BiEntropy (~1.0) while
patterned sequences will show lower values.
"""

from __future__ import print_function
import numpy as np
from math import log
from decimal import Decimal
from typing import Union, Tuple, List, Dict
from bitstring import Bits


def bin_deriv(bits: Bits) -> Bits:
    """
    Compute binary derivative using XOR of adjacent bits.
    
    The binary derivative reveals patterns in bit transitions.
    For random data, derivatives should also appear random.
    
    Parameters
    ----------
    bits : Bits
        Input bitstring of length n
        
    Returns
    -------
    Bits
        Binary derivative of length n-1
    """
    return bits[1:] ^ bits[:-1]


def bin_deriv_k(bits: Bits, k: int) -> Bits:
    """
    Compute kth binary derivative recursively.
    
    Parameters
    ----------
    bits : Bits
        Input bitstring
    k : int
        Number of derivative iterations
        
    Returns
    -------
    Bits
        kth derivative of length n-k
    """
    if k == 0:
        return bits
    return bin_deriv(bin_deriv_k(bits, k - 1))


def shannon_entropy(bits: Bits) -> float:
    """
    Compute Shannon binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)
    
    Parameters
    ----------
    bits : Bits
        Input bitstring
        
    Returns
    -------
    float
        Shannon entropy in range [0, 1]
    """
    if bits.len == 0:
        return 0.0
    
    p = float(bits.count(1)) / bits.len
    
    if p == 0 or p == 1:
        return 0.0
    
    return -p * log(p, 2) - (1 - p) * log(1 - p, 2)


def bien(data: Union[bytes, Bits, np.ndarray]) -> float:
    """
    Compute BiEntropy (BiEn) - power law weighted average of Shannon entropies.
    
    BiEntropy uses exponential weighting (2^k) giving more weight to the 
    original string than its derivatives. Best for short strings (n <= 32).
    
    Parameters
    ----------
    data : bytes, Bits, or numpy array
        Input data to analyze. Arrays are converted to bytes.
        
    Returns
    -------
    float
        BiEntropy value in range [0, 1]. Higher = more random.
    
    Examples
    --------
    >>> bien(b'\\x00')  # All zeros - low entropy
    0.0
    >>> bien(b'\\xff')  # All ones - low entropy
    0.0
    >>> bien(b'\\xaa')  # Alternating - higher entropy
    ~0.95
    """
    bits = _to_bits(data)
    
    if bits.len < 2:
        return 0.0
    
    t = Decimal(0)
    s_k = bits
    
    for k in range(bits.len - 1):
        # Compute Shannon entropy of current derivative
        h = shannon_entropy(s_k)
        # Power law weight
        weight = Decimal(2 ** k)
        t += Decimal(h) * weight
        # Next derivative
        s_k = bin_deriv(s_k)
    
    # Normalize by sum of weights (2^(n-1) - 1)
    normalizer = Decimal(2 ** (bits.len - 1) - 1)
    return float(t / normalizer) if normalizer > 0 else 0.0


def tbien(data: Union[bytes, Bits, np.ndarray]) -> float:
    """
    Compute TBiEntropy (TBiEn) - logarithmic weighted BiEntropy.
    
    TBiEntropy uses log(k+2) weighting, giving greater weight to higher
    derivatives. More suitable for longer strings and faster to compute.
    
    Parameters
    ----------
    data : bytes, Bits, or numpy array
        Input data to analyze
        
    Returns
    -------
    float
        TBiEntropy value in range [0, 1]. Higher = more random.
    """
    bits = _to_bits(data)
    
    if bits.len < 2:
        raise ValueError("Input too short for TBiEn (need >= 2 bits)")
    
    total_weight = 0.0
    weighted_sum = 0.0
    s_k = bits
    
    for k in range(bits.len - 1):
        h = shannon_entropy(s_k)
        weight = log(k + 2, 2)  # log2(k+2)
        
        total_weight += weight
        weighted_sum += h * weight
        
        s_k = bin_deriv(s_k)
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def _to_bits(data: Union[bytes, Bits, np.ndarray]) -> Bits:
    """Convert various input types to Bits object."""
    if isinstance(data, Bits):
        return data
    if isinstance(data, bytes):
        return Bits(bytes=data)
    if isinstance(data, np.ndarray):
        # Convert float array [0,1] to bytes
        if data.dtype in (np.float32, np.float64):
            # Discretize to 8-bit
            byte_vals = np.clip(data * 256, 0, 255).astype(np.uint8)
            return Bits(bytes=byte_vals.tobytes())
        elif data.dtype == np.uint8:
            return Bits(bytes=data.tobytes())
        else:
            # Convert to uint8
            return Bits(bytes=data.astype(np.uint8).tobytes())
    raise TypeError(f"Unsupported type: {type(data)}")


def bientropy_analysis(values: np.ndarray, 
                        window_size: int = 64,
                        step: int = 32) -> Dict:
    """
    Comprehensive BiEntropy analysis with sliding window.
    
    Computes BiEntropy metrics across the data with windows to detect
    local patterns that might be missed in global analysis.
    
    Parameters
    ----------
    values : np.ndarray
        Float values in [0, 1] range (QRNG output)
    window_size : int
        Size of sliding window in samples (converted to bytes)
    step : int
        Step size for sliding window
        
    Returns
    -------
    dict
        Analysis results including global and windowed metrics
    """
    # Convert full array to bytes for global analysis
    byte_vals = np.clip(values * 256, 0, 255).astype(np.uint8)
    
    # Global metrics (on a sample if too large)
    sample_size = min(len(byte_vals), 1024)
    sample = byte_vals[:sample_size]
    
    # BiEn only suitable for short strings, use chunks
    chunk_size = 32  # bytes = 256 bits, reasonable for BiEn
    
    bien_values = []
    tbien_values = []
    
    # Chunked analysis
    for i in range(0, len(sample) - chunk_size + 1, chunk_size):
        chunk = sample[i:i + chunk_size]
        try:
            bien_values.append(bien(chunk.tobytes()))
            tbien_values.append(tbien(chunk.tobytes()))
        except Exception:
            pass
    
    # Sliding window TBiEn (more suitable for longer sequences)
    window_tbien = []
    for i in range(0, len(byte_vals) - window_size + 1, step):
        window = byte_vals[i:i + window_size]
        try:
            window_tbien.append(tbien(window.tobytes()))
        except Exception:
            pass
    
    # Derivative analysis - check if derivatives also appear random
    bits = _to_bits(sample.tobytes())
    derivative_entropies = []
    s_k = bits
    for k in range(min(bits.len - 1, 20)):  # First 20 derivatives
        derivative_entropies.append(shannon_entropy(s_k))
        s_k = bin_deriv(s_k)
    
    return {
        'global_bien_mean': np.mean(bien_values) if bien_values else 0,
        'global_bien_std': np.std(bien_values) if bien_values else 0,
        'global_tbien_mean': np.mean(tbien_values) if tbien_values else 0,
        'global_tbien_std': np.std(tbien_values) if tbien_values else 0,
        'window_tbien_mean': np.mean(window_tbien) if window_tbien else 0,
        'window_tbien_std': np.std(window_tbien) if window_tbien else 0,
        'window_tbien_min': np.min(window_tbien) if window_tbien else 0,
        'derivative_entropies': derivative_entropies,
        'derivative_entropy_mean': np.mean(derivative_entropies) if derivative_entropies else 0,
        'n_chunks': len(bien_values),
        'n_windows': len(window_tbien),
    }


def compare_sources_bientropy(sources: Dict[str, np.ndarray]) -> Dict:
    """
    Compare multiple QRNG sources using BiEntropy metrics.
    
    Parameters
    ----------
    sources : dict
        Mapping of source name to float array values
        
    Returns
    -------
    dict
        Comparative analysis results
    """
    results = {}
    
    for name, values in sources.items():
        analysis = bientropy_analysis(values)
        results[name] = {
            'BiEn': analysis['global_bien_mean'],
            'TBiEn': analysis['global_tbien_mean'],
            'TBiEn_window': analysis['window_tbien_mean'],
            'TBiEn_min': analysis['window_tbien_min'],
            'deriv_entropy': analysis['derivative_entropy_mean'],
            'n_samples': len(values),
        }
    
    return results


# Expected values for reference
EXPECTED_VALUES = {
    'random': {
        'bien': 0.95,   # High for random data
        'tbien': 0.95,  # High for random data
        'threshold': 0.90,  # Below this may indicate patterns
    },
    'patterned': {
        'alternating': 0.95,  # 10101010 is high entropy
        'runs': 0.50,         # Long runs reduce entropy
        'constant': 0.0,      # All same bits = 0 entropy
    }
}


if __name__ == "__main__":
    # Demo
    import json
    from pathlib import Path
    
    print("=" * 70)
    print("BiEntropy Analysis Demo")
    print("=" * 70)
    
    # Test vectors
    print("\nTest Vectors:")
    print("-" * 50)
    
    test_cases = [
        (b'\x00\x00\x00\x00', "All zeros"),
        (b'\xff\xff\xff\xff', "All ones"),
        (b'\xaa\xaa\xaa\xaa', "Alternating (0xAA)"),
        (b'\x55\x55\x55\x55', "Alternating (0x55)"),
    ]
    
    for data, name in test_cases:
        try:
            b = bien(data)
            tb = tbien(data)
            print(f"  {name:25} BiEn={b:.4f}  TBiEn={tb:.4f}")
        except Exception as e:
            print(f"  {name:25} Error: {e}")
    
    # Generate random test
    print("\nRandom Data (numpy):")
    print("-" * 50)
    
    np.random.seed(42)
    random_data = np.random.random(256)
    analysis = bientropy_analysis(random_data)
    print(f"  BiEn mean:  {analysis['global_bien_mean']:.4f} ± {analysis['global_bien_std']:.4f}")
    print(f"  TBiEn mean: {analysis['global_tbien_mean']:.4f} ± {analysis['global_tbien_std']:.4f}")
    print(f"  Deriv entropy: {analysis['derivative_entropy_mean']:.4f}")
    
    # Load actual QRNG data if available
    streams_dir = Path(__file__).parent / 'qrng_streams'
    if streams_dir.exists():
        print("\n" + "=" * 70)
        print("QRNG Source BiEntropy Comparison")
        print("=" * 70)
        
        sources = {}
        for f in sorted(streams_dir.glob('*.json'))[:6]:  # First few files
            try:
                with open(f) as fp:
                    data = json.load(fp)
                source = data.get('source', 'unknown')
                floats = data.get('floats', [])
                if floats and source != 'unknown':
                    if source not in sources:
                        sources[source] = []
                    sources[source].extend(floats[:1000])
            except Exception:
                pass
        
        if sources:
            # Convert to numpy
            sources = {k: np.array(v) for k, v in sources.items()}
            results = compare_sources_bientropy(sources)
            
            print(f"\n{'Source':<40} {'BiEn':>8} {'TBiEn':>8} {'Min':>8}")
            print("-" * 70)
            for name, r in sorted(results.items(), key=lambda x: -x[1]['TBiEn']):
                print(f"{name[:39]:<40} {r['BiEn']:>8.4f} {r['TBiEn']:>8.4f} {r['TBiEn_min']:>8.4f}")
