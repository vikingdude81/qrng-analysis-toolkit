#!/usr/bin/env python3
"""
Chaos Theory Analysis for QRNG Data
Applies nonlinear dynamics and chaos measures to 43K+ samples.

Metrics:
- Lyapunov Exponent: Sensitivity to initial conditions
- Correlation Dimension: Fractal dimension of attractor
- Hurst Exponent: Long-term memory/persistence
- Approximate Entropy: Complexity/regularity
- Sample Entropy: Improved ApEn for short series
- Recurrence Quantification Analysis (RQA)
- Phase Space Reconstruction (Takens embedding)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


def load_all_streams():
    """Load all QRNG streams."""
    streams_dir = Path("qrng_streams")
    sources = defaultdict(list)
    
    for f in sorted(streams_dir.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        
        source = data.get("source", "unknown")
        samples = data.get("floats", data.get("samples", data.get("values", [])))
        
        if len(samples) > 0:
            sources[source].append(np.array(samples))
    
    # Combine samples per source
    combined = {}
    for source, arrays in sources.items():
        all_samples = np.concatenate(arrays)
        if len(all_samples) > 0:
            combined[source] = all_samples
    
    return combined


def lyapunov_exponent(x, m=3, tau=1, eps=0.01, max_iter=50):
    """
    Estimate largest Lyapunov exponent using Rosenstein's method.
    
    Positive λ → chaos (exponential divergence)
    Zero λ → periodic/quasiperiodic
    Negative λ → stable fixed point
    
    For truly random data, λ should be large and positive.
    """
    n = len(x)
    if n < 500:
        return np.nan, "Insufficient data"
    
    # Limit for computational efficiency
    n = min(n, 2000)
    x = x[:n]
    
    # Phase space reconstruction (Takens embedding)
    N = n - (m - 1) * tau
    if N < 100:
        return np.nan, "Insufficient embedding points"
    
    # Build embedded vectors
    embedded = np.zeros((N, m))
    for i in range(N):
        for j in range(m):
            embedded[i, j] = x[i + j * tau]
    
    # Compute all pairwise distances at once (vectorized)
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(embedded, embedded)
    
    # Set temporal neighbors to infinity (Theiler window)
    theiler = tau * m
    for i in range(N):
        for j in range(max(0, i - theiler), min(N, i + theiler + 1)):
            dist_matrix[i, j] = np.inf
    
    # Find nearest neighbors
    divergence = []
    n_samples = min(200, N - max_iter)  # Sample subset for speed
    indices = np.random.choice(N - max_iter, n_samples, replace=False)
    
    for i in indices:
        nn_idx = np.argmin(dist_matrix[i])
        min_dist = dist_matrix[i, nn_idx]
        
        if min_dist < np.inf and min_dist > 1e-10 and nn_idx + max_iter < N:
            # Track divergence
            local_div = []
            for k in range(1, min(max_iter, N - max(i, nn_idx))):
                d = np.linalg.norm(embedded[i + k] - embedded[nn_idx + k])
                if d > 1e-10:
                    local_div.append(np.log(d / min_dist))
            
            if len(local_div) > 5:
                divergence.append(local_div)
    
    if len(divergence) < 10:
        return np.nan, "Could not compute divergence"
    
    # Average divergence curves
    min_len = min(len(d) for d in divergence)
    avg_divergence = np.mean([d[:min_len] for d in divergence], axis=0)
    
    # Linear fit to get Lyapunov exponent
    t = np.arange(1, min_len + 1)
    slope, _, r_value, _, _ = stats.linregress(t, avg_divergence)
    
    return slope, f"R²={r_value**2:.3f}"


def correlation_dimension(x, m_max=10, tau=1, r_vals=None):
    """
    Estimate correlation dimension using Grassberger-Procaccia algorithm.
    
    D2 ~ 1-2: Low-dimensional chaos
    D2 ~ 2-5: Higher-dimensional chaos
    D2 → ∞ or saturates at embedding dim: Random noise
    
    True random should show D2 increasing with embedding dimension.
    """
    n = len(x)
    if n < 1000:
        return np.nan, "Insufficient data"
    
    # Limit for efficiency
    n = min(n, 3000)
    x = x[:n]
    
    results = []
    
    for m in range(2, min(m_max, 8)):
        # Embed
        N = n - (m - 1) * tau
        if N < 100:
            break
        
        embedded = np.zeros((N, m))
        for i in range(N):
            for j in range(m):
                embedded[i, j] = x[i + j * tau]
        
        # Compute pairwise distances
        if N > 500:
            # Subsample for large N
            idx = np.random.choice(N, 500, replace=False)
            embedded_sub = embedded[idx]
        else:
            embedded_sub = embedded
        
        distances = pdist(embedded_sub)
        distances = distances[distances > 1e-10]
        
        if len(distances) < 100:
            continue
        
        # Correlation sum at different radii
        r_range = np.logspace(np.log10(np.percentile(distances, 1)), 
                              np.log10(np.percentile(distances, 50)), 15)
        
        log_r = []
        log_C = []
        
        for r in r_range:
            C = np.mean(distances < r)
            if C > 0:
                log_r.append(np.log(r))
                log_C.append(np.log(C))
        
        if len(log_r) > 5:
            # Linear region slope gives D2
            slope, _, r_value, _, _ = stats.linregress(log_r, log_C)
            results.append((m, slope, r_value**2))
    
    if not results:
        return np.nan, "Could not estimate"
    
    # Take the dimension estimate from highest embedding that converged
    dims = [r[1] for r in results if r[2] > 0.9]
    if dims:
        return np.mean(dims[-3:]) if len(dims) >= 3 else dims[-1], f"m={results[-1][0]}"
    
    return results[-1][1], f"m={results[-1][0]}, low R²"


def hurst_exponent(x, min_window=10, max_window=None):
    """
    Estimate Hurst exponent using R/S analysis.
    
    H = 0.5: Random walk (no memory)
    H > 0.5: Persistent/trending
    H < 0.5: Anti-persistent/mean-reverting
    
    True random should have H ≈ 0.5
    """
    n = len(x)
    if max_window is None:
        max_window = n // 4
    
    if n < 100:
        return np.nan, "Insufficient data"
    
    # Range of window sizes
    window_sizes = []
    rs_values = []
    
    for window in range(min_window, min(max_window, n // 2), max(1, (max_window - min_window) // 20)):
        n_windows = n // window
        if n_windows < 2:
            continue
        
        rs_list = []
        for i in range(n_windows):
            segment = x[i * window:(i + 1) * window]
            
            # Mean-adjusted cumulative sum
            mean = np.mean(segment)
            cumsum = np.cumsum(segment - mean)
            
            # Range
            R = np.max(cumsum) - np.min(cumsum)
            
            # Standard deviation
            S = np.std(segment, ddof=1)
            
            if S > 1e-10:
                rs_list.append(R / S)
        
        if rs_list:
            window_sizes.append(window)
            rs_values.append(np.mean(rs_list))
    
    if len(window_sizes) < 5:
        return np.nan, "Could not estimate"
    
    # Log-log regression
    log_n = np.log(window_sizes)
    log_rs = np.log(rs_values)
    
    slope, _, r_value, _, _ = stats.linregress(log_n, log_rs)
    
    return slope, f"R²={r_value**2:.3f}"


def approximate_entropy(x, m=2, r=None):
    """
    Approximate Entropy (ApEn) - measures regularity/predictability.
    
    ApEn ≈ 0: Highly regular/predictable
    ApEn → high: Complex/random
    
    True random should have high ApEn.
    """
    n = len(x)
    if n < 200:
        return np.nan, "Insufficient data"
    
    # Use subset for efficiency
    n = min(n, 2000)
    x = x[:n]
    
    if r is None:
        r = 0.2 * np.std(x)
    
    def phi(m):
        N = n - m + 1
        
        # Build template vectors
        templates = np.zeros((N, m))
        for i in range(N):
            templates[i] = x[i:i + m]
        
        # Count matches
        counts = np.zeros(N)
        for i in range(N):
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            counts[i] = np.sum(dist <= r) / N
        
        return np.mean(np.log(counts + 1e-10))
    
    return phi(m) - phi(m + 1), f"r={r:.4f}"


def sample_entropy(x, m=2, r=None):
    """
    Sample Entropy (SampEn) - improved ApEn without self-matches.
    
    More robust for short time series.
    """
    n = len(x)
    if n < 200:
        return np.nan, "Insufficient data"
    
    n = min(n, 1000)  # Limit for O(n²) computation
    x = x[:n]
    
    if r is None:
        r = 0.2 * np.std(x)
    
    def count_matches_vectorized(templates, r):
        """Vectorized match counting using scipy distance."""
        from scipy.spatial.distance import pdist
        # Chebyshev distance (max abs diff)
        dists = pdist(templates, metric='chebyshev')
        return np.sum(dists <= r)
    
    # Template vectors of length m
    N_m = n - m
    templates_m = np.array([x[i:i + m] for i in range(N_m)])
    
    # Template vectors of length m+1
    N_m1 = n - m - 1
    templates_m1 = np.array([x[i:i + m + 1] for i in range(N_m1)])
    
    A = count_matches_vectorized(templates_m1, r)
    B = count_matches_vectorized(templates_m, r)
    
    if B == 0 or A == 0:
        return np.nan, "No matches found"
    
    return -np.log(A / B), f"r={r:.4f}"


def recurrence_quantification(x, m=3, tau=1, eps=None):
    """
    Recurrence Quantification Analysis (RQA).
    
    Metrics:
    - RR: Recurrence Rate (density of recurrence points)
    - DET: Determinism (predictability)
    - LAM: Laminarity (laminar states)
    - L_mean: Mean diagonal line length
    
    Random data: Low DET, low LAM, short lines
    Chaotic data: Moderate DET, complex patterns
    Periodic data: High DET, long diagonal lines
    """
    n = len(x)
    if n < 500:
        return {}, "Insufficient data"
    
    n = min(n, 1500)  # Limit for memory
    x = x[:n]
    
    # Embed
    N = n - (m - 1) * tau
    embedded = np.zeros((N, m))
    for i in range(N):
        for j in range(m):
            embedded[i, j] = x[i + j * tau]
    
    # Subsample if too large
    if N > 800:
        idx = np.random.choice(N, 800, replace=False)
        idx = np.sort(idx)
        embedded = embedded[idx]
        N = 800
    
    # Compute distance matrix
    dist_matrix = squareform(pdist(embedded))
    
    # Threshold
    if eps is None:
        eps = 0.1 * np.std(dist_matrix)
    
    # Recurrence matrix
    R = (dist_matrix <= eps).astype(int)
    
    # RR: Recurrence Rate
    RR = np.sum(R) / (N * N)
    
    # Find diagonal lines
    def count_diagonal_lines(R, min_len=2):
        N = len(R)
        lines = []
        
        for offset in range(-N + min_len, N - min_len + 1):
            diag = np.diag(R, offset)
            
            # Find consecutive 1s
            line_len = 0
            for val in diag:
                if val:
                    line_len += 1
                else:
                    if line_len >= min_len:
                        lines.append(line_len)
                    line_len = 0
            if line_len >= min_len:
                lines.append(line_len)
        
        return lines
    
    # Find vertical lines
    def count_vertical_lines(R, min_len=2):
        N = len(R)
        lines = []
        
        for col in range(N):
            line_len = 0
            for row in range(N):
                if R[row, col]:
                    line_len += 1
                else:
                    if line_len >= min_len:
                        lines.append(line_len)
                    line_len = 0
            if line_len >= min_len:
                lines.append(line_len)
        
        return lines
    
    diag_lines = count_diagonal_lines(R)
    vert_lines = count_vertical_lines(R)
    
    # DET: Determinism (ratio of points in diagonal lines to all recurrence points)
    total_recurrence = np.sum(R) - N  # Exclude main diagonal
    if diag_lines and total_recurrence > 0:
        total_diag = sum(diag_lines)
        DET = total_diag / total_recurrence
        L_mean = np.mean(diag_lines)
        L_max = max(diag_lines)
    else:
        DET = 0
        L_mean = 0
        L_max = 0
    
    # LAM: Laminarity
    if vert_lines:
        total_vert = sum(vert_lines)
        LAM = total_vert / max(np.sum(R), 1)
    else:
        LAM = 0
    
    return {
        'RR': RR,
        'DET': DET,
        'LAM': LAM,
        'L_mean': L_mean,
        'L_max': L_max,
        'eps': eps
    }, "OK"


def permutation_entropy(x, m=3, tau=1):
    """
    Permutation Entropy - complexity based on ordinal patterns.
    
    PE → 1: Completely random
    PE → 0: Completely deterministic
    """
    n = len(x)
    if n < 100:
        return np.nan, "Insufficient data"
    
    n = min(n, 5000)
    x = x[:n]
    
    from itertools import permutations
    import math
    
    # All possible permutations
    perms = list(permutations(range(m)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    n_perms = len(perms)
    
    # Count ordinal patterns
    counts = np.zeros(n_perms)
    N = n - (m - 1) * tau
    
    for i in range(N):
        # Get values at embedding positions
        values = [x[i + j * tau] for j in range(m)]
        
        # Get ordinal pattern (rank order)
        pattern = tuple(np.argsort(values))
        
        if pattern in perm_to_idx:
            counts[perm_to_idx[pattern]] += 1
    
    # Normalize
    probs = counts / N
    probs = probs[probs > 0]
    
    # Shannon entropy, normalized
    H = -np.sum(probs * np.log(probs))
    H_max = np.log(math.factorial(m))
    
    return H / H_max, f"m={m}"


def detrended_fluctuation_analysis(x, min_box=4, max_box=None):
    """
    Detrended Fluctuation Analysis (DFA) - self-similarity/scaling.
    
    α = 0.5: Uncorrelated (white noise)
    α < 0.5: Anti-correlated
    α > 0.5: Long-range positive correlations
    α = 1.0: 1/f noise
    α = 1.5: Brownian noise
    """
    n = len(x)
    if max_box is None:
        max_box = n // 4
    
    if n < 100:
        return np.nan, "Insufficient data"
    
    # Integrate the series
    y = np.cumsum(x - np.mean(x))
    
    box_sizes = []
    fluctuations = []
    
    for box_size in range(min_box, max_box, max(1, (max_box - min_box) // 20)):
        n_boxes = n // box_size
        if n_boxes < 2:
            continue
        
        fluct = []
        for i in range(n_boxes):
            segment = y[i * box_size:(i + 1) * box_size]
            
            # Linear detrend
            t = np.arange(box_size)
            coef = np.polyfit(t, segment, 1)
            trend = np.polyval(coef, t)
            
            # RMS fluctuation
            fluct.append(np.sqrt(np.mean((segment - trend) ** 2)))
        
        box_sizes.append(box_size)
        fluctuations.append(np.mean(fluct))
    
    if len(box_sizes) < 5:
        return np.nan, "Could not estimate"
    
    # Log-log slope
    log_n = np.log(box_sizes)
    log_F = np.log(fluctuations)
    
    slope, _, r_value, _, _ = stats.linregress(log_n, log_F)
    
    return slope, f"R²={r_value**2:.3f}"


def analyze_source(name, samples):
    """Run all chaos analyses on a source."""
    print(f"\n{'─'*70}")
    print(f"  {name}")
    print(f"  {len(samples):,} samples")
    print(f"{'─'*70}")
    
    results = {}
    
    # Lyapunov Exponent
    print("  Computing Lyapunov exponent...", end=" ", flush=True)
    lyap, lyap_info = lyapunov_exponent(samples)
    results['lyapunov'] = lyap
    if not np.isnan(lyap):
        if lyap > 0.1:
            status = "CHAOTIC/RANDOM"
        elif lyap > 0:
            status = "WEAKLY CHAOTIC"
        else:
            status = "STABLE/PERIODIC"
        print(f"λ = {lyap:.4f} ({lyap_info}) → {status}")
    else:
        print(f"Could not compute: {lyap_info}")
    
    # Hurst Exponent
    print("  Computing Hurst exponent...", end=" ", flush=True)
    hurst, hurst_info = hurst_exponent(samples)
    results['hurst'] = hurst
    if not np.isnan(hurst):
        if 0.45 < hurst < 0.55:
            status = "TRUE RANDOM (no memory)"
        elif hurst > 0.55:
            status = "PERSISTENT (trending)"
        else:
            status = "ANTI-PERSISTENT (mean-reverting)"
        print(f"H = {hurst:.4f} ({hurst_info}) → {status}")
    else:
        print(f"Could not compute: {hurst_info}")
    
    # Correlation Dimension
    print("  Computing correlation dimension...", end=" ", flush=True)
    corr_dim, dim_info = correlation_dimension(samples)
    results['corr_dim'] = corr_dim
    if not np.isnan(corr_dim):
        if corr_dim > 5:
            status = "HIGH-DIMENSIONAL (random-like)"
        elif corr_dim > 2:
            status = "MODERATE DIMENSION"
        else:
            status = "LOW-DIMENSIONAL ATTRACTOR"
        print(f"D2 = {corr_dim:.3f} ({dim_info}) → {status}")
    else:
        print(f"Could not compute: {dim_info}")
    
    # DFA
    print("  Computing DFA exponent...", end=" ", flush=True)
    dfa, dfa_info = detrended_fluctuation_analysis(samples)
    results['dfa'] = dfa
    if not np.isnan(dfa):
        if 0.45 < dfa < 0.55:
            status = "WHITE NOISE"
        elif dfa > 0.55:
            status = "LONG-RANGE CORRELATED"
        else:
            status = "ANTI-CORRELATED"
        print(f"α = {dfa:.4f} ({dfa_info}) → {status}")
    else:
        print(f"Could not compute: {dfa_info}")
    
    # Permutation Entropy
    print("  Computing permutation entropy...", end=" ", flush=True)
    pe, pe_info = permutation_entropy(samples)
    results['perm_entropy'] = pe
    if not np.isnan(pe):
        if pe > 0.95:
            status = "HIGHLY RANDOM"
        elif pe > 0.8:
            status = "MODERATELY COMPLEX"
        else:
            status = "STRUCTURED/DETERMINISTIC"
        print(f"PE = {pe:.4f} ({pe_info}) → {status}")
    else:
        print(f"Could not compute: {pe_info}")
    
    # Sample Entropy
    print("  Computing sample entropy...", end=" ", flush=True)
    samp_en, se_info = sample_entropy(samples)
    results['sample_entropy'] = samp_en
    if not np.isnan(samp_en):
        if samp_en > 2.0:
            status = "HIGHLY COMPLEX"
        elif samp_en > 1.0:
            status = "MODERATELY COMPLEX"
        else:
            status = "LOW COMPLEXITY"
        print(f"SampEn = {samp_en:.4f} ({se_info}) → {status}")
    else:
        print(f"Could not compute: {se_info}")
    
    # RQA
    print("  Computing RQA metrics...", end=" ", flush=True)
    rqa, rqa_info = recurrence_quantification(samples)
    results['rqa'] = rqa
    if rqa:
        print(f"RR={rqa['RR']:.4f}, DET={rqa['DET']:.4f}, LAM={rqa['LAM']:.4f}")
        if rqa['DET'] < 0.3:
            print(f"    → LOW DETERMINISM (random-like)")
        elif rqa['DET'] > 0.7:
            print(f"    → HIGH DETERMINISM (structured)")
        else:
            print(f"    → MODERATE DETERMINISM")
    else:
        print(f"Could not compute: {rqa_info}")
    
    return results


def interpret_results(all_results):
    """Interpret chaos analysis results across sources."""
    print("\n" + "="*80)
    print("CHAOS THEORY INTERPRETATION")
    print("="*80)
    
    print("\n" + "─"*80)
    print("SUMMARY TABLE")
    print("─"*80)
    
    header = f"{'Source':<30} {'λ':>8} {'H':>8} {'D2':>8} {'α':>8} {'PE':>8} {'DET':>8}"
    print(header)
    print("─"*80)
    
    for source, results in all_results.items():
        if 'unknown' in source.lower():
            continue
        
        name = source[:29]
        lyap = f"{results.get('lyapunov', np.nan):.3f}" if not np.isnan(results.get('lyapunov', np.nan)) else "N/A"
        hurst = f"{results.get('hurst', np.nan):.3f}" if not np.isnan(results.get('hurst', np.nan)) else "N/A"
        dim = f"{results.get('corr_dim', np.nan):.2f}" if not np.isnan(results.get('corr_dim', np.nan)) else "N/A"
        dfa = f"{results.get('dfa', np.nan):.3f}" if not np.isnan(results.get('dfa', np.nan)) else "N/A"
        pe = f"{results.get('perm_entropy', np.nan):.3f}" if not np.isnan(results.get('perm_entropy', np.nan)) else "N/A"
        det = f"{results.get('rqa', {}).get('DET', np.nan):.3f}" if results.get('rqa', {}).get('DET') else "N/A"
        
        print(f"{name:<30} {lyap:>8} {hurst:>8} {dim:>8} {dfa:>8} {pe:>8} {det:>8}")
    
    print("\n" + "─"*80)
    print("EXPECTED VALUES FOR TRUE RANDOM DATA:")
    print("─"*80)
    print("  λ (Lyapunov):    Large positive (exponential divergence)")
    print("  H (Hurst):       ≈ 0.5 (no memory/persistence)")
    print("  D2 (Corr Dim):   High/saturates (high-dimensional)")
    print("  α (DFA):         ≈ 0.5 (white noise)")
    print("  PE (Perm Ent):   → 1.0 (maximum disorder)")
    print("  DET (RQA):       Low (< 0.3, no deterministic structure)")
    
    print("\n" + "─"*80)
    print("ANALYSIS:")
    print("─"*80)
    
    # Analyze quantum vs classical
    quantum_sources = [s for s in all_results if any(q in s.lower() for q in ['outshift', 'anu', 'cipherstone']) and 'raw' not in s.lower()]
    classical_sources = [s for s in all_results if any(c in s.lower() for c in ['cpu', 'prng', 'mersenne'])]
    
    for category, sources in [("QUANTUM SOURCES", quantum_sources), ("CLASSICAL SOURCES", classical_sources)]:
        if sources:
            print(f"\n  {category}:")
            hursts = [all_results[s].get('hurst', np.nan) for s in sources]
            hursts = [h for h in hursts if not np.isnan(h)]
            
            pes = [all_results[s].get('perm_entropy', np.nan) for s in sources]
            pes = [p for p in pes if not np.isnan(p)]
            
            dfas = [all_results[s].get('dfa', np.nan) for s in sources]
            dfas = [d for d in dfas if not np.isnan(d)]
            
            if hursts:
                avg_h = np.mean(hursts)
                if 0.45 < avg_h < 0.55:
                    print(f"    Hurst: {avg_h:.3f} → ✓ True random (no memory)")
                else:
                    print(f"    Hurst: {avg_h:.3f} → ⚠️ Deviation from random")
            
            if pes:
                avg_pe = np.mean(pes)
                if avg_pe > 0.95:
                    print(f"    Permutation Entropy: {avg_pe:.3f} → ✓ Maximum randomness")
                else:
                    print(f"    Permutation Entropy: {avg_pe:.3f} → ⚠️ Some structure detected")
            
            if dfas:
                avg_dfa = np.mean(dfas)
                if 0.45 < avg_dfa < 0.55:
                    print(f"    DFA: {avg_dfa:.3f} → ✓ White noise")
                else:
                    print(f"    DFA: {avg_dfa:.3f} → ⚠️ Scaling anomaly")


def main():
    print("="*80)
    print("CHAOS THEORY ANALYSIS - QRNG DATA")
    print("="*80)
    print(f"Analysis Date: {datetime.now().strftime('%B %d, %Y %H:%M')}")
    print()
    
    # Load data
    sources = load_all_streams()
    
    total = sum(len(s) for s in sources.values())
    print(f"Loaded {total:,} samples from {len(sources)} sources")
    
    for source, samples in sorted(sources.items(), key=lambda x: -len(x[1])):
        print(f"  {source}: {len(samples):,} samples")
    
    # Run analysis on each source
    all_results = {}
    
    for source, samples in sources.items():
        if len(samples) >= 500:
            results = analyze_source(source, samples)
            all_results[source] = results
    
    # Interpretation
    interpret_results(all_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
