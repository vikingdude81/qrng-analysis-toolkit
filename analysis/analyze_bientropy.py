#!/usr/bin/env python3
"""
Run BiEntropy analysis on all QRNG sources.

Based on Croll (2013) "BiEntropy - The Approximate Entropy of a Finite Binary String"
"""

import json
import numpy as np
from pathlib import Path
from bientropy_metrics import bientropy_analysis

# Source name mappings
SOURCE_NAMES = {
    'outshift_qrng_api': 'Outshift SPDC',
    'anu_qrng_vacuum_fluctuation': 'ANU Vacuum',
    'cipherstone_qbert_conditioned': 'Cipherstone M1',
    'cipherstone_qbert_raw': 'Cipherstone M2 (Raw)',
    'cpu_hwrng_bcrypt': 'CPU RDRAND',
    'prng_mersenne_twister': 'PRNG (MT)'
}


def load_sources():
    """Load all QRNG streams grouped by source."""
    streams_dir = Path(__file__).parent / 'qrng_streams'
    sources = {}
    
    for f in sorted(streams_dir.glob('*.json')):
        try:
            with open(f) as fp:
                data = json.load(fp)
            source = data.get('source', 'unknown')
            floats = data.get('floats', [])
            if floats and source != 'unknown':
                if source not in sources:
                    sources[source] = []
                sources[source].extend(floats)
        except Exception:
            pass
    
    return {k: np.array(v) for k, v in sources.items()}


def main():
    print("=" * 80)
    print("BiEntropy Analysis of QRNG Sources")
    print("=" * 80)
    print()
    print("BiEntropy measures randomness through binary derivatives (Croll, 2013).")
    print("Higher values (~1.0) indicate more disorder/randomness.")
    print("TBiEn (logarithmic weighting) is more suitable for longer sequences.")
    print()
    
    sources = load_sources()
    
    if not sources:
        print("No QRNG streams found!")
        return
    
    total = sum(len(v) for v in sources.values())
    print(f"Loaded {total:,} samples from {len(sources)} sources\n")
    
    # Header
    header = "{:<25} {:>8} {:>8} {:>8} {:>8} {:>10}".format(
        'Source', 'N', 'TBiEn', 'BiEn', 'Deriv H', 'TBiEn Min'
    )
    print(header)
    print("-" * 75)
    
    results = {}
    for src, vals in sorted(sources.items(), key=lambda x: -len(x[1])):
        name = SOURCE_NAMES.get(src, src)[:24]
        analysis = bientropy_analysis(vals)
        results[src] = analysis
        
        row = "{:<25} {:>8,} {:>8.4f} {:>8.4f} {:>8.4f} {:>10.4f}".format(
            name,
            len(vals),
            analysis['global_tbien_mean'],
            analysis['global_bien_mean'],
            analysis['derivative_entropy_mean'],
            analysis['window_tbien_min']
        )
        print(row)
    
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("TBiEn (TBiEntropy):")
    print("  > 0.95: EXCELLENT - True random behavior")
    print("  0.90-0.95: GOOD - Minor patterns possible")
    print("  < 0.90: WARNING - Potential patterns detected")
    print()
    print("BiEn (BiEntropy):")
    print("  Measures short-range structure (32-byte chunks)")
    print("  Higher variance is normal for chunked analysis")
    print()
    print("Deriv H (Derivative Entropy):")
    print("  ~1.0: Binary derivatives appear random")
    print("  <0.9: Patterns persist through derivatives")
    print()
    print("TBiEn Min (Minimum Window):")
    print("  Lowest TBiEn in any 64-byte window")
    print("  Detects localized patterns")
    
    # Summary rating
    print()
    print("=" * 80)
    print("QUALITY RATING (by TBiEn)")
    print("=" * 80)
    print()
    
    for src, analysis in sorted(results.items(), 
                                 key=lambda x: -x[1]['global_tbien_mean']):
        name = SOURCE_NAMES.get(src, src)[:30]
        tbien = analysis['global_tbien_mean']
        
        if tbien >= 0.97:
            rating = "EXCELLENT"
        elif tbien >= 0.95:
            rating = "VERY GOOD"
        elif tbien >= 0.90:
            rating = "GOOD"
        elif tbien >= 0.80:
            rating = "FAIR"
        else:
            rating = "POOR"
        
        print(f"  {name:<30} TBiEn={tbien:.4f}  [{rating}]")


if __name__ == "__main__":
    main()
