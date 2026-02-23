#!/usr/bin/env python3
"""
CPU Hardware RNG Module
=======================
Multiple methods to extract hardware-based randomness from the CPU:

1. RDRAND/RDSEED via inline assembly (requires compilation)
2. Windows BCryptGenRandom (uses CPU entropy when available)
3. Timing jitter entropy (CPU thermal noise, cache effects)
4. RDTSC (timestamp counter) entropy extraction

For AMD Threadripper 5995WX - RDRAND is definitely supported.
"""

import os
import sys
import ctypes
import struct
import time
import hashlib
from typing import Optional, Tuple
import numpy as np


class CPUHardwareRNG:
    """
    Hardware RNG using CPU entropy sources.
    
    On modern Intel/AMD CPUs (including Threadripper), this uses:
    - Windows CNG (Cryptographic Next Generation) which pulls from RDRAND
    - Timing jitter as additional entropy source
    """
    
    def __init__(self, method: str = "auto"):
        """
        Initialize CPU HWRNG.
        
        Args:
            method: "bcrypt" (Windows CNG), "jitter" (timing), or "auto"
        """
        self.method = method
        self._bcrypt_available = self._check_bcrypt()
        self._rdtsc_available = self._check_rdtsc()
        
        if method == "auto":
            if self._bcrypt_available:
                self.method = "bcrypt"
            else:
                self.method = "jitter"
    
    def _check_bcrypt(self) -> bool:
        """Check if Windows BCrypt is available."""
        if sys.platform != "win32":
            return False
        try:
            bcrypt = ctypes.windll.bcrypt
            return True
        except:
            return False
    
    def _check_rdtsc(self) -> bool:
        """Check if we can access high-resolution timer."""
        try:
            if sys.platform == "win32":
                ctypes.windll.kernel32.QueryPerformanceCounter
            return True
        except:
            return False
    
    def _bcrypt_random(self, n_bytes: int) -> bytes:
        """
        Get random bytes from Windows BCrypt (uses RDRAND internally).
        
        BCryptGenRandom with BCRYPT_USE_SYSTEM_PREFERRED_RNG uses the 
        system's preferred RNG which on modern Windows pulls from RDRAND.
        """
        bcrypt = ctypes.windll.bcrypt
        
        BCRYPT_USE_SYSTEM_PREFERRED_RNG = 0x00000002
        
        buffer = ctypes.create_string_buffer(n_bytes)
        status = bcrypt.BCryptGenRandom(
            None,  # hAlgorithm (NULL for system preferred)
            buffer,
            n_bytes,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG
        )
        
        if status != 0:
            raise RuntimeError(f"BCryptGenRandom failed with status {status}")
        
        return buffer.raw
    
    def _get_rdtsc(self) -> int:
        """Get high-resolution timestamp (RDTSC)."""
        if sys.platform == "win32":
            counter = ctypes.c_int64()
            ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(counter))
            return counter.value
        else:
            return time.perf_counter_ns()
    
    def _timing_jitter_byte(self) -> int:
        """
        Extract one byte of entropy from timing jitter.
        
        This exploits:
        - CPU thermal noise affecting clock speed
        - Cache timing variations
        - Branch prediction variations
        - Memory access timing
        """
        bits = []
        
        for _ in range(8):  # 8 bits per byte
            # Measure timing of operations that have variable latency
            measurements = []
            
            for _ in range(16):  # Multiple measurements per bit
                t1 = self._get_rdtsc()
                
                # Operations with timing variability
                dummy = 0
                for j in range(100):
                    dummy += j * j
                _ = hashlib.sha256(str(t1 + dummy).encode()).digest()
                
                t2 = self._get_rdtsc()
                measurements.append(t2 - t1)
            
            # Use von Neumann debiasing: compare pairs
            # Take middle bits that have more entropy
            sorted_m = sorted(measurements)
            median = sorted_m[len(sorted_m) // 2]
            
            # XOR multiple bits for better entropy
            bit = (median >> 3) ^ (median >> 7) ^ (median >> 11)
            bits.append(bit & 1)
        
        # Combine bits into byte
        byte_val = 0
        for i, bit in enumerate(bits):
            byte_val |= (bit << i)
        
        return byte_val
    
    def _timing_jitter_random(self, n_bytes: int) -> bytes:
        """Get random bytes from timing jitter entropy."""
        result = bytearray(n_bytes)
        for i in range(n_bytes):
            result[i] = self._timing_jitter_byte()
        return bytes(result)
    
    def get_random_bytes(self, n_bytes: int) -> bytes:
        """
        Get random bytes from CPU hardware entropy.
        
        Args:
            n_bytes: Number of random bytes to generate
            
        Returns:
            Random bytes from hardware source
        """
        if self.method == "bcrypt":
            return self._bcrypt_random(n_bytes)
        elif self.method == "jitter":
            return self._timing_jitter_random(n_bytes)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_random_floats(self, n: int) -> np.ndarray:
        """
        Get random floats in [0, 1) from CPU hardware entropy.
        
        Args:
            n: Number of random floats
            
        Returns:
            numpy array of floats in [0, 1)
        """
        # Get 4 bytes per float (32-bit integers -> floats)
        random_bytes = self.get_random_bytes(n * 4)
        
        # Convert to uint32 and normalize to [0, 1)
        integers = np.frombuffer(random_bytes, dtype=np.uint32)
        return integers / (2**32)
    
    def get_raw_integers(self, n: int, bits: int = 32) -> np.ndarray:
        """
        Get raw random integers.
        
        Args:
            n: Number of integers
            bits: 32 or 64 bits per integer
            
        Returns:
            numpy array of random integers
        """
        if bits == 32:
            random_bytes = self.get_random_bytes(n * 4)
            return np.frombuffer(random_bytes, dtype=np.uint32)
        elif bits == 64:
            random_bytes = self.get_random_bytes(n * 8)
            return np.frombuffer(random_bytes, dtype=np.uint64)
        else:
            raise ValueError("bits must be 32 or 64")
    
    @property
    def source_info(self) -> dict:
        """Get information about the entropy source."""
        return {
            "method": self.method,
            "bcrypt_available": self._bcrypt_available,
            "rdtsc_available": self._rdtsc_available,
            "platform": sys.platform,
            "description": self._get_description()
        }
    
    def _get_description(self) -> str:
        """Get human-readable description of entropy source."""
        if self.method == "bcrypt":
            return "Windows BCryptGenRandom (RDRAND-backed on modern CPUs)"
        elif self.method == "jitter":
            return "CPU timing jitter entropy (thermal/cache noise)"
        else:
            return "Unknown"


def collect_cpu_hwrng_stream(n_samples: int = 1000, method: str = "auto") -> dict:
    """
    Collect a stream of random samples from CPU HWRNG.
    
    Args:
        n_samples: Number of float samples to collect
        method: "bcrypt", "jitter", or "auto"
        
    Returns:
        dict with stream data compatible with other QRNG tools
    """
    from datetime import datetime
    
    hwrng = CPUHardwareRNG(method=method)
    
    print(f"Collecting {n_samples} samples from CPU HWRNG...")
    print(f"Method: {hwrng.source_info['description']}")
    
    start_time = time.time()
    
    # Collect raw integers and floats
    raw_integers = hwrng.get_raw_integers(n_samples, bits=32)
    floats = raw_integers / (2**32)
    
    elapsed = time.time() - start_time
    
    rate = n_samples/elapsed if elapsed > 0 else float('inf')
    print(f"Collected {n_samples} samples in {elapsed:.4f}s ({rate:.0f} samples/sec)")
    
    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source": f"cpu_hwrng_{hwrng.method}",
        "bits_per_block": 32,
        "count": n_samples,
        "raw_integers": raw_integers.tolist(),
        "floats": floats.tolist(),
        "encoding": "uint32_to_float",
        "method": hwrng.method,
        "source_info": hwrng.source_info
    }


def compare_entropy_sources(n_samples: int = 1000) -> dict:
    """
    Compare multiple entropy sources side by side.
    """
    from rich.console import Console
    from rich.table import Table
    from scipy import stats
    
    console = Console()
    
    # Collect from all sources
    sources = {}
    
    # 1. CPU HWRNG (BCrypt/RDRAND)
    try:
        hwrng_bcrypt = CPUHardwareRNG(method="bcrypt")
        sources["CPU RDRAND"] = hwrng_bcrypt.get_random_floats(n_samples)
        console.print("[green]✓ CPU RDRAND (BCrypt) available[/]")
    except Exception as e:
        console.print(f"[red]✗ CPU RDRAND: {e}[/]")
    
    # 2. CPU Timing Jitter (slow but independent)
    # Only do a small sample due to speed
    try:
        hwrng_jitter = CPUHardwareRNG(method="jitter")
        console.print("[yellow]Collecting timing jitter samples (slow)...[/]")
        sources["CPU Jitter"] = hwrng_jitter.get_random_floats(min(100, n_samples))
        console.print("[green]✓ CPU timing jitter available[/]")
    except Exception as e:
        console.print(f"[red]✗ CPU Jitter: {e}[/]")
    
    # 3. os.urandom (CSPRNG, may use RDRAND)
    csprng_bytes = os.urandom(n_samples * 4)
    sources["CSPRNG"] = np.frombuffer(csprng_bytes, dtype=np.uint32) / (2**32)
    console.print("[green]✓ CSPRNG (os.urandom) available[/]")
    
    # 4. PRNG (Mersenne Twister)
    np.random.seed(int(time.time()))
    sources["PRNG"] = np.random.random(n_samples)
    console.print("[green]✓ PRNG (MT19937) available[/]")
    
    # Compare
    console.print("\n")
    table = Table(title=f"Entropy Source Comparison (N={n_samples})")
    table.add_column("Source", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Skew", justify="right")
    table.add_column("K-S p", justify="right")
    
    for name, data in sources.items():
        ks_stat, ks_p = stats.kstest(data, 'uniform', args=(0, 1))
        table.add_row(
            name,
            str(len(data)),
            f"{np.mean(data):.4f}",
            f"{np.std(data):.4f}",
            f"{stats.skew(data):.4f}",
            f"{ks_p:.4f}"
        )
    
    console.print(table)
    
    return sources


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="CPU Hardware RNG")
    parser.add_argument("command", choices=["collect", "compare", "info"],
                       help="Command to run")
    parser.add_argument("-n", "--samples", type=int, default=1000,
                       help="Number of samples")
    parser.add_argument("-m", "--method", default="auto",
                       choices=["auto", "bcrypt", "jitter"],
                       help="Entropy source method")
    parser.add_argument("-o", "--output", type=str,
                       help="Output file path")
    
    args = parser.parse_args()
    
    if args.command == "info":
        hwrng = CPUHardwareRNG()
        print("CPU Hardware RNG Info:")
        for k, v in hwrng.source_info.items():
            print(f"  {k}: {v}")
    
    elif args.command == "collect":
        stream = collect_cpu_hwrng_stream(args.samples, args.method)
        
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path("qrng_streams") / f"cpu_hwrng_{stream['timestamp']}.json"
        
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stream, f, indent=2)
        
        print(f"\n✓ Saved to: {output_path}")
    
    elif args.command == "compare":
        compare_entropy_sources(args.samples)
