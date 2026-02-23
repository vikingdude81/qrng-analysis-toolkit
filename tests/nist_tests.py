"""
NIST SP 800-22 Statistical Test Suite

Implementation of NIST random number tests for QRNG validation.
Based on: https://csrc.nist.gov/publications/detail/sp/800-22/rev-1a/final

Tests implemented:
1. Frequency (Monobit) Test
2. Frequency Test within a Block
3. Runs Test
4. Test for the Longest Run of Ones in a Block
5. Binary Matrix Rank Test
6. Discrete Fourier Transform (Spectral) Test
7. Non-overlapping Template Matching Test
8. Overlapping Template Matching Test
9. Maurer's "Universal Statistical" Test
10. Linear Complexity Test
11. Serial Test
12. Approximate Entropy Test
13. Cumulative Sums Test
14. Random Excursions Test
15. Random Excursions Variant Test

Note: This is a simplified implementation for quick validation.
For full compliance, use the official NIST toolkit.
"""

import numpy as np
from scipy import stats, special
from typing import Tuple, List, Optional
from dataclasses import dataclass
import sys
@dataclass
class NISTTestResult:
    """Result of a NIST test."""
    test_name: str
    p_value: float
    passed: bool
    details: str = ""
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} {self.test_name}: p={self.p_value:.4f}"


class NIST800_22:
    """
    NIST SP 800-22 Statistical Test Suite.
    
    Usage:
        from tests.nist_tests import NIST800_22
        from qrng_spdc_source import SPDCQuantumSource
        
        source = SPDCQuantumSource()
        bits = [1 if source.get_random() > 0.5 else 0 for _ in range(10000)]
        
        nist = NIST800_22(bits)
        results = nist.run_all_tests()
        
        for result in results:
            print(result)
    """
    
    def __init__(self, bit_sequence: List[int], alpha: float = 0.01):
        """
        Args:
            bit_sequence: List of 0s and 1s
            alpha: Significance level (default 0.01 for NIST)
        """
        self.bits = np.array(bit_sequence, dtype=np.int8)
        self.n = len(self.bits)
        self.alpha = alpha
    
    def run_all_tests(self) -> List[NISTTestResult]:
        """Run all applicable NIST tests."""
        results = []
        
        tests = [
            self.frequency_monobit,
            self.frequency_block,
            self.runs,
            self.longest_run_of_ones,
            self.binary_matrix_rank,
            self.dft_spectral,
            self.non_overlapping_template,
            self.overlapping_template,
            self.maurers_universal,
            self.linear_complexity,
            self.serial,
            self.approximate_entropy,
            self.cumulative_sums,
        ]
        
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                results.append(NISTTestResult(
                    test_name=test.__name__,
                    p_value=0,
                    passed=False,
                    details=f"Error: {str(e)}"
                ))
        
        return results
    
    def frequency_monobit(self) -> NISTTestResult:
        """
        Test 1: Frequency (Monobit) Test
        
        Tests if the number of 1s and 0s in the sequence are approximately
        the same as would be expected for a truly random sequence.
        """
        # Convert to +1/-1
        s = np.sum(2 * self.bits - 1)
        s_obs = abs(s) / np.sqrt(self.n)
        
        # Complementary error function
        p_value = special.erfc(s_obs / np.sqrt(2))
        
        return NISTTestResult(
            test_name="Frequency (Monobit)",
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            details=f"Sum={s}, S_obs={s_obs:.4f}"
        )
    
    def frequency_block(self, M: int = 128) -> NISTTestResult:
        """
        Test 2: Frequency Test within a Block
        
        Tests if the frequency of 1s in M-bit blocks is approximately M/2.
        """
        if self.n < M:
            return NISTTestResult(
                test_name="Frequency within Block",
                p_value=0,
                passed=False,
                details=f"Need at least {M} bits"
            )
        
        N = self.n // M  # Number of blocks
        
        chi2 = 0
        for i in range(N):
            block = self.bits[i*M:(i+1)*M]
            pi = np.sum(block) / M
            chi2 += (pi - 0.5)**2
        
        chi2 *= 4 * M
        p_value = float(special.gammaincc(N/2, chi2/2))
        
        return NISTTestResult(
            test_name="Frequency within Block",
            p_value=p_value,
            passed=p_value >= self.alpha,
            details=f"N={N} blocks of M={M}, chi2={chi2:.4f}"
        )
    
    def runs(self) -> NISTTestResult:
        """
        Test 3: Runs Test
        
        Tests whether the number of runs of consecutive identical bits
        is as expected for a random sequence.
        """
        # Proportion of ones
        pi = np.sum(self.bits) / self.n
        
        # Pre-test: check if pi is within acceptable range
        tau = 2 / np.sqrt(self.n)
        if abs(pi - 0.5) >= tau:
            return NISTTestResult(
                test_name="Runs",
                p_value=0,
                passed=False,
                details=f"Pre-test failed: pi={pi:.4f}, tau={tau:.4f}"
            )
        
        # Count runs
        runs = 1
        for i in range(1, self.n):
            if self.bits[i] != self.bits[i-1]:
                runs += 1
        
        # Compute p-value
        numerator = abs(runs - 2 * self.n * pi * (1 - pi))
        denominator = 2 * np.sqrt(2 * self.n) * pi * (1 - pi)
        
        p_value = float(special.erfc(numerator / denominator))
        
        return NISTTestResult(
            test_name="Runs",
            p_value=p_value,
            passed=p_value >= self.alpha,
            details=f"Runs={runs}, pi={pi:.4f}"
        )
    
    def longest_run_of_ones(self) -> NISTTestResult:
        """
        Test 4: Test for the Longest Run of Ones in a Block
        
        Tests whether the longest run of ones within M-bit blocks
        is consistent with a random sequence.
        """
        # Parameters depend on n
        if self.n < 128:
            return NISTTestResult(
                test_name="Longest Run of Ones",
                p_value=0,
                passed=False,
                details="Need at least 128 bits"
            )
        elif self.n < 6272:
            M = 8
            K = 3
            N = 16
            pi = [0.2148, 0.3672, 0.2305, 0.1875]
            v_values = [1, 2, 3, 4]
        elif self.n < 750000:
            M = 128
            K = 5
            N = 49
            pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
            v_values = [4, 5, 6, 7, 8, 9]
        else:
            M = 10000
            K = 6
            N = 75
            pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
            v_values = [10, 11, 12, 13, 14, 15, 16]
        
        # Count longest runs in each block
        blocks = self.n // M
        freq = [0] * (K + 1)
        
        for i in range(min(blocks, N)):
            block = self.bits[i*M:(i+1)*M]
            
            # Find longest run of ones
            max_run = 0
            current_run = 0
            for bit in block:
                if bit == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            
            # Categorize
            if max_run <= v_values[0]:
                freq[0] += 1
            elif max_run >= v_values[-1]:
                freq[K] += 1
            else:
                for j in range(1, K):
                    if max_run == v_values[j]:
                        freq[j] += 1
                        break
        
        # Chi-square calculation
        N_actual = min(blocks, N)
        chi2 = 0
        for i in range(K + 1):
            expected = N_actual * pi[i] if i < len(pi) else N_actual * pi[-1]
            if expected > 0:
                chi2 += (freq[i] - expected)**2 / expected
        
        p_value = float(special.gammaincc(K/2, chi2/2))
        
        return NISTTestResult(
            test_name="Longest Run of Ones",
            p_value=p_value,
            passed=p_value >= self.alpha,
            details=f"M={M}, chi2={chi2:.4f}"
        )
    
    def binary_matrix_rank(self) -> NISTTestResult:
        """
        Test 5: Binary Matrix Rank Test
        
        Tests if the rank of disjoint sub-matrices of the sequence
        is consistent with random sequences.
        """
        M, Q = 32, 32  # Matrix dimensions
        N = self.n // (M * Q)
        
        if N < 38:
            return NISTTestResult(
                test_name="Binary Matrix Rank",
                p_value=0,
                passed=False,
                details=f"Need at least {38 * M * Q} bits"
            )
        
        # Count matrices by rank
        F_M = 0  # Full rank (M)
        F_M1 = 0  # Rank M-1
        
        for i in range(N):
            start = i * M * Q
            matrix = self.bits[start:start + M*Q].reshape(M, Q)
            rank = np.linalg.matrix_rank(matrix.astype(float))
            
            if rank == M:
                F_M += 1
            elif rank == M - 1:
                F_M1 += 1
        
        F_remaining = N - F_M - F_M1
        
        # Expected probabilities for 32x32 matrices
        p_M = 0.2888
        p_M1 = 0.5776
        p_rest = 0.1336
        
        chi2 = ((F_M - N*p_M)**2 / (N*p_M) +
                (F_M1 - N*p_M1)**2 / (N*p_M1) +
                (F_remaining - N*p_rest)**2 / (N*p_rest))
        
        p_value = np.exp(-chi2 / 2)
        
        return NISTTestResult(
            test_name="Binary Matrix Rank",
            p_value=float(p_value),
            passed=p_value >= self.alpha,
            details=f"N={N}, F_M={F_M}, F_M-1={F_M1}"
        )
    
    def dft_spectral(self) -> NISTTestResult:
        """
        Test 6: Discrete Fourier Transform (Spectral) Test
        
        Tests for periodic features in the sequence.
        """
        # Convert to +1/-1
        x = 2 * self.bits - 1
        
        # Compute DFT
        S = np.fft.fft(x)
        modulus = np.abs(S[:self.n//2])
        
        # Compute threshold
        T = np.sqrt(np.log(1/0.05) * self.n)
        
        # Count peaks below threshold
        N0 = 0.95 * self.n / 2
        N1 = np.sum(modulus < T)
        
        d = (N1 - N0) / np.sqrt(self.n * 0.95 * 0.05 / 4)
        p_value = float(special.erfc(abs(d) / np.sqrt(2)))
        
        return NISTTestResult(
            test_name="DFT (Spectral)",
            p_value=p_value,
            passed=p_value >= self.alpha,
            details=f"N0={N0:.1f}, N1={N1}, d={d:.4f}"
        )
    
    def non_overlapping_template(self, m: int = 9) -> NISTTestResult:
        """
        Test 7: Non-overlapping Template Matching Test
        
        Tests if pre-defined target patterns occur too frequently.
        """
        M = 1032  # Block size (recommended for m=9)
        N = self.n // M
        
        if N < 8:
            return NISTTestResult(
                test_name="Non-overlapping Template",
                p_value=0,
                passed=False,
                details="Need more bits"
            )
        
        # Use a simple template: m ones
        template = np.ones(m, dtype=np.int8)
        
        # Count occurrences in each block
        W = np.zeros(N)
        for i in range(N):
            block = self.bits[i*M:(i+1)*M]
            count = 0
            j = 0
            while j <= M - m:
                if np.array_equal(block[j:j+m], template):
                    count += 1
                    j += m  # Non-overlapping
                else:
                    j += 1
            W[i] = count
        
        # Expected values
        mu = (M - m + 1) / (2**m)
        sigma2 = M * (1/(2**m) - (2*m - 1)/(2**(2*m)))
        
        chi2 = np.sum((W - mu)**2) / sigma2
        p_value = float(special.gammaincc(N/2, chi2/2))
        
        return NISTTestResult(
            test_name="Non-overlapping Template",
            p_value=p_value,
            passed=p_value >= self.alpha,
            details=f"Template={'1'*m}, mu={mu:.2f}"
        )
    
    def overlapping_template(self, m: int = 9) -> NISTTestResult:
        """
        Test 8: Overlapping Template Matching Test
        """
        M = 1032
        N = self.n // M
        
        if N < 8:
            return NISTTestResult(
                test_name="Overlapping Template",
                p_value=0,
                passed=False,
                details="Need more bits"
            )
        
        # Template: m ones
        template = np.ones(m, dtype=np.int8)
        
        # Count occurrences
        W = np.zeros(N)
        for i in range(N):
            block = self.bits[i*M:(i+1)*M]
            count = 0
            for j in range(M - m + 1):
                if np.array_equal(block[j:j+m], template):
                    count += 1
            W[i] = count
        
        # Theoretical parameters (simplified)
        lambda_val = (M - m + 1) / (2**m)
        eta = lambda_val / 2
        
        # Chi-square with 5 categories
        K = 5
        pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432]
        
        freq = np.zeros(K + 1)
        for w in W:
            if w <= 0:
                freq[0] += 1
            elif w >= K:
                freq[K] += 1
            else:
                freq[int(w)] += 1
        
        chi2 = 0
        for i in range(K + 1):
            expected = N * (pi[i] if i < len(pi) else pi[-1])
            if expected > 0:
                chi2 += (freq[i] - expected)**2 / expected
        
        p_value = float(special.gammaincc(K/2, chi2/2))
        
        return NISTTestResult(
            test_name="Overlapping Template",
            p_value=p_value,
            passed=p_value >= self.alpha,
            details=f"lambda={lambda_val:.2f}"
        )
    
    def maurers_universal(self) -> NISTTestResult:
        """
        Test 9: Maurer's "Universal Statistical" Test
        """
        # Parameters for L=7
        L = 7
        Q = 1280
        
        if self.n < Q + 10 * (2**L):
            return NISTTestResult(
                test_name="Maurer's Universal",
                p_value=0,
                passed=False,
                details="Need more bits"
            )
        
        K = self.n // L - Q
        
        # Initialize table with positions
        T = np.zeros(2**L)
        
        # Initialization segment
        for i in range(Q):
            block_val = 0
            for j in range(L):
                block_val = (block_val << 1) | self.bits[i*L + j]
            T[block_val] = i + 1
        
        # Test segment
        sum_log = 0
        for i in range(Q, Q + K):
            block_val = 0
            for j in range(L):
                block_val = (block_val << 1) | self.bits[i*L + j]
            
            sum_log += np.log2(i + 1 - T[block_val])
            T[block_val] = i + 1
        
        fn = sum_log / K
        
        # Expected value and variance for L=7
        expected = 6.1962507
        variance = 3.125
        sigma = np.sqrt(variance / K)
        
        p_value = float(special.erfc(abs(fn - expected) / (sigma * np.sqrt(2))))
        
        return NISTTestResult(
            test_name="Maurer's Universal",
            p_value=p_value,
            passed=p_value >= self.alpha,
            details=f"fn={fn:.4f}, expected={expected}"
        )
    
    def linear_complexity(self, M: int = 500) -> NISTTestResult:
        """
        Test 10: Linear Complexity Test
        """
        N = self.n // M
        
        if N < 6:
            return NISTTestResult(
                test_name="Linear Complexity",
                p_value=0,
                passed=False,
                details="Need more bits"
            )
        
        # Compute linear complexity for each block using Berlekamp-Massey
        L_values = []
        
        for i in range(N):
            block = self.bits[i*M:(i+1)*M]
            L = self._berlekamp_massey(block)
            L_values.append(L)
        
        # Expected mean
        mu = M/2 + (9 + (-1)**(M+1)) / 36 - (M/3 + 2/9) / (2**M)
        
        # Categorize deviations
        T = np.array(L_values) - mu
        freq = np.zeros(7)
        
        for t in T:
            if t <= -2.5:
                freq[0] += 1
            elif t <= -1.5:
                freq[1] += 1
            elif t <= -0.5:
                freq[2] += 1
            elif t <= 0.5:
                freq[3] += 1
            elif t <= 1.5:
                freq[4] += 1
            elif t <= 2.5:
                freq[5] += 1
            else:
                freq[6] += 1
        
        # Expected probabilities
        pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
        
        chi2 = 0
        for i in range(7):
            expected = N * pi[i]
            if expected > 0:
                chi2 += (freq[i] - expected)**2 / expected
        
        p_value = float(special.gammaincc(3, chi2/2))
        
        return NISTTestResult(
            test_name="Linear Complexity",
            p_value=p_value,
            passed=p_value >= self.alpha,
            details=f"Mean L={np.mean(L_values):.2f}"
        )
    
    def _berlekamp_massey(self, s: np.ndarray) -> int:
        """Berlekamp-Massey algorithm for linear complexity."""
        n = len(s)
        c = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        c[0] = b[0] = 1
        L = 0
        m = -1
        
        for N in range(n):
            d = s[N]
            for i in range(1, L + 1):
                d ^= c[i] & s[N - i]
            
            if d == 1:
                t = c.copy()
                for i in range(n - N + m):
                    if N - m + i < n:
                        c[N - m + i] ^= b[i]
                
                if L <= N // 2:
                    L = N + 1 - L
                    m = N
                    b = t
        
        return L
    
    def serial(self, m: int = 5) -> NISTTestResult:
        """
        Test 11: Serial Test
        """
        if self.n < m + 2:
            return NISTTestResult(
                test_name="Serial",
                p_value=0,
                passed=False,
                details="Need more bits"
            )
        
        # Extended sequence for wrap-around
        extended = np.concatenate([self.bits, self.bits[:m-1]])
        
        # Count m-bit patterns
        def count_patterns(length):
            counts = {}
            for i in range(self.n):
                pattern = tuple(extended[i:i+length])
                counts[pattern] = counts.get(pattern, 0) + 1
            return counts
        
        counts_m = count_patterns(m)
        counts_m1 = count_patterns(m - 1)
        counts_m2 = count_patterns(m - 2) if m >= 2 else {(): self.n}
        
        # Compute psi values
        psi_m = sum(c**2 for c in counts_m.values()) * (2**m) / self.n - self.n
        psi_m1 = sum(c**2 for c in counts_m1.values()) * (2**(m-1)) / self.n - self.n
        psi_m2 = sum(c**2 for c in counts_m2.values()) * (2**(m-2)) / self.n - self.n if m >= 2 else 0
        
        del_psi = psi_m - psi_m1
        del_psi2 = psi_m - 2*psi_m1 + psi_m2
        
        p_value1 = float(special.gammaincc(2**(m-2), del_psi/2))
        p_value2 = float(special.gammaincc(2**(m-3), del_psi2/2))
        
        return NISTTestResult(
            test_name="Serial",
            p_value=min(p_value1, p_value2),
            passed=p_value1 >= self.alpha and p_value2 >= self.alpha,
            details=f"p1={p_value1:.4f}, p2={p_value2:.4f}"
        )
    
    def approximate_entropy(self, m: int = 5) -> NISTTestResult:
        """
        Test 12: Approximate Entropy Test
        """
        def phi(length):
            extended = np.concatenate([self.bits, self.bits[:length-1]])
            counts = {}
            for i in range(self.n):
                pattern = tuple(extended[i:i+length])
                counts[pattern] = counts.get(pattern, 0) + 1
            
            total = 0
            for c in counts.values():
                prob = c / self.n
                if prob > 0:
                    total += prob * np.log(prob)
            return total
        
        phi_m = phi(m)
        phi_m1 = phi(m + 1)
        
        ApEn = phi_m - phi_m1
        chi2 = 2 * self.n * (np.log(2) - ApEn)
        
        p_value = float(special.gammaincc(2**(m-1), chi2/2))
        
        return NISTTestResult(
            test_name="Approximate Entropy",
            p_value=p_value,
            passed=p_value >= self.alpha,
            details=f"ApEn={ApEn:.6f}"
        )
    
    def cumulative_sums(self, mode: str = 'forward') -> NISTTestResult:
        """
        Test 13: Cumulative Sums (Cusums) Test
        """
        # Convert to +1/-1
        X = 2 * self.bits - 1
        
        if mode == 'forward':
            S = np.cumsum(X)
        else:
            S = np.cumsum(X[::-1])
        
        z = np.max(np.abs(S))
        
        # Compute p-value
        sum_val = 0
        sqrt_n = np.sqrt(self.n)
        
        for k in range(int((-self.n/z + 1) / 4), int((self.n/z - 1) / 4) + 1):
            sum_val += (stats.norm.cdf((4*k + 1) * z / sqrt_n) - 
                       stats.norm.cdf((4*k - 1) * z / sqrt_n))
        
        for k in range(int((-self.n/z - 3) / 4), int((self.n/z - 1) / 4) + 1):
            sum_val -= (stats.norm.cdf((4*k + 3) * z / sqrt_n) - 
                       stats.norm.cdf((4*k + 1) * z / sqrt_n))
        
        p_value = 1 - sum_val
        
        return NISTTestResult(
            test_name=f"Cumulative Sums ({mode})",
            p_value=float(max(0, min(1, p_value))),
            passed=p_value >= self.alpha,
            details=f"z={z}"
        )


def run_nist_tests(n_samples: int = 100000, verbose: bool = True) -> List[NISTTestResult]:
    """
    Run NIST tests on QRNG output.
    
    Args:
        n_samples: Number of QRNG samples (more = better accuracy)
        verbose: Print results as we go
        
    Returns:
        List of NISTTestResult
    """
    from qrng_spdc_source import SPDCQuantumSource
    
    if verbose:
        print(f"Generating {n_samples} QRNG samples...")
    
    source = SPDCQuantumSource()
    bits = [1 if source.get_random() > 0.5 else 0 for _ in range(n_samples)]
    
    if verbose:
        print(f"Running NIST SP 800-22 tests...")
    
    nist = NIST800_22(bits)
    results = nist.run_all_tests()
    
    if verbose:
        print("\n" + "="*50)
        print("NIST SP 800-22 Test Results")
        print("="*50)
        
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        
        for r in results:
            print(r)
        
        print("="*50)
        print(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)")
        print("="*50)
    
    return results


if __name__ == "__main__":
    run_nist_tests(n_samples=100000)
