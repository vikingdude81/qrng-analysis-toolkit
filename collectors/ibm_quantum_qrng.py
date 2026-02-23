"""
IBM Quantum QRNG Source
=======================
Generate true quantum random numbers using IBM's superconducting qubit processors.

Mechanism: Hadamard gates place qubits into equal superposition |+⟩ = (|0⟩+|1⟩)/√2,
then measurement collapses each qubit to 0 or 1 with 50/50 probability.
Each "shot" of an N-qubit circuit produces N random bits from genuine quantum
mechanical collapse events on IBM's Heron-series processors.

This is fundamentally different from the other QRNG sources:
  - Outshift: SPDC photon pair detection (optical)
  - ANU: Vacuum fluctuation shot noise (optical)  
  - Cipherstone: QBert photonic chip (optical)
  - IBM Quantum: Superconducting transmon qubit collapse (microwave/cryogenic)

Requirements:
  pip install qiskit qiskit-ibm-runtime
  
  Save credentials:
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(token="YOUR_API_KEY", instance="YOUR_CRN")
"""

import json
import time
import struct
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def check_ibm_quantum_available() -> bool:
    """Check if Qiskit and IBM Quantum credentials are configured."""
    if not QISKIT_AVAILABLE:
        print("  Error: qiskit or qiskit-ibm-runtime not installed")
        print("  Install: pip install qiskit qiskit-ibm-runtime")
        return False
    try:
        service = QiskitRuntimeService()
        backends = service.backends(simulator=False, operational=True)
        if not backends:
            print("  Error: No operational IBM Quantum backends available")
            return False
        return True
    except Exception as e:
        print(f"  Error: IBM Quantum authentication failed: {e}")
        print("  Save credentials first:")
        print("    from qiskit_ibm_runtime import QiskitRuntimeService")
        print("    QiskitRuntimeService.save_account(token='...', instance='...')")
        return False


def build_qrng_circuit(n_qubits: int = 32) -> QuantumCircuit:
    """
    Build a quantum circuit for random number generation.
    
    Creates a circuit with Hadamard gates on all qubits followed by measurement.
    Each shot produces n_qubits random bits from quantum collapse.
    
    Args:
        n_qubits: Number of qubits (bits per shot). Default 32 for 32-bit integers.
        
    Returns:
        QuantumCircuit ready for execution
    """
    qc = QuantumCircuit(n_qubits)
    
    # Apply Hadamard to all qubits: |0⟩ → (|0⟩+|1⟩)/√2
    for i in range(n_qubits):
        qc.h(i)
    
    # Measure all qubits
    qc.measure_all()
    
    return qc


def bitstring_to_float(bitstring: str) -> float:
    """Convert a binary string to a float in [0, 1)."""
    n = int(bitstring, 2)
    max_val = 2 ** len(bitstring)
    return n / max_val


def collect_ibm_quantum(
    count: int = 1000,
    n_qubits: int = 32,
    backend_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """
    Collect quantum random numbers from IBM Quantum hardware.
    
    Args:
        count: Number of random numbers to generate (= number of shots)
        n_qubits: Bits per random number (default: 32)
        backend_name: Specific backend to use (default: least busy)
        output_dir: Directory to save results (default: qrng_streams/)
        
    Returns:
        Dict with collection results, or None on failure
    """
    if not QISKIT_AVAILABLE:
        print("  Error: Qiskit not installed")
        return None
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "qrng_streams"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Connect to IBM Quantum
        service = QiskitRuntimeService()
        
        # Select backend
        if backend_name:
            backend = service.backend(backend_name)
        else:
            backend = service.least_busy(simulator=False, operational=True)
        
        print(f"  Backend: {backend.name} ({backend.num_qubits} qubits)")
        
        # Build the QRNG circuit
        # Use min of requested qubits and available qubits
        actual_qubits = min(n_qubits, backend.num_qubits)
        qc = build_qrng_circuit(actual_qubits)
        
        # Transpile for the target backend
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_circuit = pm.run(qc)
        
        # Execute using SamplerV2
        print(f"  Submitting {count} shots on {actual_qubits} qubits...")
        sampler = SamplerV2(mode=backend)
        job = sampler.run([(isa_circuit,)], shots=count)
        job_id = job.job_id()
        print(f"  Job ID: {job_id}")
        
        # Wait for results
        print(f"  Waiting for quantum hardware execution...")
        start_time = time.time()
        result = job.result()
        elapsed = time.time() - start_time
        print(f"  Execution completed in {elapsed:.1f}s")
        
        # Extract bitstrings from the result
        pub_result = result[0]
        
        # Get the measurement data
        # SamplerV2 returns BitArray in data.<classical register name>
        # For measure_all(), the register is called 'meas'
        bit_array = pub_result.data.meas
        
        # Convert to bitstrings
        bitstrings = bit_array.get_bitstrings()
        
        # Convert bitstrings to floats and integers
        raw_integers = []
        floats = []
        
        for bs in bitstrings[:count]:
            integer_val = int(bs, 2)
            raw_integers.append(integer_val)
            floats.append(integer_val / (2 ** actual_qubits))
        
        mean_val = np.mean(floats)
        print(f"  ✓ {len(floats)} samples (mean={mean_val:.4f})")
        
        # Build result data
        data = {
            'timestamp': timestamp,
            'source': 'ibm_quantum_superconducting',
            'source_detail': f'IBM Quantum {backend.name}',
            'backend': backend.name,
            'n_qubits': actual_qubits,
            'n_backend_qubits': backend.num_qubits,
            'job_id': job_id,
            'execution_time_s': round(elapsed, 2),
            'quantum_mechanism': 'superconducting_transmon_qubit_collapse',
            'circuit': f'H^{actual_qubits} + measure_all',
            'count': len(floats),
            'raw_integers': raw_integers,
            'floats': floats,
            'bits_per_number': actual_qubits,
        }
        
        # Save to file
        filename = f"ibmquantum_{backend.name}_{timestamp}.json"
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Saved to {filepath}")
        
        return data
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


def collect_ibm_quantum_batch(
    count: int = 1000,
    n_qubits: int = 32,
    output_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """
    Collect from IBM Quantum with automatic batching.
    
    IBM hardware has a max shots limit per job. This function
    handles splitting into multiple jobs if needed.
    
    Args:
        count: Total number of random numbers wanted
        n_qubits: Bits per random number
        output_dir: Output directory
        
    Returns:
        Combined results dict
    """
    # IBM Quantum typically allows up to 100,000 shots per job
    # For free tier, there may be lower limits
    MAX_SHOTS_PER_JOB = 10000
    
    if count <= MAX_SHOTS_PER_JOB:
        return collect_ibm_quantum(count, n_qubits, output_dir=output_dir)
    
    # Need to batch
    remaining = count
    all_floats = []
    all_integers = []
    batch_num = 0
    
    while remaining > 0:
        batch_size = min(remaining, MAX_SHOTS_PER_JOB)
        batch_num += 1
        print(f"\n  Batch {batch_num}: {batch_size} shots...")
        
        result = collect_ibm_quantum(batch_size, n_qubits, output_dir=output_dir)
        if result is None:
            break
        
        all_floats.extend(result['floats'])
        all_integers.extend(result['raw_integers'])
        remaining -= batch_size
    
    if not all_floats:
        return None
    
    # Return combined results
    return {
        'count': len(all_floats),
        'floats': all_floats,
        'raw_integers': all_integers,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IBM Quantum QRNG Collector")
    parser.add_argument("-n", "--count", type=int, default=1000,
                       help="Number of random numbers to generate (default: 1000)")
    parser.add_argument("-q", "--qubits", type=int, default=32,
                       help="Number of qubits per shot (default: 32)")
    parser.add_argument("-b", "--backend", type=str, default=None,
                       help="Specific backend name (default: least busy)")
    parser.add_argument("--list-backends", action="store_true",
                       help="List available backends and exit")
    
    args = parser.parse_args()
    
    if args.list_backends:
        if not QISKIT_AVAILABLE:
            print("Qiskit not installed. Run: pip install qiskit qiskit-ibm-runtime")
        else:
            service = QiskitRuntimeService()
            backends = service.backends(simulator=False, operational=True)
            print(f"\nAvailable IBM Quantum backends ({len(backends)}):")
            for b in backends:
                status = b.status()
                print(f"  {b.name}: {b.num_qubits} qubits, "
                      f"pending_jobs={status.pending_jobs}")
    else:
        print(f"\n{'='*60}")
        print(f"IBM Quantum QRNG Collection")
        print(f"{'='*60}")
        print(f"Generating {args.count} quantum random numbers...")
        print(f"Qubits per shot: {args.qubits}")
        print()
        
        result = collect_ibm_quantum(
            count=args.count,
            n_qubits=args.qubits,
            backend_name=args.backend,
        )
        
        if result:
            print(f"\n{'='*60}")
            print(f"Collection Summary")
            print(f"{'='*60}")
            print(f"  Backend: {result['backend']}")
            print(f"  Samples: {result['count']:,}")
            print(f"  Mean: {np.mean(result['floats']):.6f}")
            print(f"  Std Dev: {np.std(result['floats']):.6f}")
            print(f"  Execution: {result['execution_time_s']:.1f}s")
            print(f"  Mechanism: Superconducting transmon qubit collapse")
        else:
            print("\nCollection failed. Check credentials and backend availability.")
