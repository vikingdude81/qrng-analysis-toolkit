"""Test QRNG API and save stream for later analysis."""
import json
from datetime import datetime
from pathlib import Path

from qrng_outshift_client import OutshiftQRNGClient

def test_qrng_api():
    """Test the Outshift QRNG API and save results."""
    print("Initializing QRNG client...")
    client = OutshiftQRNGClient()
    print("Client initialized successfully!")
    
    # Generate random numbers
    print("\nGenerating quantum random numbers...")
    result = client.generate(
        bits_per_block=32,
        number_of_blocks=1000,  # Get 1000 numbers
        format='decimal'
    )
    
    random_numbers_raw = result.get('random_numbers', [])
    # Extract decimal values (API returns dicts like {'decimal': '123'})
    random_numbers = [int(n['decimal']) if isinstance(n, dict) else int(n) for n in random_numbers_raw]
    print(f"Got {len(random_numbers)} random numbers")
    print(f"First 10: {random_numbers[:10]}")
    
    # Convert to floats [0, 1)
    max_val = 2**32
    floats = [n / max_val for n in random_numbers]
    print(f"\nAs floats [0,1): {floats[:5]}")
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("qrng_streams")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"qrng_stream_{timestamp}.json"
    
    stream_data = {
        "timestamp": timestamp,
        "source": "outshift_qrng_api",
        "bits_per_block": 32,
        "count": len(random_numbers),
        "raw_integers": random_numbers,
        "floats": floats,
        "encoding": result.get("encoding"),
    }
    
    with open(output_file, 'w') as f:
        json.dump(stream_data, f, indent=2)
    
    print(f"\nStream saved to: {output_file}")
    print(f"Total numbers: {len(random_numbers)}")
    
    # Basic statistics
    import numpy as np
    arr = np.array(floats)
    print(f"\nBasic statistics:")
    print(f"  Mean: {arr.mean():.4f} (expected: 0.5)")
    print(f"  Std:  {arr.std():.4f} (expected: ~0.289)")
    print(f"  Min:  {arr.min():.4f}")
    print(f"  Max:  {arr.max():.4f}")
    
    return stream_data

if __name__ == "__main__":
    test_qrng_api()
