"""
Outshift QRNG API Client

Provides access to quantum random number generation via the Outshift QRNG API.
This is a true QRNG service suitable for consciousness influence detection experiments.

API Documentation: https://api.qrng.outshift.com/
"""

import os
import json
import requests
from typing import List, Dict, Optional, Literal
from pathlib import Path


class OutshiftQRNGClient:
    """
    Client for Outshift QRNG API.
    
    Supports various encodings and formats for quantum random numbers.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize the QRNG client.
        
        Args:
            api_key: API key for authentication. If None, reads from .env file.
            api_url: API endpoint URL. If None, reads from .env or uses default.
        """
        # Try to load from .env file
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            self._load_env(env_file)
        
        self.api_key = api_key or os.getenv('QRNG_OUTSHIFT_API_KEY')
        self.api_url = api_url or os.getenv(
            'QRNG_OUTSHIFT_API_URL',
            'https://api.qrng.outshift.com/api/v1/random_numbers'
        )
        
        if not self.api_key:
            raise ValueError(
                "API key required. Set QRNG_OUTSHIFT_API_KEY environment variable "
                "or pass api_key parameter. See .env.example for template."
            )
    
    def _load_env(self, env_file: Path):
        """Load environment variables from .env file."""
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip()
                    # Remove surrounding quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    os.environ[key.strip()] = value
    
    def generate(
        self,
        bits_per_block: int = 10,
        number_of_blocks: int = 10,
        encoding: Literal['base64', 'raw'] = 'raw',
        format: Literal['binary', 'octal', 'decimal', 'hexadecimal', 'all'] = 'decimal'
    ) -> Dict:
        """
        Generate quantum random numbers.
        
        Args:
            bits_per_block: Number of bits per random number (1-10000)
            number_of_blocks: Number of random numbers to generate (1-1000)
            encoding: Encoding format ('base64' or 'raw')
            format: Output format ('binary', 'octal', 'decimal', 'hexadecimal', 'all')
            
        Returns:
            API response as dictionary containing random numbers
            
        Example response:
            {
                "encoding": "raw",
                "random_numbers": [
                    {
                        "binary": "1010101010",
                        "octal": "1252",
                        "decimal": "682",
                        "hexadecimal": "2aa"
                    },
                    ...
                ]
            }
        """
        headers = {
            "Content-Type": "application/json",
            "x-id-api-key": self.api_key
        }
        
        data = {
            "encoding": encoding,
            "format": format,
            "bits_per_block": bits_per_block,
            "number_of_blocks": number_of_blocks
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(
                f"QRNG API error {response.status_code}: {response.text}"
            )
    
    def generate_normalized_floats(
        self,
        count: int = 100,
        bits_per_number: int = 32
    ) -> List[float]:
        """
        Generate normalized floats in [0, 1) suitable for trajectory analysis.
        
        Args:
            count: Number of random floats to generate
            bits_per_number: Bits per number (higher = more precision)
            
        Returns:
            List of floats in range [0, 1)
        """
        # Generate decimal random numbers
        result = self.generate(
            bits_per_block=bits_per_number,
            number_of_blocks=count,
            encoding='raw',
            format='decimal'
        )
        
        # Normalize to [0, 1)
        max_value = 2 ** bits_per_number
        numbers = [int(block['decimal']) for block in result['random_numbers']]
        return [n / max_value for n in numbers]
    
    def generate_binary_stream(
        self,
        bits: int = 1000
    ) -> str:
        """
        Generate a binary stream of quantum random bits.
        
        Args:
            bits: Total number of bits to generate
            
        Returns:
            String of '0' and '1' characters
        """
        # API limits: max 10000 bits per block, max 1000 blocks
        # So we can get up to 10,000,000 bits per request
        
        if bits <= 10000:
            # Single block
            result = self.generate(
                bits_per_block=bits,
                number_of_blocks=1,
                encoding='raw',
                format='binary'
            )
            return result['random_numbers'][0]['binary']
        else:
            # Multiple blocks
            blocks_needed = (bits + 9999) // 10000  # Ceiling division
            if blocks_needed > 1000:
                raise ValueError(f"Requested {bits} bits requires {blocks_needed} blocks (max 1000)")
            
            bits_per_block = min(10000, bits)
            result = self.generate(
                bits_per_block=bits_per_block,
                number_of_blocks=blocks_needed,
                encoding='raw',
                format='binary'
            )
            
            # Concatenate blocks
            binary_stream = ''.join([block['binary'] for block in result['random_numbers']])
            return binary_stream[:bits]  # Trim to exact length
    
    def stream_to_qrng_scope(
        self,
        scope,
        count: int = 100,
        bits_per_number: int = 16
    ):
        """
        Feed quantum random numbers directly to a QRNGStreamScope.
        
        Args:
            scope: QRNGStreamScope instance
            count: Number of values to generate and feed
            bits_per_number: Bits per random number
            
        Returns:
            List of metrics from each update
        """
        values = self.generate_normalized_floats(count, bits_per_number)
        return [scope.update_from_stream(v) for v in values]


# Convenience function
def get_qrng_client() -> OutshiftQRNGClient:
    """Get a configured QRNG client (loads from .env)."""
    return OutshiftQRNGClient()


if __name__ == "__main__":
    """
    Quick test of the QRNG API.
    """
    print("=== Outshift QRNG Client Test ===\n")
    
    try:
        client = get_qrng_client()
        print(f"✓ Client initialized")
        print(f"  API URL: {client.api_url}")
        
        # Test 1: Generate some random numbers
        print("\n--- Test 1: Generate 10 random numbers (10 bits each) ---")
        result = client.generate(bits_per_block=10, number_of_blocks=10, format='all')
        decimals = [block['decimal'] for block in result['random_numbers']]
        print(f"Decimal values: {decimals[:5]}...")
        
        # Test 2: Generate normalized floats
        print("\n--- Test 2: Generate normalized floats ---")
        floats = client.generate_normalized_floats(count=10, bits_per_number=16)
        print(f"Floats [0,1): {[f'{x:.4f}' for x in floats[:5]]}...")
        
        # Test 3: Generate binary stream
        print("\n--- Test 3: Generate binary stream ---")
        binary = client.generate_binary_stream(bits=100)
        print(f"Binary (first 50 bits): {binary[:50]}...")
        print(f"Total bits: {len(binary)}")
        
        # Test 4: Integration with QRNGStreamScope
        print("\n--- Test 4: Feed to QRNGStreamScope ---")
        try:
            from helios_anomaly_scope import QRNGStreamScope
            
            scope = QRNGStreamScope(history_len=50)
            metrics_list = client.stream_to_qrng_scope(scope, count=60)
            
            final_metrics = metrics_list[-1]
            print(f"✓ Fed 60 quantum random values to scope")
            print(f"  Final position: ({final_metrics.get('x', 0):.3f}, {final_metrics.get('y', 0):.3f})")
            print(f"  Epiplexity: {final_metrics.get('epiplexity', 0):.2f}")
            print(f"  Time-bounded entropy: {final_metrics.get('time_bounded_entropy', 0):.3f}")
            print(f"  Structural emergence: {final_metrics.get('structural_emergence', False)}")
            
            if scope.events:
                print(f"\n  Events detected: {len(scope.events)}")
                for event in scope.events[:3]:
                    print(f"    - {event.event_type} (confidence: {event.confidence:.2f})")
            else:
                print(f"\n  No anomalies detected (as expected for pure QRNG)")
        
        except ImportError:
            print("  (Skipped - helios_anomaly_scope not available)")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
