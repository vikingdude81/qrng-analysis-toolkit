#!/usr/bin/env python3
"""
QRNG Bridge Module
==================
Connects the helios-trajectory-analysis QRNG data streams to the
inference framework's RandomnessProvider.

This bridge allows the inference experiments to use:
1. Pre-collected Outshift QRNG streams (quantum photon source)
2. Real-time Outshift API calls
3. CPU RDRAND hardware RNG
4. Standard CSPRNG/PRNG controls

The key hypothesis: Strange Attractor and Tensegrity architectures
may show different inference dynamics with QRNG vs PRNG.
"""

import json
import os
import sys
import urllib.error
from pathlib import Path
from typing import Optional, List, Callable, Iterator
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpu_hwrng import CPUHardwareRNG


class QRNGSourceType(Enum):
    """Available QRNG/RNG sources."""
    OUTSHIFT_STREAM = "outshift_stream"      # Pre-collected Outshift QRNG (SPDC photons)
    OUTSHIFT_LIVE = "outshift_live"          # Real-time Outshift API
    ANU_QRNG = "anu_qrng"                    # ANU vacuum fluctuation QRNG
    CIPHERSTONE_QRNG = "cipherstone_qrng"    # Cipherstone Qbert QRNG (CloudFlare tunnel)
    CPU_RDRAND = "cpu_rdrand"                # CPU hardware RNG (thermal noise)
    CSPRNG = "csprng"                        # os.urandom
    PRNG = "prng"                            # Mersenne Twister


class CipherstoneQRNGMode(Enum):
    """Cipherstone Qbert operating modes."""
    MODE_1_CONDITIONED = "mode_1"    # Raw with automatic noise conditioning based on live health tests
    MODE_2_RAW = "mode_2"            # Raw with no conditioning whatsoever


@dataclass
class StreamStats:
    """Statistics about a loaded QRNG stream."""
    source: str
    timestamp: str
    count: int
    mean: float
    std: float
    consumed: int
    remaining: int


class QRNGStreamProvider:
    """
    Provides QRNG values from pre-collected streams.
    
    This class manages a pool of collected QRNG streams and provides
    random values to the inference framework.
    """
    
    def __init__(self, streams_dir: Optional[Path] = None):
        """
        Initialize with path to QRNG streams directory.
        
        Args:
            streams_dir: Path to qrng_streams/ directory
        """
        if streams_dir is None:
            streams_dir = Path(__file__).parent.parent / "qrng_streams"
        
        self.streams_dir = Path(streams_dir)
        self._streams: List[dict] = []
        self._all_values: np.ndarray = np.array([])
        self._index: int = 0
        self._loaded = False
        
    def load_streams(self, pattern: str = "qrng_stream_*.json") -> int:
        """
        Load all QRNG streams matching pattern.
        
        Args:
            pattern: Glob pattern for stream files
            
        Returns:
            Total number of values loaded
        """
        self._streams = []
        all_values = []
        
        for f in sorted(self.streams_dir.glob(pattern)):
            with open(f) as fp:
                data = json.load(fp)
                floats = data.get('floats', data.get('values', []))
                if floats:
                    self._streams.append({
                        'file': f.name,
                        'timestamp': data.get('timestamp', ''),
                        'source': data.get('source', 'unknown'),
                        'count': len(floats),
                        'values': floats
                    })
                    all_values.extend(floats)
        
        self._all_values = np.array(all_values)
        self._index = 0
        self._loaded = True
        
        return len(self._all_values)
    
    def get_random(self) -> float:
        """
        Get next random value from the stream pool.
        
        Returns:
            Random float in [0, 1)
            
        Raises:
            RuntimeError: If streams exhausted
        """
        if not self._loaded:
            self.load_streams()
        
        if self._index >= len(self._all_values):
            # Wrap around (or could raise error for strict mode)
            self._index = 0
            
        value = self._all_values[self._index]
        self._index += 1
        return float(value)
    
    def get_random_batch(self, n: int) -> np.ndarray:
        """Get n random values at once."""
        if not self._loaded:
            self.load_streams()
            
        values = []
        for _ in range(n):
            values.append(self.get_random())
        return np.array(values)
    
    def reset(self):
        """Reset index to beginning of stream."""
        self._index = 0
    
    def shuffle(self, seed: Optional[int] = None):
        """Shuffle the pooled values (destroys temporal order)."""
        rng = np.random.RandomState(seed)
        rng.shuffle(self._all_values)
        self._index = 0
    
    @property
    def stats(self) -> StreamStats:
        """Get statistics about loaded streams."""
        if not self._loaded:
            self.load_streams()
            
        return StreamStats(
            source="outshift_qrng_pooled",
            timestamp=self._streams[0]['timestamp'] if self._streams else "",
            count=len(self._all_values),
            mean=float(np.mean(self._all_values)) if len(self._all_values) > 0 else 0,
            std=float(np.std(self._all_values)) if len(self._all_values) > 0 else 0,
            consumed=self._index,
            remaining=len(self._all_values) - self._index
        )
    
    @property
    def total_available(self) -> int:
        """Total values available."""
        if not self._loaded:
            self.load_streams()
        return len(self._all_values)


class LiveOutshiftProvider:
    """
    Provides real-time QRNG values from Outshift API.
    
    Fetches fresh quantum random numbers on demand.
    Note: API has rate limits (~100k bits/day).
    """
    
    def __init__(self):
        self._client = None
        self._buffer: List[float] = []
        self._buffer_index = 0
        
    def _ensure_client(self):
        """Lazy-load the Outshift client."""
        if self._client is None:
            from qrng_outshift_client import OutshiftQRNGClient
            self._client = OutshiftQRNGClient()
    
    def _refill_buffer(self, count: int = 100):
        """Fetch more values from API."""
        self._ensure_client()
        result = self._client.fetch_random_numbers(count=count)
        self._buffer = result['floats']
        self._buffer_index = 0
    
    def get_random(self) -> float:
        """Get a random value, fetching from API if needed."""
        if self._buffer_index >= len(self._buffer):
            self._refill_buffer()
        
        value = self._buffer[self._buffer_index]
        self._buffer_index += 1
        return float(value)


class ANUQRNGProvider:
    """
    Provides QRNG values from ANU Quantum Random Numbers Server.
    
    Uses the new authenticated API at:
    https://api.quantumnumbers.anu.edu.au
    
    ANU uses vacuum fluctuation measurements (shot noise from 
    quantum vacuum) which is a different physical process than
    Outshift's SPDC photon source.
    
    API supports:
    - uint8: integers 0-255
    - uint16: integers 0-65535
    - hex8: hexadecimal 00-ff
    - hex16: hexadecimal 0000-ffff
    """
    
    def __init__(self, cache_size: int = 1024):
        """
        Initialize ANU QRNG provider.
        
        Args:
            cache_size: Number of values to fetch per API call (max 1024)
        """
        self._buffer: List[float] = []
        self._buffer_index = 0
        self._cache_size = min(cache_size, 1024)  # API max is 1024
        self._api_key = os.environ.get("ANU_QRNG_API_KEY", "")
        self._api_url = os.environ.get("ANU_QRNG_API_URL", "https://api.quantumnumbers.anu.edu.au")
        self._available = bool(self._api_key)
    
    @property
    def available(self) -> bool:
        """Check if ANU QRNG is available (API key set)."""
        return self._available
    
    def _fetch_from_api(self, length: int = 100, data_type: str = "uint16") -> List[int]:
        """
        Fetch random numbers from ANU API.
        
        Args:
            length: Number of values to fetch (1-1024)
            data_type: 'uint8', 'uint16', 'hex8', or 'hex16'
            
        Returns:
            List of random integers
        """
        import urllib.request
        import urllib.parse
        import time as _time
        
        if not self._api_key:
            raise RuntimeError("ANU_QRNG_API_KEY not set in environment")
        
        params = urllib.parse.urlencode({
            'length': min(length, 1024),
            'type': data_type
        })
        
        url = f"{self._api_url}?{params}"
        
        # Retry logic for queue empty errors
        max_retries = 3
        for attempt in range(max_retries):
            req = urllib.request.Request(url)
            req.add_header('x-api-key', self._api_key)
            
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = json.loads(response.read().decode('utf-8'))
                
                if data.get('success', False):
                    return data.get('data', [])
                
                # Queue empty - wait and retry
                if 'queue' in str(data.get('message', '')).lower():
                    if attempt < max_retries - 1:
                        _time.sleep(2 * (attempt + 1))  # Exponential backoff
                        continue
                
                raise RuntimeError(f"ANU API error: {data}")
                
            except urllib.error.HTTPError as e:
                if e.code == 400 and attempt < max_retries - 1:
                    # May be queue empty, retry
                    _time.sleep(2 * (attempt + 1))
                    continue
                raise
        
        raise RuntimeError("ANU API: Max retries exceeded (queue may be empty)")
    
    def _refill_buffer(self):
        """Fetch more values from API and fill buffer."""
        raw_values = self._fetch_from_api(length=self._cache_size, data_type='uint16')
        # Convert uint16 (0-65535) to float [0, 1)
        self._buffer = [v / 65536.0 for v in raw_values]
        self._buffer_index = 0
    
    def get_random(self) -> float:
        """
        Get a random float in [0, 1) from ANU QRNG.
        
        Fetches from API in batches for efficiency.
        """
        if not self._available:
            raise RuntimeError("ANU_QRNG_API_KEY not set. Add to .env file.")
        
        if self._buffer_index >= len(self._buffer):
            self._refill_buffer()
        
        value = self._buffer[self._buffer_index]
        self._buffer_index += 1
        return value
    
    def get_random_batch(self, n: int) -> np.ndarray:
        """Get n random values from ANU QRNG."""
        return np.array([self.get_random() for _ in range(n)])
    
    def get_randint(self, min_val: int = 0, max_val: int = 256) -> int:
        """
        Get a random integer in [min_val, max_val).
        
        Uses uint8 for small ranges, uint16 for larger.
        """
        if not self._available:
            raise RuntimeError("ANU_QRNG_API_KEY not set")
        
        range_size = max_val - min_val
        if range_size <= 256:
            # Use uint8 for efficiency
            raw = self._fetch_from_api(length=1, data_type='uint8')[0]
            return min_val + (raw % range_size)
        else:
            # Use uint16
            raw = self._fetch_from_api(length=1, data_type='uint16')[0]
            return min_val + (raw % range_size)


class CipherstoneQRNGProvider:
    """
    Provides QRNG values from Cipherstone Qbert QRNG service.
    
    Uses CloudFlare tunnel for delivery at:
    https://qbert.cipherstone.co/
    
    Cipherstone Qbert provides quantum random numbers with two operating modes:
    - Mode 1: Raw with automatic noise conditioning based on live health tests
    - Mode 2: Raw with no conditioning whatsoever
    
    API supports:
    - uint8: integers 0-255
    - uint16: integers 0-65535
    - hex8: hexadecimal 00-ff
    - hex16: hexadecimal 0000-ffff
    
    Note: The default API keys below are public keys provided by Cipherstone
    for general use. They can be overridden via environment variables for
    production deployments with custom keys.
    """
    
    # Default API keys (public keys provided by Cipherstone for general use)
    # Can be overridden via environment variables
    DEFAULT_MODE1_KEY = os.environ.get("CIPHERSTONE_QRNG_MODE1_KEY", "")
    DEFAULT_MODE2_KEY = os.environ.get("CIPHERSTONE_QRNG_MODE2_KEY", "")
    DEFAULT_API_URL = "https://qbert.cipherstone.co/"
    
    def __init__(self, mode: CipherstoneQRNGMode = CipherstoneQRNGMode.MODE_1_CONDITIONED, cache_size: int = 1024):
        """
        Initialize Cipherstone QRNG provider.
        
        Args:
            mode: Operating mode (MODE_1_CONDITIONED or MODE_2_RAW)
            cache_size: Number of values to fetch per API call (max 1024)
        """
        self._buffer: List[float] = []
        self._buffer_index = 0
        self._cache_size = min(cache_size, 1024)  # API max is 1024
        self._mode = mode
        
        # Get API configuration from environment or use defaults
        self._mode1_key = os.environ.get("CIPHERSTONE_QRNG_MODE1_KEY", self.DEFAULT_MODE1_KEY)
        self._mode2_key = os.environ.get("CIPHERSTONE_QRNG_MODE2_KEY", self.DEFAULT_MODE2_KEY)
        self._api_url = os.environ.get("CIPHERSTONE_QRNG_API_URL", self.DEFAULT_API_URL)
        
        # Always available since default keys are provided
        self._available = True
    
    @property
    def available(self) -> bool:
        """Check if Cipherstone QRNG is available."""
        return self._available
    
    @property
    def mode(self) -> CipherstoneQRNGMode:
        """Get current operating mode."""
        return self._mode
    
    def set_mode(self, mode: CipherstoneQRNGMode):
        """
        Set operating mode.
        
        Args:
            mode: New operating mode
        """
        self._mode = mode
        # Clear buffer to ensure next fetch uses new mode
        self._buffer = []
        self._buffer_index = 0
    
    def _get_api_key(self) -> str:
        """Get API key for current mode."""
        if self._mode == CipherstoneQRNGMode.MODE_1_CONDITIONED:
            return self._mode1_key
        else:
            return self._mode2_key
    
    def _fetch_from_api(self, length: int = 100, data_type: str = "uint16") -> List[int]:
        """
        Fetch random numbers from Cipherstone API.
        
        Args:
            length: Number of values to fetch (1-1024)
            data_type: 'uint8', 'uint16', 'hex8', or 'hex16'
            
        Returns:
            List of random integers
        """
        import urllib.request
        import urllib.parse
        import time as _time
        
        api_key = self._get_api_key()
        
        params = urllib.parse.urlencode({
            'length': min(length, 1024),
            'type': data_type
        })
        
        url = f"{self._api_url}?{params}"
        
        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            req = urllib.request.Request(url)
            req.add_header('x-api-key', api_key)
            
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = json.loads(response.read().decode('utf-8'))
                
                # Check if data is a list (successful response)
                if isinstance(data, list):
                    return data
                
                # Check if data is a dict with 'data' field (new API format)
                if isinstance(data, dict) and 'data' in data and data.get('success', False):
                    return data['data']
                
                # If response is a dict with error info, raise
                if isinstance(data, dict):
                    raise RuntimeError(f"Cipherstone API error: {data}")
                
                raise RuntimeError(f"Unexpected API response format: {type(data)}")
                
            except urllib.error.HTTPError as e:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    _time.sleep(2 ** (attempt + 1))
                    continue
                raise RuntimeError(f"Cipherstone API HTTP error: {e.code} {e.reason}")
            except urllib.error.URLError as e:
                if attempt < max_retries - 1:
                    _time.sleep(2 ** (attempt + 1))
                    continue
                raise RuntimeError(f"Cipherstone API connection error: {e.reason}")
        
        raise RuntimeError("Cipherstone API: Max retries exceeded")
    
    def _refill_buffer(self):
        """Fetch more values from API and fill buffer."""
        raw_values = self._fetch_from_api(length=self._cache_size, data_type='uint16')
        # Convert uint16 (0-65535) to float [0, 1)
        self._buffer = [v / 65536.0 for v in raw_values]
        self._buffer_index = 0
    
    def get_random(self) -> float:
        """
        Get a random float in [0, 1) from Cipherstone QRNG.
        
        Fetches from API in batches for efficiency.
        """
        if not self._available:
            raise RuntimeError("Cipherstone QRNG not available")
        
        if self._buffer_index >= len(self._buffer):
            self._refill_buffer()
        
        value = self._buffer[self._buffer_index]
        self._buffer_index += 1
        return value
    
    def get_random_batch(self, n: int) -> np.ndarray:
        """Get n random values from Cipherstone QRNG."""
        return np.array([self.get_random() for _ in range(n)])
    
    def get_random_uint8(self) -> int:
        """
        Get a random uint8 (0-255) from Cipherstone QRNG.
        
        Returns:
            Random integer in [0, 255]
        """
        if not self._available:
            raise RuntimeError("Cipherstone QRNG not available")
        
        raw = self._fetch_from_api(length=1, data_type='uint8')[0]
        return raw
    
    def get_random_uint16(self) -> int:
        """
        Get a random uint16 (0-65535) from Cipherstone QRNG.
        
        Returns:
            Random integer in [0, 65535]
        """
        if not self._available:
            raise RuntimeError("Cipherstone QRNG not available")
        
        raw = self._fetch_from_api(length=1, data_type='uint16')[0]
        return raw
    
    def get_randint(self, min_val: int = 0, max_val: int = 256) -> int:
        """
        Get a random integer in [min_val, max_val).
        
        Uses uint8 for small ranges, uint16 for larger.
        
        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (exclusive)
            
        Returns:
            Random integer in [min_val, max_val)
        """
        if not self._available:
            raise RuntimeError("Cipherstone QRNG not available")
        
        range_size = max_val - min_val
        if range_size <= 256:
            # Use uint8 for efficiency
            raw = self._fetch_from_api(length=1, data_type='uint8')[0]
            return min_val + (raw % range_size)
        else:
            # Use uint16
            raw = self._fetch_from_api(length=1, data_type='uint16')[0]
            return min_val + (raw % range_size)


class UnifiedRandomnessProvider:
    """
    Unified provider for all randomness sources.
    
    Wraps:
    - Pre-collected Outshift QRNG streams (SPDC photon source)
    - Live Outshift API
    - ANU QRNG (vacuum fluctuation)
    - CPU RDRAND (thermal noise)
    - CSPRNG (os.urandom)
    - PRNG (Mersenne Twister)
    
    Compatible with inference_framework.experiment.RandomnessProvider interface.
    """
    
    def __init__(
        self,
        streams_dir: Optional[Path] = None,
        enable_live_api: bool = False,
        cipherstone_mode: CipherstoneQRNGMode = CipherstoneQRNGMode.MODE_1_CONDITIONED
    ):
        """
        Initialize all randomness sources.
        
        Args:
            streams_dir: Path to QRNG streams directory
            enable_live_api: Whether to enable live Outshift API calls
            cipherstone_mode: Operating mode for Cipherstone QRNG
        """
        # Pre-collected streams
        self._stream_provider = QRNGStreamProvider(streams_dir)
        
        # Live API (lazy loaded)
        self._live_provider: Optional[LiveOutshiftProvider] = None
        self._enable_live = enable_live_api
        
        # ANU QRNG (lazy loaded)
        self._anu_provider: Optional[ANUQRNGProvider] = None
        
        # Cipherstone QRNG (lazy loaded)
        self._cipherstone_provider: Optional[CipherstoneQRNGProvider] = None
        self._cipherstone_mode = cipherstone_mode
        
        # CPU RDRAND
        try:
            self._cpu_hwrng = CPUHardwareRNG(method="bcrypt")
            self._cpu_available = True
        except Exception:
            self._cpu_hwrng = None
            self._cpu_available = False
        
        # PRNG with configurable seed
        self._prng = np.random.RandomState()
        
    def get_source(self, source_type: QRNGSourceType, seed: Optional[int] = None) -> Callable[[], float]:
        """
        Get a randomness source function.
        
        Args:
            source_type: Which source to use
            seed: Seed for PRNG (ignored for hardware sources)
            
        Returns:
            Callable that returns random floats in [0, 1)
        """
        if source_type == QRNGSourceType.OUTSHIFT_STREAM:
            # Use pre-collected streams
            return self._stream_provider.get_random
        
        elif source_type == QRNGSourceType.OUTSHIFT_LIVE:
            # Use live API
            if self._live_provider is None:
                self._live_provider = LiveOutshiftProvider()
            return self._live_provider.get_random
        
        elif source_type == QRNGSourceType.ANU_QRNG:
            # Use ANU vacuum fluctuation QRNG
            if self._anu_provider is None:
                self._anu_provider = ANUQRNGProvider()
            if not self._anu_provider.available:
                raise RuntimeError("ANU QRNG not available. Run: pip install quantumrandom")
            return self._anu_provider.get_random
        
        elif source_type == QRNGSourceType.CIPHERSTONE_QRNG:
            # Use Cipherstone Qbert QRNG
            if self._cipherstone_provider is None:
                self._cipherstone_provider = CipherstoneQRNGProvider(mode=self._cipherstone_mode)
            if not self._cipherstone_provider.available:
                raise RuntimeError("Cipherstone QRNG not available")
            return self._cipherstone_provider.get_random
        
        elif source_type == QRNGSourceType.CPU_RDRAND:
            # Use CPU hardware RNG
            if not self._cpu_available:
                raise RuntimeError("CPU RDRAND not available")
            return lambda: float(self._cpu_hwrng.get_random_floats(1)[0])
        
        elif source_type == QRNGSourceType.CSPRNG:
            # Use os.urandom
            import struct
            def csprng_random():
                return struct.unpack('>I', os.urandom(4))[0] / (2**32)
            return csprng_random
        
        elif source_type == QRNGSourceType.PRNG:
            # Use Mersenne Twister
            if seed is not None:
                self._prng.seed(seed)
            return lambda: float(self._prng.random())
        
        raise ValueError(f"Unknown source type: {source_type}")
    
    def get_qrng_source(self) -> Callable[[], float]:
        """Get the primary QRNG source (pre-collected streams)."""
        return self.get_source(QRNGSourceType.OUTSHIFT_STREAM)
    
    def get_control_source(self, seed: Optional[int] = None) -> Callable[[], float]:
        """Get a control PRNG source."""
        return self.get_source(QRNGSourceType.PRNG, seed=seed)
    
    @property
    def stream_stats(self) -> StreamStats:
        """Get stats about loaded QRNG streams."""
        return self._stream_provider.stats
    
    def reset_streams(self):
        """Reset stream index to beginning."""
        self._stream_provider.reset()
    
    def get_cipherstone_provider(self) -> CipherstoneQRNGProvider:
        """
        Get direct access to Cipherstone provider for mode switching.
        
        Returns:
            CipherstoneQRNGProvider instance
        """
        if self._cipherstone_provider is None:
            self._cipherstone_provider = CipherstoneQRNGProvider(mode=self._cipherstone_mode)
        return self._cipherstone_provider


def create_inference_experiment_with_qrng(
    llm_call: Callable,
    streams_dir: Optional[Path] = None,
    experiment_id: str = "qrng_inference_001"
):
    """
    Factory function to create an InferenceExperiment with QRNG integration.
    
    Args:
        llm_call: Function to call the LLM
        streams_dir: Path to QRNG streams directory
        experiment_id: Unique experiment identifier
        
    Returns:
        Configured InferenceExperiment instance
    """
    from .experiment import InferenceExperiment, ExperimentConfig, RandomnessProvider
    
    # Create unified provider
    unified = UnifiedRandomnessProvider(streams_dir=streams_dir)
    
    # Create experiment config
    config = ExperimentConfig(
        experiment_id=experiment_id,
        trials_per_condition=10,
        max_iterations=10
    )
    
    # Create the experiment with QRNG source
    experiment = InferenceExperiment(
        llm_call=llm_call,
        config=config,
        qrng_source=unified.get_qrng_source()
    )
    
    return experiment, unified


# =============================================================================
# CLI for testing
# =============================================================================

def main():
    """Test the QRNG bridge."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    console.print("[bold cyan]QRNG Bridge Test[/]")
    console.print()
    
    # Test stream provider
    console.print("[bold]1. Testing Stream Provider[/]")
    stream_provider = QRNGStreamProvider()
    count = stream_provider.load_streams()
    console.print(f"   Loaded {count} values from streams")
    
    # Get some values
    sample = [stream_provider.get_random() for _ in range(10)]
    console.print(f"   Sample: {[f'{v:.4f}' for v in sample]}")
    console.print(f"   Stats: {stream_provider.stats}")
    
    # Test unified provider
    console.print("\n[bold]2. Testing Unified Provider[/]")
    unified = UnifiedRandomnessProvider()
    
    table = Table(title="Randomness Sources")
    table.add_column("Source", style="cyan")
    table.add_column("Sample (5 values)")
    table.add_column("Mean")
    
    sources = [
        (QRNGSourceType.OUTSHIFT_STREAM, "Outshift QRNG"),
        (QRNGSourceType.CPU_RDRAND, "CPU RDRAND"),
        (QRNGSourceType.CSPRNG, "CSPRNG"),
        (QRNGSourceType.PRNG, "PRNG"),
        # Uncomment to test live APIs (avoid during testing to prevent rate limits):
        # (QRNGSourceType.ANU_QRNG, "ANU QRNG"),
        # (QRNGSourceType.CIPHERSTONE_QRNG, "Cipherstone QRNG"),
    ]
    
    for source_type, name in sources:
        try:
            source = unified.get_source(source_type, seed=42)
            values = [source() for _ in range(100)]
            sample_str = ", ".join(f"{v:.3f}" for v in values[:5])
            mean = np.mean(values)
            table.add_row(name, sample_str, f"{mean:.4f}")
        except Exception as e:
            table.add_row(name, f"[red]Error: {e}[/]", "-")
    
    console.print(table)
    
    # Display configuration info
    console.print("\n[bold]3. QRNG Configuration[/]")
    
    # Check ANU availability using the proper method
    try:
        anu_source = unified.get_source(QRNGSourceType.ANU_QRNG)
        console.print(f"   ANU QRNG Available: True")
    except RuntimeError:
        console.print(f"   ANU QRNG Available: False (API key not set)")
    
    # Show Cipherstone config
    cs_provider = unified.get_cipherstone_provider()
    console.print(f"   Cipherstone QRNG Available: {cs_provider.available}")
    console.print(f"   Cipherstone Mode: {cs_provider.mode.value}")
    console.print(f"   Cipherstone URL: {cs_provider._api_url}")
    
    console.print("\n[green]✓ QRNG Bridge ready for inference experiments[/]")


if __name__ == "__main__":
    main()
