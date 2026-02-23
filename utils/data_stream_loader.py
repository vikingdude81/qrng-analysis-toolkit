"""
Data Stream Loader
==================

Utilities for loading and streaming various data formats
for trajectory analysis.

Supports:
- CSV files (value per row)
- Binary files (int8, float32, etc.)
- Network streams (TCP/UDP)
- Generator functions

Usage:
    from data_stream_loader import StreamLoader, CSVStream, BinaryStream

    # From CSV
    loader = StreamLoader(CSVStream('data.csv'))
    for value in loader:
        process(value)
    
    # From binary
    loader = StreamLoader(BinaryStream('data.bin', dtype='float32'))
    
    # From network
    loader = StreamLoader(NetworkStream('localhost', 5000))
"""

import numpy as np
from typing import Generator, Union, Optional, Iterator, Protocol, BinaryIO
from pathlib import Path
from dataclasses import dataclass
import struct
import socket
import threading
from queue import Queue
import csv


class DataStream(Protocol):
    """Protocol for data stream sources."""
    
    def __iter__(self) -> Iterator[float]:
        ...
    
    def close(self) -> None:
        ...


@dataclass
class StreamStats:
    """Statistics about a data stream."""
    count: int = 0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    sum_val: float = 0.0
    sum_sq: float = 0.0
    
    def update(self, value: float):
        self.count += 1
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        self.sum_val += value
        self.sum_sq += value * value
    
    @property
    def mean(self) -> float:
        return self.sum_val / self.count if self.count > 0 else 0
    
    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0
        return (self.sum_sq - self.sum_val ** 2 / self.count) / (self.count - 1)
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)


class CSVStream:
    """Stream values from a CSV file."""
    
    def __init__(self, filepath: Union[str, Path], column: int = 0, 
                 has_header: bool = False, delimiter: str = ','):
        """
        Args:
            filepath: Path to CSV file
            column: Column index to read values from
            has_header: Whether first row is header
            delimiter: CSV delimiter
        """
        self.filepath = Path(filepath)
        self.column = column
        self.has_header = has_header
        self.delimiter = delimiter
        self._file = None
        self._reader = None
        
    def __iter__(self) -> Iterator[float]:
        with open(self.filepath, 'r') as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            
            if self.has_header:
                next(reader)  # Skip header
            
            for row in reader:
                if len(row) > self.column:
                    try:
                        yield float(row[self.column])
                    except ValueError:
                        continue
    
    def close(self):
        pass


class BinaryStream:
    """Stream values from a binary file."""
    
    DTYPE_FORMATS = {
        'int8': ('b', 1),
        'uint8': ('B', 1),
        'int16': ('h', 2),
        'uint16': ('H', 2),
        'int32': ('i', 4),
        'uint32': ('I', 4),
        'int64': ('q', 8),
        'uint64': ('Q', 8),
        'float32': ('f', 4),
        'float64': ('d', 8),
    }
    
    def __init__(self, filepath: Union[str, Path], dtype: str = 'float32',
                 normalize: bool = True):
        """
        Args:
            filepath: Path to binary file
            dtype: Data type (int8, uint8, float32, etc.)
            normalize: If True, normalize integers to [0, 1]
        """
        self.filepath = Path(filepath)
        self.dtype = dtype
        self.normalize = normalize
        
        if dtype not in self.DTYPE_FORMATS:
            raise ValueError(f"Unknown dtype: {dtype}. Use one of {list(self.DTYPE_FORMATS.keys())}")
        
        self.fmt, self.size = self.DTYPE_FORMATS[dtype]
        
        # Normalization factor for integer types
        if dtype.startswith('int'):
            bits = int(dtype[3:])
            self.norm_factor = 2 ** (bits - 1) - 1
        elif dtype.startswith('uint'):
            bits = int(dtype[4:])
            self.norm_factor = 2 ** bits - 1
        else:
            self.norm_factor = 1.0
    
    def __iter__(self) -> Iterator[float]:
        with open(self.filepath, 'rb') as f:
            while True:
                data = f.read(self.size)
                if not data or len(data) < self.size:
                    break
                
                value = struct.unpack(self.fmt, data)[0]
                
                if self.normalize and self.dtype.startswith(('int', 'uint')):
                    value = value / self.norm_factor
                
                yield float(value)
    
    def close(self):
        pass


class NetworkStream:
    """Stream values from a network socket."""
    
    def __init__(self, host: str, port: int, protocol: str = 'tcp',
                 dtype: str = 'float32', buffer_size: int = 1000):
        """
        Args:
            host: Host to connect to
            port: Port number
            protocol: 'tcp' or 'udp'
            dtype: Data type of incoming values
            buffer_size: Size of internal buffer
        """
        self.host = host
        self.port = port
        self.protocol = protocol
        self.dtype = dtype
        self.buffer = Queue(maxsize=buffer_size)
        
        self._socket = None
        self._running = False
        self._thread = None
        
        if dtype not in BinaryStream.DTYPE_FORMATS:
            raise ValueError(f"Unknown dtype: {dtype}")
        
        self.fmt, self.size = BinaryStream.DTYPE_FORMATS[dtype]
    
    def start(self):
        """Start receiving data."""
        if self.protocol == 'tcp':
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self.host, self.port))
        else:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.bind(('', self.port))
        
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
    
    def _receive_loop(self):
        """Internal receive loop."""
        while self._running:
            try:
                data = self._socket.recv(self.size)
                if not data:
                    break
                
                if len(data) >= self.size:
                    value = struct.unpack(self.fmt, data[:self.size])[0]
                    if not self.buffer.full():
                        self.buffer.put(float(value))
            except Exception:
                break
    
    def __iter__(self) -> Iterator[float]:
        if not self._running:
            self.start()
        
        while self._running:
            try:
                value = self.buffer.get(timeout=1.0)
                yield value
            except:
                continue
    
    def close(self):
        """Close the network connection."""
        self._running = False
        if self._socket:
            self._socket.close()


class GeneratorStream:
    """Wrap a generator function as a stream."""
    
    def __init__(self, generator_fn, *args, **kwargs):
        """
        Args:
            generator_fn: A function that returns a generator
            *args, **kwargs: Arguments to pass to the generator function
        """
        self.generator_fn = generator_fn
        self.args = args
        self.kwargs = kwargs
    
    def __iter__(self) -> Iterator[float]:
        gen = self.generator_fn(*self.args, **self.kwargs)
        for value in gen:
            yield float(value)
    
    def close(self):
        pass


class StreamLoader:
    """
    Main class for loading and processing data streams.
    
    Provides:
    - Lazy iteration
    - Statistics tracking
    - Batching
    - Filtering
    """
    
    def __init__(self, source: DataStream, track_stats: bool = True):
        """
        Args:
            source: A DataStream source
            track_stats: Whether to track running statistics
        """
        self.source = source
        self.track_stats = track_stats
        self.stats = StreamStats() if track_stats else None
    
    def __iter__(self) -> Iterator[float]:
        for value in self.source:
            if self.stats:
                self.stats.update(value)
            yield value
    
    def batch(self, size: int) -> Generator[np.ndarray, None, None]:
        """Yield values in batches."""
        batch = []
        for value in self:
            batch.append(value)
            if len(batch) >= size:
                yield np.array(batch)
                batch = []
        
        if batch:
            yield np.array(batch)
    
    def filter(self, min_val: Optional[float] = None, 
               max_val: Optional[float] = None) -> Generator[float, None, None]:
        """Filter values by range."""
        for value in self:
            if min_val is not None and value < min_val:
                continue
            if max_val is not None and value > max_val:
                continue
            yield value
    
    def close(self):
        """Close the underlying stream."""
        self.source.close()


def load_from_file(filepath: Union[str, Path], **kwargs) -> StreamLoader:
    """
    Auto-detect file type and return appropriate loader.
    
    Args:
        filepath: Path to file
        **kwargs: Additional arguments for the stream type
    
    Returns:
        StreamLoader instance
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.csv':
        return StreamLoader(CSVStream(filepath, **kwargs))
    elif suffix == '.bin':
        return StreamLoader(BinaryStream(filepath, **kwargs))
    elif suffix in ('.npy', '.npz'):
        # For numpy files, wrap in a generator
        data = np.load(filepath)
        if isinstance(data, np.ndarray):
            return StreamLoader(GeneratorStream(lambda: iter(data.flatten())))
        else:
            # .npz file - use first array
            key = list(data.keys())[0]
            return StreamLoader(GeneratorStream(lambda: iter(data[key].flatten())))
    else:
        raise ValueError(f"Unknown file type: {suffix}")


def create_test_csv(filepath: str, count: int = 1000, 
                    include_bias: bool = False) -> None:
    """Create a test CSV file with random values."""
    with open(filepath, 'w') as f:
        f.write("value,timestamp\n")
        for i in range(count):
            if include_bias and i > count // 2:
                value = np.random.random() * 0.3 + 0.6  # Biased
            else:
                value = np.random.random()
            f.write(f"{value},{i}\n")


def create_test_binary(filepath: str, count: int = 1000,
                        dtype: str = 'float32') -> None:
    """Create a test binary file with random values."""
    values = np.random.random(count).astype(dtype)
    values.tofile(filepath)


if __name__ == "__main__":
    # Demo
    import tempfile
    import os
    
    print("=== Data Stream Loader Demo ===\n")
    
    # Create test CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
        f.write("value\n")
        for _ in range(100):
            f.write(f"{np.random.random()}\n")
    
    print(f"Created test CSV: {csv_path}")
    
    # Load and process
    loader = StreamLoader(CSVStream(csv_path, has_header=True))
    
    values = list(loader)
    print(f"Loaded {len(values)} values")
    print(f"Stats: mean={loader.stats.mean:.4f}, std={loader.stats.std:.4f}")
    print(f"       min={loader.stats.min_val:.4f}, max={loader.stats.max_val:.4f}")
    
    # Cleanup
    os.unlink(csv_path)
    
    print("\n✓ Demo complete")
