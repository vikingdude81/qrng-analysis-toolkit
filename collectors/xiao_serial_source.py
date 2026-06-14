"""
Seeed XIAO Serial QRNG Source
==============================
Hardware adapter for the XIAO SAMD21 QRNG device connected via USB serial.
Matches the SPDCQuantumSource interface so it slots into the Helios pipeline
as a drop-in real-hardware entropy source.

Integration with HELIOS:
    from collectors.xiao_serial_source import XIAOQuantumSource
    from helios_anomaly_scope import QRNGStreamScope          # if available
    from metrics.helios_anomaly_scope import QRNGStreamScope  # canonical path

    qrng = XIAOQuantumSource(port="COM4")
    qrng.connect()

    for _ in range(1000):
        value = qrng.get_random()        # float in [0, 1)
        raw   = qrng.get_bytes(32)       # raw quantum bytes
        seed  = qrng.get_seed()          # 256-bit int

    qrng.disconnect()

    # or as a context manager:
    with XIAOQuantumSource() as qrng:
        value = qrng.get_random()
"""

import struct
import threading
import time
from typing import Optional

import numpy as np
import serial


class XIAOQuantumSource:
    """
    Real-hardware QRNG source using the Seeed XIAO SAMD21 over USB serial.

    The XIAO firmware streams debiased entropy bytes at 115200 baud.
    This class runs a background thread that continuously drains the serial
    buffer into an internal bytearray, letting callers pull values without
    blocking on I/O.

    Interface is compatible with SPDCQuantumSource (get_random, get_bytes,
    get_seed, get_floats) so it works anywhere in the Helios pipeline.
    """

    SOURCE_NAME = "xiao_serial"
    SOURCE_DESCRIPTION = "Seeed XIAO SAMD21 hardware QRNG via USB serial"

    def __init__(self, port: str = "COM4", baud: int = 115200, buffer_min: int = 512):
        self.port = port
        self.baud = baud
        self._buffer_min = buffer_min
        self._buf: bytearray = bytearray()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._total_bytes = 0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def connect(self, timeout: float = 10.0) -> "XIAOQuantumSource":
        """Open the serial port and start the background reader."""
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="xiao-qrng")
        self._thread.start()
        deadline = time.monotonic() + timeout
        while len(self._buf) < self._buffer_min:
            if time.monotonic() > deadline:
                self._running = False
                raise TimeoutError(
                    f"XIAO QRNG on {self.port} did not produce {self._buffer_min} bytes "
                    f"within {timeout:.0f} s. Is the device plugged in?"
                )
            time.sleep(0.05)
        return self

    def disconnect(self):
        """Stop the background reader and release the port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def __enter__(self):
        return self.connect()

    def __exit__(self, *_):
        self.disconnect()

    # ------------------------------------------------------------------ #
    # Internal reader                                                      #
    # ------------------------------------------------------------------ #

    def _reader_loop(self):
        while self._running:
            try:
                with serial.Serial(self.port, self.baud, timeout=1) as ser:
                    while self._running:
                        chunk = ser.read(256)
                        if chunk:
                            with self._lock:
                                self._buf.extend(chunk)
                                self._total_bytes += len(chunk)
            except serial.SerialException as exc:
                if self._running:
                    print(f"[XIAO QRNG] Serial error: {exc}; retrying in 2 s…")
                    time.sleep(2)

    def _pull(self, n: int) -> bytes:
        """Block until n bytes are available, then return and remove them."""
        while True:
            with self._lock:
                if len(self._buf) >= n:
                    data = bytes(self._buf[:n])
                    del self._buf[:n]
                    return data
            time.sleep(0.005)

    # ------------------------------------------------------------------ #
    # Public API — matches SPDCQuantumSource interface                     #
    # ------------------------------------------------------------------ #

    def get_random(self) -> float:
        """Return one uniform float in [0, 1) from quantum hardware."""
        raw = self._pull(4)
        return struct.unpack(">I", raw)[0] / 2**32

    def get_bytes(self, n: int) -> bytes:
        """Return n raw quantum-random bytes."""
        return self._pull(n)

    def get_seed(self, bits: int = 256) -> int:
        """Return a quantum-random integer for seeding an RNG."""
        return int.from_bytes(self._pull(bits // 8), "big")

    def get_floats(self, count: int) -> np.ndarray:
        """Return a numpy array of `count` uniform floats in [0, 1)."""
        raw = self._pull(4 * count)
        ints = np.frombuffer(raw, dtype=">u4").astype(np.float64)
        return ints / 2**32

    def get_int(self, low: int, high: int) -> int:
        """Return a uniform integer in [low, high] with no modulo bias."""
        span = high - low + 1
        bits_needed = span.bit_length()
        n_bytes = max(1, (bits_needed + 7) // 8)
        mask = (1 << bits_needed) - 1
        while True:
            candidate = int.from_bytes(self._pull(n_bytes), "big") & mask
            if candidate < span:
                return low + candidate

    def get_normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        """Return a quantum-normal float via Box-Muller transform."""
        u1 = max(1e-10, self.get_random())
        u2 = self.get_random()
        z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        return mean + std * z

    # ------------------------------------------------------------------ #
    # Seeding helpers                                                      #
    # ------------------------------------------------------------------ #

    def seed_numpy(self) -> np.random.Generator:
        """Seed numpy's default RNG and return the Generator object."""
        seed = self.get_seed()
        np.random.seed(seed % (2**32))
        rng = np.random.default_rng(seed)
        return rng

    def seed_stdlib(self):
        """Seed Python's built-in random module."""
        import random
        random.seed(self.get_seed())

    def seed_torch(self):
        """Seed PyTorch RNG (CPU + CUDA if available)."""
        try:
            import torch
            torch.manual_seed(self.get_seed(64))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.get_seed(64))
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def total_bytes_received(self) -> int:
        return self._total_bytes

    def status(self) -> dict:
        return {
            "source": self.SOURCE_NAME,
            "port": self.port,
            "baud": self.baud,
            "running": self._running,
            "buffer_bytes": self.buffer_size,
            "total_bytes": self._total_bytes,
        }
