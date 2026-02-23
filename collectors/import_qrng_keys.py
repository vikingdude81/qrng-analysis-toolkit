"""
Import QRNG Hardware Keys
=========================

Converts base64-encoded 256-bit QRNG keys (from actual quantum hardware)
into the standard qrng_streams JSON format used by the HELIOS analysis pipeline.

Each 256-bit key yields 8 × 32-bit integers → 8 floats in [0, 1).
48,279 keys × 8 = 386,232 float values total.

We split into chunks of 1000 floats to match existing stream file sizes.

Input:  saved_keys.csv  (one base64 key per line, from Downloads)
Output: qrng_streams/qrng_stream_qhw_NNNN.json   (multiple stream files)
"""

import base64
import json
import struct
import sys
from datetime import datetime
import os
from pathlib import Path

INPUT_FILE = Path(os.environ.get("QRNG_KEYS_CSV", "saved_keys.csv"))
OUTPUT_DIR = Path(__file__).parent / "qrng_streams"
SOURCE_NAME = "quantum_hardware_256bit"
CHUNK_SIZE = 1000  # floats per stream file (matches existing streams)


def decode_key(b64_key: str) -> list[int]:
    """Decode a base64 256-bit key into 8 × 32-bit unsigned integers."""
    raw = base64.b64decode(b64_key.strip())
    if len(raw) != 32:
        raise ValueError(f"Expected 32 bytes, got {len(raw)}")
    # Unpack as 8 big-endian uint32
    return list(struct.unpack(">8I", raw))


def ints_to_floats(integers: list[int]) -> list[float]:
    """Convert 32-bit unsigned ints to floats in [0, 1)."""
    return [i / (2**32) for i in integers]


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Read all keys
    lines = INPUT_FILE.read_text().strip().splitlines()
    print(f"Loaded {len(lines)} base64 keys from {INPUT_FILE.name}")

    # Decode all keys → flat list of uint32 integers
    all_ints = []
    bad_keys = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            all_ints.extend(decode_key(line))
        except Exception as e:
            bad_keys += 1
            if bad_keys <= 3:
                print(f"  Warning: key {i} decode failed: {e}")

    all_floats = ints_to_floats(all_ints)
    print(f"Decoded {len(all_ints)} uint32 values → {len(all_floats)} floats")
    if bad_keys:
        print(f"  ({bad_keys} keys skipped due to errors)")

    # Split into chunks and write stream files
    n_files = 0
    for chunk_start in range(0, len(all_floats), CHUNK_SIZE):
        chunk_floats = all_floats[chunk_start : chunk_start + CHUNK_SIZE]
        chunk_ints = all_ints[chunk_start : chunk_start + CHUNK_SIZE]

        stream = {
            "timestamp": f"{timestamp}",
            "source": SOURCE_NAME,
            "bits_per_block": 32,
            "count": len(chunk_floats),
            "raw_integers": chunk_ints,
            "floats": chunk_floats,
            "encoding": "base64_256bit_to_uint32",
            "metadata": {
                "origin": "quantum_hardware",
                "original_key_bits": 256,
                "keys_per_chunk": CHUNK_SIZE // 8,
                "total_keys": len(lines),
                "total_floats": len(all_floats),
                "chunk_index": chunk_start // CHUNK_SIZE,
            },
        }

        fname = f"qrng_stream_qhw_{n_files:04d}.json"
        outpath = OUTPUT_DIR / fname
        outpath.write_text(json.dumps(stream, indent=2))
        n_files += 1

    print(f"\nWrote {n_files} stream files to {OUTPUT_DIR}")
    print(f"  Source tag: '{SOURCE_NAME}'")
    print(f"  Total floats: {len(all_floats)}")
    print(f"  Chunk size: {CHUNK_SIZE}")

    # Quick sanity stats
    import statistics
    sample = all_floats[:10000]
    print(f"\n--- Quick Sanity Check (first 10K floats) ---")
    print(f"  Mean:   {statistics.mean(sample):.6f}  (ideal: 0.500000)")
    print(f"  Stdev:  {statistics.stdev(sample):.6f}  (ideal: ~0.2887)")
    print(f"  Min:    {min(sample):.6f}")
    print(f"  Max:    {max(sample):.6f}")


if __name__ == "__main__":
    main()
