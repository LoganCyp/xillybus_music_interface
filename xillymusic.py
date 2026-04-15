#!/usr/bin/env python3
"""
xillybus_music.py — Real-time MUSIC DOA from Xillybus PCIe stream.

Receives two 8x8 matrices of 64-bit signed integers (I then Q),
combines them into a complex covariance matrix, runs MUSIC.

Frame format (1024 bytes):
  Bytes    0-511:  I matrix (real part) — 64 int64 values in row-major order
  Bytes  512-1023: Q matrix (imag part) — 64 int64 values in row-major order
"""

import numpy as np
import time
import sys
from music_api import run_music

CH_COUNT     = 8
MATRIX_ELEMS = 64          # 8 × 8
MATRIX_BYTES = 512         # 64 elements × 8 bytes each
FRAME_BYTES  = 1024        # I matrix + Q matrix


def read_frame(dev):
    """Read exactly 1024 bytes (one I matrix + one Q matrix)."""
    data = b''
    while len(data) < FRAME_BYTES:
        chunk = dev.read(FRAME_BYTES - len(data))
        if not chunk:
            raise IOError("Xillybus stream closed")
        data += chunk
    return data


def unpack_to_cov(data):
    """Unpack 1024 bytes into an (8, 8) complex covariance matrix."""
    all_vals = np.frombuffer(data, dtype=np.int64)
    Q_matrix = all_vals[:MATRIX_ELEMS].reshape(CH_COUNT, CH_COUNT)
    I_matrix = all_vals[MATRIX_ELEMS:].reshape(CH_COUNT, CH_COUNT)
    
    return I_matrix.astype(np.float64) + 1j * Q_matrix.astype(np.float64)


def main():
    device = "/dev/xillybus_read_32"
    print(f"Opening {device} ...")

    try:
        dev = open(device, 'rb')
    except FileNotFoundError:
        print(f"ERROR: {device} not found.")
        print("  Run: lspci | grep -i xill")
        print("  Run: sudo modprobe xillybus_pcie")
        sys.exit(1)
    except PermissionError:
        print(f"ERROR: Permission denied. Run: sudo chmod 666 {device}")
        sys.exit(1)

    print("Connected. Running MUSIC... (Ctrl+C to stop)\n")
    count = 0
    t0 = time.time()

    try:
        while True:
            R = unpack_to_cov(read_frame(dev))
            angles, powers_db, d_sig, spectrum = run_music(R)
            count += 1

            if count % 100 == 0:
                fps = count / (time.time() - t0)
                print(f"Frame {count:6d} | {fps:.1f} fps | "
                      f"{d_sig} source(s):")
                for a, p in zip(angles, powers_db):
                    print(f"    theta={a[0]:+7.2f} deg  "
                          f"phi={a[1]:+7.2f} deg  "
                          f"power={p:.1f} dB")
                print()
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n{count} frames in {elapsed:.1f}s "
              f"({count/elapsed:.1f} fps)")
    finally:
        dev.close()


if __name__ == "__main__":
    main()