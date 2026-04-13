#!/usr/bin/env python3

import numpy as np
import sys

FRAME_BYTES = 576

device = "/dev/xillybus_read_32"
print(f"Opening {device}")

try:
    dev = open(device, 'rb')
except FileNotFoundError:
    print(f"{device} not found.")
    sys.exit(1)
except PermissionError:
    print(f"ERROR: Permission denied on {device}")
    sys.exit(1)

# Read one complete frame
print(f"Reading one frame {FRAME_BYTES}")
data = b''
while len(data) < FRAME_BYTES:
    chunk = dev.read(FRAME_BYTES - len(data))
    if not chunk:
        print("Stream closed before getting a full frame.")
        sys.exit(1)
    data += chunk
dev.close()

raw = np.frombuffer(data, dtype=np.uint32)

print(f"\nGot {len(raw)} words ({len(data)} bytes)")

print(raw)
# print(f"Any nonzero: {np.any(raw != 0)}")
# print(f"All 0xFFFFFFFF: {np.all(raw == 0xFFFFFFFF)}")
# print()

# print(f"First 9 words (Final_I_ch0):  {raw[0:9]}")
# print(f"Next 9 words  (Final_Q_ch0):  {raw[9:18]}")
# print(f"Next 9 words  (Final_I_ch1):  {raw[18:27]}")
# print()

# if np.any(raw != 0) and not np.all(raw == 0xFFFFFFFF):
#     print("SUCCESS: Data looks valid. You can run xillybus_music.py now.")
# elif np.all(raw == 0):
#     print("WARNING: All zeros. ADC data may not be connected,")
#     print("  or Cov_matrix may not be receiving samples yet.")
# else:
#     print("WARNING: All 0xFFFFFFFF — this usually means the FIFO")
#     print("  is not connected properly. Check the wiring in xillydemo.v.")
