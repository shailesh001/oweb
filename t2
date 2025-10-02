"""Simple reproducibility/sanity check for matrix multiplication on MPS.

This script creates two random matrices on the Apple Metal Performance Shaders
(MPS) device with half precision (float16), computes their matrix product once
and stores it as a reference. It then recomputes the product 1000 times and
asserts that the result exactly matches the reference every time. The intent is
to detect non-determinism or unexpected variation in repeated matrix
multiplications on the MPS backend.

Notes:
- Using float16 (half precision) can be faster on supported hardware but is
    sensitive to numerical differences; this script requires exact equality so
    only bit-identical results will pass.
- The script intentionally uses an assert with an exact max absolute difference
    of 0 to flag any deviation.
"""

import torch


# Create two random 2048x2048 matrices on the MPS device using float16
# torch.randn draws samples from a normal distribution (mean=0, std=1).
A = torch.randn(2048, 2048, device='mps', dtype=torch.float16)
B = torch.randn(2048, 2048, device='mps', dtype=torch.float16)

# Compute a reference matrix multiplication result once
ref = torch.mm(A, B)

# Repeat the multiplication multiple times and ensure results match exactly.
# If any run differs, the assert will raise an AssertionError.
for _ in range(1000):
        diff = (torch.mm(A, B) - ref).abs().max().item()
        # Use == 0 to require exact equality (no numerical difference)
        assert diff == 0
