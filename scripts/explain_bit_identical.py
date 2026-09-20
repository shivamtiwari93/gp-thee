"""Why does a matrix multiply give bit-identical results on this Mac's CPU and GPU?

Run:  uv run python scripts/explain_bit_identical.py   (takes about a minute)

Two devices usually disagree in the last digit of a matrix multiply, because each
output is a sum of thousands of products and the rounded total depends on the order
of the additions. scripts/smoke_test_gpu.py found no disagreement at all here.

This script tests one explanation: both devices add the products in index order,
using a fused multiply-add (a*b+c with one rounding instead of two). It recomputes
60 of the 4 million outputs that way, in plain Python, and counts exact matches.
Reverse order, and multiply-then-add, are the controls.
"""
import math, random, numpy as np, torch
torch.manual_seed(0)
a, b = torch.randn(2048, 2048), torch.randn(2048, 2048)
cpu = (a @ b).numpy(); gpu = (a.to("mps") @ b.to("mps")).cpu().numpy()
A, B = a.numpy().astype(np.float64), b.numpy().astype(np.float64)
f32 = lambda x: float(np.float32(x))
def dot(i, j, order, fused):
    acc = 0.0
    for k in order:
        acc = f32(math.fma(A[i, k], B[k, j], acc)) if fused else f32(f32(A[i, k] * B[k, j]) + acc)
    return np.float32(acc)
random.seed(1); cells = [(random.randrange(2048), random.randrange(2048)) for _ in range(60)]
fwd, rev = range(2048), range(2047, -1, -1)
for name, order, fused in [("index order, fused multiply-add", fwd, True), ("reverse order, fused multiply-add", rev, True), ("index order, separate multiply then add", fwd, False)]:
    hits_cpu = sum(dot(i, j, order, fused) == cpu[i, j] for i, j in cells)
    hits_gpu = sum(dot(i, j, order, fused) == gpu[i, j] for i, j in cells)
    print(f"{name:<42} matches cpu {hits_cpu}/60 | matches gpu {hits_gpu}/60")
