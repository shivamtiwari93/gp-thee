"""Does the Mac GPU compute the same answers as the CPU?

Run:  uv run python scripts/smoke_test_gpu.py

Compares the operations a GPT is made of on the CPU and on the GPU ("mps"),
and checks both against a 64-bit reference. It also runs a control that MUST
show a difference, to prove the comparison is able to fail.

This is a smoke test, not proof. The full check, one real training batch
through the whole model on both devices, comes with the model.
"""

import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)
GPU = "mps"


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    # Move to the CPU first: the GPU has no 64-bit floats.
    return (a.cpu().double() - b.cpu().double()).abs().max().item()


def report(name: str, cpu: torch.Tensor, gpu: torch.Tensor, ref: torch.Tensor) -> None:
    assert cpu.device.type == "cpu" and gpu.device.type == GPU, "results are not on the devices we think"
    assert not torch.isnan(gpu).any(), "NaN on the GPU"
    print(f"{name:<34} gpu vs cpu {max_diff(gpu, cpu):.2e} | cpu vs 64-bit {max_diff(cpu, ref):.2e} "
          f"| gpu vs 64-bit {max_diff(gpu, ref):.2e} | bit-identical: {torch.equal(gpu.cpu(), cpu)}")


print(f"torch {torch.__version__} | mps available: {torch.backends.mps.is_available()}\n")

# 1. Matrix multiply: the operation a GPT spends nearly all its time in.
a, b = torch.randn(2048, 2048), torch.randn(2048, 2048)
report("matrix multiply 2048x2048", a @ b, a.to(GPU) @ b.to(GPU), a.double() @ b.double())

# 2. Matrix multiply with a transposed left side: the shape of attention's q @ k^T,
#    and the case hit by pytorch/pytorch#193487 on versions up to 2.13.0.
report("matrix multiply, transposed left", a.T @ b, a.to(GPU).T @ b.to(GPU), a.double().T @ b.double())

# 3. LayerNorm -> Linear -> GELU: the rest of a transformer block.
x = torch.randn(64, 256, 384)
ln, lin = torch.nn.LayerNorm(384, bias=False), torch.nn.Linear(384, 1536, bias=False)
block = lambda t, n, l: F.gelu(l(n(t)))
ref = block(x.double(), ln.double(), lin.double())
ln, lin = ln.float(), lin.float()
cpu = block(x, ln, lin)
gpu = block(x.to(GPU), ln.to(GPU), lin.to(GPU))
report("layernorm -> linear -> gelu", cpu.detach(), gpu.detach(), ref.detach())

# 4. Causal attention, in the shape GP-Thee will use: 6 heads of 64, context 256.
q, k, v = (torch.randn(8, 6, 256, 64) for _ in range(3))
attend = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
report("causal attention", attend(q, k, v), attend(q.to(GPU), k.to(GPU), v.to(GPU)),
       attend(q.double(), k.double(), v.double()))

# 5. Does causal attention leak the future? Change the LAST token only: every earlier
#    position must be unchanged. (pytorch/pytorch#195910 leaked in 16-bit on <= 2.12.)
k2, v2 = k.clone(), v.clone()
k2[:, :, -1], v2[:, :, -1] = 0.0, 0.0
before = attend(q.to(GPU), k.to(GPU), v.to(GPU))[:, :, :-1]
after = attend(q.to(GPU), k2.to(GPU), v2.to(GPU))[:, :, :-1]
print(f"{'causal leak (must be 0)':<34} {max_diff(before, after):.2e}")

# 6. Control: a comparison that MUST differ, to prove this script can report a difference.
control = max_diff(a.to(GPU) @ b.to(GPU), (a + 1e-3) @ b)
assert control > 0, "the control shows no difference: the comparison itself is broken"
print(f"{'control (must NOT be 0)':<34} {control:.2e}")

# 7. Speed. GPU work is queued, so wait for it to finish before reading the clock.
def time_matmuls(device: str) -> float:
    m, n = a.to(device), b.to(device)
    m @ n  # warm-up
    if device == GPU:
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        m @ n
    if device == GPU:
        torch.mps.synchronize()
    return (time.perf_counter() - start) * 1000


print(f"\n50 matrix multiplies: cpu {time_matmuls('cpu'):.0f} ms | gpu {time_matmuls(GPU):.0f} ms")
