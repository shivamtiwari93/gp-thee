"""How fast does this machine train the model? Measure before planning any run.

Run:  uv run python scripts/benchmark.py        (about five minutes; plug the laptop in first)

Times one full training step (forward pass, backward pass, optimizer update) at the batch size we will
train at, for every combination of:

    attention     the hand-written one in model.py, or PyTorch's built-in function
    number format 32-bit floats, or 16-bit "bfloat16" for the heavy arithmetic (the weights, their gradients
                  and the optimizer's averages stay 32-bit throughout)
    tokenizer     characters (98 pieces), or BPE with 2,048 pieces

A note on the two attentions. On Apple GPUs with PyTorch 2.14, the built-in function is a fast fused kernel
only when gradients are off. During training it runs the same separate steps as the hand-written version, so
in 32-bit the two rows measure the same operations and any difference between them is noise. In 16-bit the
built-in function quietly does its attention in 32-bit, which makes it slower and bigger than ours.

Every configuration is timed three times, in rotation, and the median is reported with the range. One pass in
a fixed order cannot tell a real 5% difference from the machine warming up.

Then it trains the same model for 300 steps in both number formats from the same starting point, with the
learning-rate warm-up a real run will have, to see whether the faster format learns the same way.

A GPU runs ahead of Python: a call returns before the work is done. So every timing here waits for the
GPU to finish (torch.mps.synchronize) before reading the clock. Forget that, and you measure how fast
Python can queue work, which is a much more flattering number.

Writes docs/benchmark.json.
"""

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from gp_thee.data import load_tokens, random_batch
from gp_thee.model import GPT, Config

ROOT = Path(__file__).resolve().parent.parent
GPU = "mps"
BATCH, CONTEXT = 64, 256  # 16,384 tokens per step
WARMUP, TIMED, ROUNDS = 5, 12, 3


def train_steps(model, stream, steps: int, device: str, sixteen_bit: bool, seed: int = 0, warm_up: int = 0, optimizer=None) -> list[float]:
    optimizer = optimizer or torch.optim.AdamW(model.parameter_groups(0.1), lr=1e-3, betas=(0.9, 0.99))
    rng, losses = np.random.default_rng(seed), []
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = 1e-3 * min(1.0, (step + 1) / warm_up) if warm_up else 1e-3
        tokens, targets = random_batch(stream, BATCH, CONTEXT, rng, device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=sixteen_bit):
            loss = model(tokens, targets)[1]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss)
    return [l.item() for l in losses]  # reading a loss waits for the GPU, so do it once, at the end


class Timed:
    """One configuration: built once, warmed up once, then timed whenever its turn comes round."""

    def __init__(self, tokenizer_name: str, device: str, builtin: bool, sixteen_bit: bool):
        self.stream, self.device, self.sixteen_bit = load_tokens(tokenizer_name, "train"), device, sixteen_bit
        self.row = {"tokenizer": tokenizer_name, "device": device, "attention": "built-in" if builtin else "hand-written",
                    "numbers": "16-bit" if sixteen_bit else "32-bit"}
        torch.manual_seed(0)
        self.model = GPT(Config(vocab_size=int(self.stream[0]) + 1, builtin_attention=builtin)).to(device).train()
        self.optimizer = torch.optim.AdamW(self.model.parameter_groups(0.1), lr=1e-3, betas=(0.9, 0.99))
        self.seconds, self.peak = [], 0.0
        self.run(WARMUP if device == GPU else 1)

    def run(self, steps: int) -> float:
        if self.device == GPU:
            torch.mps.synchronize()
        started = time.perf_counter()
        train_steps(self.model, self.stream, steps, self.device, self.sixteen_bit, optimizer=self.optimizer)
        if self.device == GPU:
            torch.mps.synchronize()
        return (time.perf_counter() - started) / steps

    def measure_memory(self) -> None:
        """Memory in use when it is highest: right after the forward pass, before the backward pass frees it."""
        tokens, targets = random_batch(self.stream, BATCH, CONTEXT, np.random.default_rng(1), self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.sixteen_bit):
            loss = self.model(tokens, targets)[1]
        torch.mps.synchronize()
        self.peak = torch.mps.current_allocated_memory() / 1e9
        loss.backward()
        self.optimizer.zero_grad(set_to_none=True)


chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[0].strip()
print(f"{chip}, macOS {platform.mac_ver()[0]}, torch {torch.__version__}; batch {BATCH} x {CONTEXT} = {BATCH * CONTEXT:,} tokens per step")
print(f"power: {power}\n")
if "Battery" in power:
    print("  WARNING: running on battery. A laptop on battery may run slower; plug in before quoting these numbers.\n")

configs = [Timed(name, GPU, builtin, sixteen_bit) for name in ("char", "bpe-2048") for builtin in (False, True) for sixteen_bit in (False, True)]
for _ in range(ROUNDS):  # in rotation, so that the machine warming up or cooling down hits every configuration alike
    for config in configs:
        config.seconds.append(config.run(TIMED))
for config in configs:
    config.measure_memory()
cpu = Timed("char", "cpu", False, False)
cpu.seconds = [cpu.run(3)]

print(f"{'tokenizer':<10}{'device':<7}{'attention':<14}{'numbers':<9}{'ms/step (range)':>22}{'tokens/s':>11}{'memory in use':>15}{'5,000 steps':>13}")
rows = []
for config in configs + [cpu]:
    ms = sorted(1000 * t for t in config.seconds)
    median = ms[len(ms) // 2]
    row = {**config.row, "ms_per_step": round(median, 1), "ms_per_step_range": [round(ms[0], 1), round(ms[-1], 1)],
           "tokens_per_second": round(BATCH * CONTEXT / median * 1000), "gpu_memory_in_use_gb": round(config.peak, 2) if config.peak else None}
    rows.append(row)
    memory = f"{config.peak:>12.2f} GB" if config.peak else f"{'':>15}"
    print(f"{row['tokenizer']:<10}{row['device']:<7}{row['attention']:<14}{row['numbers']:<9}{median:>9.1f} ({ms[0]:.0f}-{ms[-1]:.0f}){row['tokens_per_second']:>11,}{memory}{median * 5:>11.0f} s")
del configs, cpu
torch.mps.empty_cache()

# Does 16-bit learn the same way? Same starting weights, same batches, 300 steps, with a 100-step warm-up.
# (Without the warm-up, a learning rate of 1e-3 from the first step sends the loss from 3.4 to 7.2 at step 5,
# and a comparison made through a spike like that mostly measures the spike.)
print("\nsame start, same batches, 300 steps (characters, hand-written attention, no dropout):")
stream, curves = load_tokens("char", "train"), {}
for sixteen_bit in (False, True):
    torch.manual_seed(0)
    model = GPT(Config(vocab_size=int(stream[0]) + 1, dropout=0.0)).to(GPU).train()  # no dropout, so the two runs see the same thing
    curves["16-bit" if sixteen_bit else "32-bit"] = train_steps(model, stream, 300, GPU, sixteen_bit, warm_up=100)
for step in (0, 9, 49, 99, 199, 299):
    print(f"  step {step + 1:>3}: loss {curves['32-bit'][step]:.4f} in 32-bit, {curves['16-bit'][step]:.4f} in 16-bit")
late = np.mean(curves["16-bit"][200:]) - np.mean(curves["32-bit"][200:])
print(f"  over steps 201-300, 16-bit is {late:+.4f} nats against 32-bit. (Two 32-bit runs from the same seed also differ a little on this GPU.)")

result = {"machine": chip, "power": power, "macos": platform.mac_ver()[0], "torch": torch.__version__, "batch": BATCH, "context": CONTEXT,
          "note": f"one full training step: forward, backward, gradient clipping, AdamW update; dropout 0.2; median of {ROUNDS} interleaved rounds of {TIMED} steps",
          "timings": rows,
          "loss_32_vs_16_bit": {k: [round(v, 4) for v in curve] for k, curve in curves.items()}}
(ROOT / "docs" / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote docs/benchmark.json")
