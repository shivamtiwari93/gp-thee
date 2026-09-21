"""The correctness gate: run this before trusting any training run.

Run:  uv run python scripts/check_model.py        (a minute or two on an M5 Max)

First it runs the unit tests (uv run pytest), which check a tiny model on the CPU against definitions and
against an independent reference. If they fail, it stops. Then it checks the REAL model, at full size, on the
GPU, with real Shakespeare, in both number formats we might train in. Each check targets a specific way a
from-scratch model goes wrong without announcing it:

  1. size            the parameter count is the number derived in the blog, and the output table IS the input table
  2. first loss      an untrained model should be as surprised as theory says: ln(vocabulary) + 0.02^2 x width / 2
  3. no peeking      changing later tokens must not change earlier predictions AT ALL. Every cut point, both
                     attentions, 32-bit and 16-bit, and in BOTH modes: gradients off (evaluation, sampling) and
                     gradients on (training). On Apple GPUs those two modes run different code inside PyTorch,
                     and a PyTorch bug once let 16-bit attention see the future.
  4. two attentions  the hand-written attention and PyTorch's built-in one agree, in both modes
  5. GPU vs CPU      the same batch gives the same predictions and the same gradients on both
  6. memorise        the full model can learn one batch by heart, so model, loss and optimizer are wired up
  7. same answers    with the memorised weights (attention is sharp now, so faults show): every combination of
                     attention, number format and device gives the same loss, on rows it memorised and rows it
                     never saw. This is the check that would catch a fault that exists only in 16-bit.

With --trained runs/<name>/best.pt it instead repeats check 3 on TRAINED weights and text the model has never seen,
and then compares the loss at every token between every combination of attention, device and number format. A
fresh model attends to everything about equally, which hides some faults. A trained one has learned sharp habits
of attention, so if the future can leak anywhere, this is where it shows.

Nothing is written. Any failure exits with an error.
"""

import argparse
import math
import subprocess
import sys
import time

import numpy as np
import torch

from gp_thee.data import load_tokens, random_batch
from gp_thee.model import GPT, Config
from gp_thee.train import load_checkpoint

GPU = "mps"
EXPECTED_PARAMETERS = {"char": 10_757_760, "bpe-2048": 11_506_560}  # derived by hand in blog parts 1 and 3
failures = []


def report(name: str, ok: bool, detail: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<52}{detail}")
    if not ok:
        failures.append(name)


def build(vocab_size: int, device: str, **changes) -> GPT:
    torch.manual_seed(0)  # same seed, same starting weights, on any device
    return GPT(Config(vocab_size=vocab_size, dropout=0.0, **changes)).to(device)


def largest_leak(model: GPT, row: torch.Tensor, vocab_size: int, rng, sixteen_bit: bool, training: bool) -> tuple[float, int]:
    """For EVERY cut point k, scramble positions k.. of `row` and see whether any score before k moves.

    Each batch holds the untouched row first and scrambled copies after it, so both are computed in the same
    call, with the same shapes. Returns the largest change seen in the past, and how many cut points changed
    the future (all of them should: that is the control).
    """
    length, worst, controls = len(row), 0.0, 0
    model.train(training)
    cuts = list(range(1, length))
    for begin in range(0, len(cuts), 63):
        chunk = cuts[begin:begin + 63]
        rows = row.repeat(len(chunk) + 1, 1)
        for i, k in enumerate(chunk, start=1):
            # every scrambled token is moved by 1 to V-2 places round the vocabulary (START left out), so it is certain to differ
            rows[i, k:] = (row[k:] + torch.from_numpy(rng.integers(1, vocab_size - 1, size=length - k))) % (vocab_size - 1)
        with torch.set_grad_enabled(training), torch.autocast(GPU, dtype=torch.bfloat16, enabled=sixteen_bit):
            scores = model(rows.to(GPU))[0].detach().float().cpu()
        for i, k in enumerate(chunk, start=1):
            worst = max(worst, (scores[0, :k] - scores[i, :k]).abs().max().item())
            controls += int(not torch.equal(scores[0, k:], scores[i, k:]))
    return worst, controls


def check_trained(path: str) -> None:
    _, saved = load_checkpoint(path, "cpu")
    name, context = saved["run_config"]["tokenizer"], saved["model_config"]["context"]
    stream = load_tokens(name, "validation")
    vocab_size, rng = saved["model_config"]["vocab_size"], np.random.default_rng(0)
    tokens, targets = random_batch(stream, 8, context, rng, "cpu")
    print(f"{path}: step {saved['step']:,}, {name} tokenizer. No peeking: one window of the validation works. Same answers: eight.\n")

    def trained(device: str, builtin: bool) -> GPT:
        # dropout off: with it on, two copies of the same row would differ by chance and hide a real leak
        model = GPT(Config(**{**saved["model_config"], "dropout": 0.0, "builtin_attention": builtin})).to(device)
        model.load_state_dict(saved["model"])
        return model

    losses = {}  # for each combination, the loss at every one of the 8 x 256 tokens
    for builtin in (False, True):
        model = trained(GPU, builtin)
        for sixteen_bit in (False, True):
            for training in (False, True):
                worst, controls = largest_leak(model, tokens[0], vocab_size, rng, sixteen_bit, training)
                title = f"3. no peeking ({'built-in' if builtin else 'hand-written'}, {'16' if sixteen_bit else '32'}-bit, {'training' if training else 'evaluation'})"
                report(title, worst == 0.0 and controls == context - 1, f"{context - 1} cut points, largest change in the past {worst:.1e}")
        for device in ("cpu", GPU):
            for sixteen_bit in ((False,) if device == "cpu" else (False, True)):
                candidate = trained(device, builtin).eval()
                with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=sixteen_bit):
                    scores = candidate(tokens.to(device))[0].float().cpu()
                losses[(builtin, device, sixteen_bit)] = torch.nn.functional.cross_entropy(scores.view(-1, vocab_size), targets.view(-1), reduction="none")
    reference = losses[(False, GPU, False)]  # what training and evaluation use: hand-written attention, GPU, 32-bit
    gap_32 = max((losses[k] - reference).abs().max().item() for k in losses if not k[2])
    gap_16 = max((losses[k] - reference).abs().max().item() for k in losses)
    mean_16 = max(abs(losses[k].mean().item() - reference.mean().item()) for k in losses)
    # A mean over 2,048 tokens would hide one token that moved by 0.2 nats, so the 32-bit limit is on the worst single token.
    # 16-bit rounds differently at every token, so there the limit on a single token is loose and the limit on the mean is tight.
    # Measured on a 34-pass character model: 1.2e-5 at the worst token in 32-bit; 0.099 at the worst token and 5.3e-4 in the mean with 16-bit.
    report("same loss at every token, both attentions, GPU, CPU", gap_32 < 1e-4 and gap_16 < 0.5 and mean_16 < 2e-3,
           f"mean loss {reference.mean():.4f}; largest difference at any one token {gap_32:.1e} in 32-bit, {gap_16:.1e} with 16-bit (means within {mean_16:.1e})")


parser = argparse.ArgumentParser(description="The correctness gate.")
parser.add_argument("--trained", help="a checkpoint, e.g. runs/pilot-char-17/best.pt: repeat the attention checks on trained weights")
args = parser.parse_args()
if args.trained:
    if not torch.backends.mps.is_available():
        sys.exit("this needs an Apple GPU")
    check_trained(args.trained)
    if failures:
        sys.exit(f"{len(failures)} check(s) FAILED: {failures}")
    print("\nall checks passed")
    sys.exit(0)

print("unit tests first:")
tests = subprocess.run(["uv", "run", "pytest", "-q", "-x"], capture_output=True, text=True)
print("  " + tests.stdout.strip().splitlines()[-1])
if tests.returncode:
    sys.exit("the unit tests failed; fix those before looking at the GPU")
print(f"\ntorch {torch.__version__}, GPU available: {torch.backends.mps.is_available()}\n")
if not torch.backends.mps.is_available():
    sys.exit("the rest of this script needs an Apple GPU")

for tokenizer_name in ("char", "bpe-2048"):
    stream = load_tokens(tokenizer_name, "train")
    vocab_size = int(stream[0]) + 1  # every stream starts with START, which is the last id
    if int(stream.max()) >= vocab_size:
        sys.exit(f"{tokenizer_name}: the stream contains ids outside the vocabulary")
    rng = np.random.default_rng(0)
    tokens, targets = random_batch(stream, 8, 256, rng, "cpu")
    print(f"{tokenizer_name}: vocabulary {vocab_size:,}")

    # 1. size
    model = build(vocab_size, GPU).eval()
    counted = sum(p.numel() for p in model.parameters())
    report("1. size, and one table not two", counted == EXPECTED_PARAMETERS[tokenizer_name] and model.to_scores.weight is model.token_embedding.weight,
           f"{counted:,} parameters")

    # 2. first loss, on uniformly random tokens, where theory gives the answer
    expected = math.log(vocab_size) + 0.02 ** 2 * model.config.width / 2
    random_tokens = torch.from_numpy(rng.integers(0, vocab_size, size=(16, 256)))
    with torch.no_grad():
        first = model(random_tokens.to(GPU), random_tokens.roll(-1, dims=1).to(GPU))[1].item()
        on_text = model(tokens.to(GPU), targets.to(GPU))[1].item()
    report("2. first loss is what theory predicts", abs(first - expected) < 0.03, f"{first:.3f} against {expected:.3f} (on real text: {on_text:.3f})")

    # 3. no peeking: every cut point, both attentions, both number formats, both modes
    for builtin in (False, True):
        model = build(vocab_size, GPU, builtin_attention=builtin)
        for sixteen_bit in (False, True):
            for training in (False, True):
                worst, controls = largest_leak(model, tokens[0], vocab_size, rng, sixteen_bit, training)
                name = f"3. no peeking ({'built-in' if builtin else 'hand-written'}, {'16' if sixteen_bit else '32'}-bit, {'training' if training else 'evaluation'})"
                report(name, worst == 0.0 and controls == 255, f"255 cut points, largest change in the past {worst:.1e}")

    # 4. the two attentions agree, with gradients off and on
    by_hand, built_in = build(vocab_size, GPU, builtin_attention=False), build(vocab_size, GPU, builtin_attention=True)
    gaps = []
    for training in (False, True):
        with torch.set_grad_enabled(training):
            gaps.append((by_hand.train(training)(tokens.to(GPU))[0] - built_in.train(training)(tokens.to(GPU))[0]).abs().max().item())
    report("4. hand-written attention = built-in", max(gaps) < 1e-4, f"largest difference in scores: {max(gaps):.1e}")

    # 5 to 7 share one helper: loss and gradients of a model on a batch
    def loss_and_gradients(model, batch_tokens, batch_targets, sixteen_bit=False):
        device = next(model.parameters()).device.type
        model.zero_grad()
        with torch.autocast(device, dtype=torch.bfloat16, enabled=sixteen_bit):
            scores, loss = model(batch_tokens.to(device), batch_targets.to(device))
        loss.backward()
        gradients = [p.grad for p in model.parameters()]
        return scores.detach().float().cpu(), loss.item(), (None if any(g is None for g in gradients) else [g.cpu() for g in gradients])

    def compare_gpu_with_cpu(label: str, state: dict | None) -> None:
        for builtin in (False, True):
            on_cpu, on_gpu = build(vocab_size, "cpu", builtin_attention=builtin), build(vocab_size, GPU, builtin_attention=builtin)
            if state is not None:
                on_cpu.load_state_dict(state)
                on_gpu.load_state_dict(state)
            scores_cpu, _, grads_cpu = loss_and_gradients(on_cpu, tokens, targets)
            scores_gpu, _, grads_gpu = loss_and_gradients(on_gpu, tokens, targets)
            if grads_cpu is None or grads_gpu is None:
                report(f"{label} ({'built-in' if builtin else 'hand-written'})", False, "a parameter received no gradient")
                continue
            score_gap = (scores_cpu - scores_gpu).abs().max().item()
            grad_gap = max(((a - b).abs().max() / a.abs().max()).item() for a, b in zip(grads_cpu, grads_gpu))
            report(f"{label} ({'built-in' if builtin else 'hand-written'})", score_gap < 1e-4 and grad_gap < 1e-4,
                   f"scores differ by {score_gap:.1e}, gradients by {grad_gap:.1e} of their size")

    compare_gpu_with_cpu("5. GPU = CPU, fresh weights", None)

    # 6. memorise the first four rows of the batch. (A gentle learning rate: at 1e-3 this check was a coin toss.)
    model = build(vocab_size, GPU)
    optimizer = torch.optim.AdamW(model.parameter_groups(0.0), lr=3e-4)
    known, known_targets = tokens[:4].to(GPU), targets[:4].to(GPU)
    started_at = model(known, known_targets)[1].item()
    started = time.perf_counter()
    for _ in range(200):
        optimizer.zero_grad()
        loss = model(known, known_targets)[1]
        loss.backward()
        optimizer.step()
    torch.mps.synchronize()
    report("6. memorises one batch", loss.item() < 0.1, f"loss {started_at:.2f} -> {loss.item():.4f} in 200 steps, {time.perf_counter() - started:.0f} s")

    # 7. the memorised weights give the same answers everywhere
    memorised = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    compare_gpu_with_cpu("7a. GPU = CPU, memorised weights", memorised)
    for rows, which in ((slice(0, 4), "memorised rows"), (slice(4, 8), "unseen rows")):
        losses = {}
        for builtin in (False, True):
            for device in ("cpu", GPU):
                for sixteen_bit in ((False,) if device == "cpu" else (False, True)):
                    candidate = build(vocab_size, device, builtin_attention=builtin).eval()
                    candidate.load_state_dict(memorised)
                    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=sixteen_bit):
                        losses[(builtin, device, sixteen_bit)] = candidate(tokens[rows].to(device), targets[rows].to(device))[1].item()
        gap_32 = max(v for k, v in losses.items() if not k[2]) - min(v for k, v in losses.items() if not k[2])
        gap_16 = max(losses.values()) - min(losses.values())
        report(f"7b. same loss everywhere ({which})", gap_32 < 1e-4 and gap_16 < 0.02,
               f"loss {losses[(False, GPU, False)]:.4f}; 32-bit spread {gap_32:.1e}, with 16-bit {gap_16:.1e}")
    print()

if failures:
    sys.exit(f"{len(failures)} check(s) FAILED: {failures}")
print("all checks passed")
