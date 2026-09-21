"""Training: the loop, the learning-rate schedule, honest evaluation, checkpoints, and a small sampler.

The heart of the loop is six short lines that every PyTorch training script shares: take a batch, run the
model and measure the loss, forget the last step's gradients, work out the new ones, clip them, nudge the
parameters. They carry comments in train(). Everything else in this file exists to answer one of three
questions truthfully:

  * How good is the model on text it has never seen?     -> evaluate(), bits_per_character()
  * Which moment of the run was the best one?            -> the best checkpoint, chosen on validation
  * Could someone else, or we ourselves next month, get this run again?   -> RunConfig, checkpoints, the log

The LENGTH of a run is counted in PASSES over the training text, never in steps or "epochs". We cut windows
from random places, so there are no epochs, and the same number of steps is 17 passes over the character
stream but 54 over the shortest word-fragment stream. One pass means: as many tokens as the stream holds.
(Not: every token once. Windows are cut at random, so in one pass about 37% of the text is not read at all
and about 26% is read more than once.) The evaluations are spread evenly over the run for the same reason.
One setting is still in steps, on purpose: the warm-up, which is about the optimizer settling down and not
about the text.
"""

import csv
import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gp_thee.data import DATA, load_tokens, random_batch
from gp_thee.model import GPT, Config
from gp_thee.tokenizer import load as load_tokenizer

ROOT = DATA.parent
PROMPT = "\n\nHAMLET.\n"  # every sample of every run starts here: a speaker label on a line of its own
LOG_COLUMNS = ["step", "passes", "learning_rate", "training_loss", "gradient_length", "share_clipped", "seen_bpc", "validation_bpc",
               "validation_loss", "tokens_per_second", "minutes"]


@dataclasses.dataclass(frozen=True)
class RunConfig:
    name: str                          # the run's folder under runs/
    tokenizer: str = "char"            # which saved tokenizer, and so which token streams
    seed: int = 0                      # fixes the starting weights, the batches and the dropout masks
    passes: float = 17.0               # how long to train, in passes over the training stream
    batch: int = 64                    # windows per step
    peak_rate: float = 1e-3            # learning rate after the warm-up
    floor_rate: float = 1e-4           # learning rate at the end of the run
    warm_up: int = 100                 # STEPS spent climbing to the peak rate. Without them the loss lurches about in the first ten steps
    weight_decay: float = 0.1          # applied to matrices and tables, never to norm scales
    beta_2: float = 0.99               # AdamW's average of squared gradients remembers about 1 / (1 - beta_2) = 100 steps (PyTorch's default 0.999: 1,000)
    clip: float = 1.0                  # if all the gradients, taken as one long vector, are longer than this, they are scaled down to it
    dropout: float = 0.2               # the share of values switched off at random while training, so the model cannot lean on any one of them
    layers: int = 6                    # the model's shape: these four are handed to model.Config
    heads: int = 6
    width: int = 384
    context: int = 256                 # tokens per window
    evaluations: int = 40              # how many times to stop and score the validation works, evenly spread over the run (plus once before step 1)
    sixteen_bit: bool = False          # do the heavy arithmetic in bfloat16. Faster, not yet shown to learn the same way
    device: str = "mps"                # "mps" is the Mac's GPU; "cpu" works everywhere, about eight times slower (docs/benchmark.json)

    def steps(self, training_tokens: int) -> int:
        return max(1, round(self.passes * training_tokens / (self.batch * self.context)))

    def evaluation_steps(self, last_step: int) -> list[int]:
        """The same number of stops for every run, however many steps it has, so that a short run and a long one
        get the same number of chances to catch their best moment."""
        return sorted({round(i * last_step / self.evaluations) for i in range(self.evaluations + 1)})


# ---------------------------------------------------------------------------------------------- schedule
def learning_rate(step: int, warm_up: int, last_step: int, peak: float, floor: float) -> float:
    """Climb in a straight line to `peak` over the warm-up, then follow half a cosine wave down to `floor`.

    Big steps early throw the fresh model around. (Seen in the benchmark: with 1e-3 from the first step the loss
    went from 3.4 to 7.2 at step 5. Where and how far it jumps depends on the seed.) So the rate starts near zero.
    Big steps late would stop the model settling, so the rate tapers off.
    """
    if step < warm_up:
        return peak * (step + 1) / warm_up
    if step >= last_step:
        return floor
    progress = (step - warm_up) / max(1, last_step - warm_up)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------------------------- evaluation
def plan_windows(targets: int, context: int, stride: int) -> list[tuple[int, int]]:
    """Where to put the windows so that every target is scored EXACTLY once, with as much text behind it as possible.

    Returns (begin, first_scored) pairs. The window at `begin` reads tokens begin .. begin + context - 1, and we
    keep only its predictions from position `first_scored` onwards. The first window keeps everything. Each later
    one overlaps the one before and keeps only its last `stride` positions, so everything it scores has at least
    context - stride tokens of text behind it. A last window is squeezed in at the very end if anything is left.
    """
    if targets <= context:
        return [(0, 0)]
    plan, scored = [(0, 0)], context
    while scored + stride <= targets:
        begin = scored + stride - context
        plan.append((begin, context - stride))
        scored += stride
    if scored < targets:
        begin = targets - context
        plan.append((begin, scored - begin))
    return plan


@torch.no_grad()
def evaluate(model: GPT, stream: np.ndarray, device: str, stride: int | None = None, batch: int = 64) -> np.ndarray:
    """The surprise, in nats, at every token of `stream` after the first. One number per token, each scored once.

    Always the same windows in the same order, always 32-bit, dropout off. Summing the result and dividing ONCE
    is how every validation figure in this project is made. Averaging the averages of unequal batches gives a
    different, wrong, number: surprises [1, 1, 1] in one batch and [5] in another average to 2, but the average
    of the two batch averages is 3. (The training_loss column of the log is the one exception: a plain mean over
    equal-sized training batches, with dropout on. It is there to watch, not to report.)
    """
    context = model.config.context
    stride = context // 2 if stride is None else stride
    if not 0 < stride <= context:
        raise ValueError(f"a stride of {stride} makes no sense with a context of {context}")
    was_training = model.training
    model.eval()
    try:
        tokens = torch.from_numpy(np.asarray(stream).astype(np.int64))
        targets = len(tokens) - 1
        length = min(context, targets)
        plan = plan_windows(targets, context, stride)
        surprise, times_scored = np.zeros(targets), np.zeros(targets, dtype=int)
        for i in range(0, len(plan), batch):
            chunk = plan[i:i + batch]
            windows = torch.stack([tokens[b:b + length] for b, _ in chunk]).to(device)
            answers = torch.stack([tokens[b + 1:b + length + 1] for b, _ in chunk]).to(device)
            with torch.autocast(torch.device(device).type, enabled=False):  # 32-bit even if the caller is in 16-bit mode
                scores, _ = model(windows)
            nats = F.cross_entropy(scores.float().view(-1, scores.shape[-1]), answers.view(-1), reduction="none").view(len(chunk), length).cpu().numpy()
            for row, (begin, first) in enumerate(chunk):
                surprise[begin + first:begin + length] = nats[row, first:]
                times_scored[begin + first:begin + length] += 1
    finally:
        model.train(was_training)  # leave the model as we found it, even if something went wrong
    if (times_scored != 1).any():
        raise AssertionError("the windows did not score every token exactly once")
    if not np.isfinite(surprise).all():
        raise FloatingPointError("the model's predictions are no longer numbers: the run has diverged")
    return surprise


def characters_in(stream: np.ndarray, tokenizer) -> int:
    """How many characters of text the scored tokens (all but the first) stand for. START stands for none."""
    lengths = np.array([0 if piece == tokenizer.vocab[tokenizer.start_id] else len(piece) for piece in tokenizer.vocab])
    return int(lengths[np.asarray(stream[1:]).astype(np.int64)].sum())


def bits_per_character(surprise: np.ndarray, characters: int) -> float:
    """The one number that is comparable between tokenizers: total surprise, in bits, per character of text."""
    return float(surprise.sum() / math.log(2) / characters)


def seen_sample(training: np.ndarray, tokens: int, context: int, pieces: int = 16) -> list[tuple[np.ndarray, int]]:
    """`tokens` tokens of TRAINING text, taken as `pieces` slices spread evenly through the stream.

    Each slice comes with up to `context` tokens of run-up in front of it. The run-up is read but not scored, so
    the slice's first tokens are predicted with text behind them, as they would be in the middle of a long
    stream. Without it we would be scoring 16 cold openings against the validation stream's one.
    Returns (run-up + slice, length of the run-up) pairs.
    """
    piece, sample = tokens // pieces, []
    if piece < 2 or len(training) < tokens:
        raise ValueError(f"a training stream of {len(training):,} tokens is too short to cut {tokens:,} tokens from")
    for begin in np.linspace(0, len(training) - piece - 1, pieces).astype(int):
        run_up = min(context, int(begin))
        sample.append((np.asarray(training[begin - run_up:begin + piece]), run_up))
    return sample


def score_seen(model: GPT, sample: list[tuple[np.ndarray, int]], tokenizer, device: str) -> float:
    surprise = np.concatenate([evaluate(model, part, device)[run_up:] for part, run_up in sample])
    return bits_per_character(surprise, sum(characters_in(part[run_up:], tokenizer) for part, run_up in sample))


# ---------------------------------------------------------------------------------------------- sampling
@torch.no_grad()
def generate(model: GPT, tokenizer, prompt: str, tokens: int, temperature: float = 0.8, seed: int = 0) -> str:
    """Continue `prompt` one token at a time. A minimal sampler, for watching a run; the careful one comes later.

    Temperature divides the scores before they become probabilities: below 1 the model plays safe, above 1 it
    gambles. The dice are rolled on the CPU from a fixed seed, so the same weights give the same text, and the
    samples taken at different moments of a run differ only because the model does.

    Every new token re-reads the whole window. For 200 characters that is about 100 times the arithmetic that a
    cache of earlier results would need. It takes under two seconds here, so the simple way stays.
    """
    if temperature <= 0 or not prompt:
        raise ValueError("generate needs a prompt of at least one character and a temperature above zero")
    ids = tokenizer.encode(prompt)  # before touching the model: a prompt with an unknown character stops here
    device = next(model.parameters()).device
    dice = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        for _ in range(tokens):
            window = torch.tensor([ids[-model.config.context:]], device=device)
            scores = model(window)[0][0, -1].float().cpu() / temperature
            scores[tokenizer.start_id] = float("-inf")  # START means "a new work begins"; a sample just keeps going
            ids.append(int(torch.multinomial(torch.softmax(scores, dim=-1), 1, generator=dice)))
    finally:
        model.train(was_training)
    return tokenizer.decode(ids)


# ---------------------------------------------------------------------------------------------- checkpoints
def git_commit() -> str:
    """The commit the code is at, and whether anything that can change a run differs from that commit."""
    matters = ["src", "scripts/train.py", "pyproject.toml", "uv.lock", "data/tokenizers", "data/processed"]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *matters], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return commit + (" (with uncommitted changes to the code or data)" if dirty else "")
    except Exception:
        return "unknown"


def fingerprints(tokenizer_name: str, training: np.ndarray, validation: np.ndarray) -> dict:
    """Checksums of the tokenizer file and of the very tokens this run reads. A name alone does not say which file it was."""
    return {"tokenizer": hashlib.sha256((DATA / "tokenizers" / f"{tokenizer_name}.json").read_bytes()).hexdigest(),
            "training_stream": hashlib.sha256(np.ascontiguousarray(training)).hexdigest(),
            "validation_stream": hashlib.sha256(np.ascontiguousarray(validation)).hexdigest()}


def save_checkpoint(paths: list[Path], model: GPT, optimizer, run: RunConfig, step: int, batches: np.random.Generator, facts: dict, stamp: dict) -> None:
    """Everything needed to carry on: the weights, AdamW's running averages, the step, the best score so far, the
    minutes used, and the state of all three dice (PyTorch's on the CPU, the GPU's for dropout, the batch cutter's).

    On the CPU a resumed run is, bit for bit, the run that was never stopped (tested). On the Mac's GPU it is the
    same run only as far as two uninterrupted runs from one seed are: same start, same batches, same dropout
    masks, last digits free (docs/BUILD_LOG.md, entry 13).

    Written under another name and then renamed, so a run that is killed in the middle of a save cannot leave half
    a file where the last good checkpoint used to be. When the same moment goes to two files (best.pt and last.pt),
    both are written first and then renamed one straight after the other, so the two cannot disagree for long.
    """
    device = next(model.parameters()).device.type
    checkpoint = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "model_config": dataclasses.asdict(model.config), "run_config": dataclasses.asdict(run),
        "step": step, "facts": facts, **stamp,
        "random": {"torch": torch.get_rng_state(), "gpu": torch.mps.get_rng_state() if device == "mps" else None,
                   "batches": batches.bit_generator.state},
    }
    for path in paths:
        torch.save(checkpoint, path.with_name(path.name + ".tmp"))
    for path in paths:
        os.replace(path.with_name(path.name + ".tmp"), path)


def load_checkpoint(path: Path, device: str):
    """Returns (model, saved). `saved` holds the optimizer state, the step and the random states for resuming.

    weights_only=True: the file may hold numbers, text and lists, and nothing that runs. A .pt file is otherwise a
    program, and loading one from a stranger hands them your computer. (The released model will not be a .pt at all.)
    """
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):  # the one harmless extra our first checkpoints contain
        saved = torch.load(path, map_location="cpu", weights_only=True)  # to the CPU first: AdamW wants its step counters there
    model = GPT(Config(**saved["model_config"])).to(device)
    model.load_state_dict(saved["model"])
    return model, saved


# ---------------------------------------------------------------------------------------------- the run
def train(run: RunConfig, resume: bool = False, overwrite: bool = False, stop_at: int | None = None, say=print) -> dict | None:
    """Train one model from start to finish. Returns the result, which is also written to disk.

    Everything lands in runs/<name>/:
        config.json    every setting, the code's commit, checksums of the tokenizer and the token streams
        log.csv        one row per evaluation
        samples.txt    a short sample at every evaluation, always from the same prompt and the same dice
        last.pt        the latest checkpoint, to resume from
        best.pt        the checkpoint with the best validation score: the run's answer
        result.json    written at the very end; a folder without it holds an unfinished run

    `resume` carries on from last.pt, with the settings the run began with; anything else is an error.
    `overwrite` allows a new run to replace an old one of the same name. `stop_at` stops after that step as if
    the process had been killed (for tests).
    """
    folder = ROOT / "runs" / run.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run.name):
        raise ValueError(f"a run's name must be a plain folder name of letters, digits, dots, dashes and underscores, not {run.name!r}")
    if resume and overwrite:
        raise ValueError("resume and overwrite contradict each other")
    if min(run.batch, run.context, run.evaluations) < 1 or run.warm_up < 0:
        raise ValueError("batch, context and evaluations must be at least 1, and the warm-up cannot be negative")
    tokenizer = load_tokenizer(DATA / "tokenizers" / f"{run.tokenizer}.json")
    training, validation = load_tokens(run.tokenizer, "train"), load_tokens(run.tokenizer, "validation")
    if max(int(training.max()), int(validation.max())) >= tokenizer.vocab_size:
        raise ValueError("the token streams hold ids outside the vocabulary")
    last_step = run.steps(len(training))
    if run.warm_up >= last_step:
        raise ValueError(f"this run has {last_step} steps, so a warm-up of {run.warm_up} would be the whole of it")
    if resume and not (folder / "last.pt").exists():
        raise FileNotFoundError(f"nothing to resume: {folder / 'last.pt'} does not exist")
    if not resume and not overwrite and ((folder / "last.pt").exists() or (folder / "result.json").exists()):
        raise FileExistsError(f"runs/{run.name} already holds a run. Resume it, pick another name, or use --overwrite to replace it.")
    evaluate_at = set(run.evaluation_steps(last_step))
    validation_characters = characters_in(validation, tokenizer)
    # As many tokens of TRAINING text as validation holds, from 16 places, scored the same way. The gap between the two
    # says how much better the model does on works it has trained on than on works it has not. That is not all
    # learning by heart: a 5-gram, which cannot recite anything longer than five characters, shows a gap of 0.35.
    # And for the word-fragment tokenizers part of the gap is the tokenizer itself, which was fitted on the
    # training works. So: watch this number WITHIN a run (it falling while validation rises is over-fitting);
    # never compare it between tokenizers.
    seen = seen_sample(training, len(validation), run.context)
    stamp = {"git_commit": git_commit(), "torch": str(torch.__version__), "fingerprints": fingerprints(run.tokenizer, training, validation)}  # taken once: the code and data THIS process runs on

    torch.manual_seed(run.seed)
    batches = np.random.default_rng(run.seed)
    model = GPT(Config(vocab_size=tokenizer.vocab_size, context=run.context, layers=run.layers, heads=run.heads,
                       width=run.width, dropout=run.dropout)).to(run.device)
    optimizer = torch.optim.AdamW(model.parameter_groups(run.weight_decay), lr=run.peak_rate, betas=(0.9, run.beta_2))
    step, best, facts, minutes_before, skip_evaluation_at = 0, {"validation_bpc": float("inf")}, None, 0.0, None
    folder.mkdir(parents=True, exist_ok=True)  # nothing is written before this line: settings that make no sense have been refused by now
    for leftover in folder.glob("*.tmp"):
        leftover.unlink()  # half a checkpoint, from a run that was killed while saving
    if resume:
        model, saved = load_checkpoint(folder / "last.pt", run.device)
        if "fingerprints" not in saved or "row" not in saved["facts"]:
            raise ValueError(f"runs/{run.name} was written by an older train.py. It can still be scored (scripts/evaluate.py), but not resumed.")
        changed = {k: (saved["run_config"].get(k), v) for k, v in dataclasses.asdict(run).items() if k != "device" and saved["run_config"].get(k) != v}
        if changed:
            raise ValueError(f"runs/{run.name} was started with other settings (then, now): {changed}. A resumed run keeps its settings.")
        if saved["fingerprints"] != stamp["fingerprints"]:
            raise ValueError("the tokenizer or the token streams have changed since this run began")
        optimizer = torch.optim.AdamW(model.parameter_groups(run.weight_decay), lr=run.peak_rate, betas=(0.9, run.beta_2))
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["random"]["torch"])
        if saved["random"]["gpu"] is not None:
            torch.mps.set_rng_state(saved["random"]["gpu"])
        batches.bit_generator.state = saved["random"]["batches"]
        step, facts = saved["step"], saved["facts"]
        best, minutes_before = facts["best"], facts["minutes"]
        # At every stop the checkpoint is written first, then the log row, then the sample. The checkpoint carries the row
        # and the sample's heading, so if the run died between those writes we can finish the job here.
        with open(folder / "log.csv", newline="") as f:
            logged = {row[0] for row in csv.reader(f) if row}
        if str(step) not in logged:
            with open(folder / "log.csv", "a", newline="") as f:
                csv.writer(f).writerow(facts["row"])
        if facts["heading"] not in (folder / "samples.txt").read_text(encoding="utf-8"):
            with open(folder / "samples.txt", "a", encoding="utf-8") as f:
                f.write(f"{facts['heading']}\n{generate(model, tokenizer, PROMPT, 200 if run.tokenizer == 'char' else 80)}\n\n")
        skip_evaluation_at = step
    else:
        for stale in [folder / "last.pt", folder / "best.pt", folder / "result.json", *folder.glob("evaluation-*.json")]:
            stale.unlink(missing_ok=True)  # only reached with `overwrite`: nothing of the old run may outlive it
        (folder / "config.json").write_text(json.dumps({**dataclasses.asdict(run), "steps": last_step, "parameters": model.config.parameter_count(),
                                                        **stamp}, indent=2) + "\n")
        with open(folder / "log.csv", "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLUMNS)
        (folder / "samples.txt").write_text("")

    say(f"{run.name}: {model.config.parameter_count():,} parameters, {run.tokenizer} tokenizer, {last_step:,} steps = {run.passes:g} passes, seed {run.seed}")
    model.train()
    losses, lengths, started = [], [], time.perf_counter()
    leg_started, leg_first_step = started, step
    while True:
        if step in evaluate_at and step != skip_evaluation_at:
            training_loss = float(torch.stack(losses).mean()) if losses else float("nan")  # reading this waits for the GPU to catch up
            gradient_length = float(torch.stack(lengths).mean()) if lengths else float("nan")
            share_clipped = float((torch.stack(lengths) > run.clip).float().mean()) if lengths else float("nan")
            tokens_per_second = (step - leg_first_step) * run.batch * run.context / (time.perf_counter() - leg_started) if step > leg_first_step else float("nan")
            losses, lengths = [], []
            surprise = evaluate(model, validation, run.device)
            validation_bpc, seen_bpc = bits_per_character(surprise, validation_characters), score_seen(model, seen, tokenizer, run.device)
            passes = step * run.batch * run.context / len(training)
            minutes = minutes_before + (time.perf_counter() - started) / 60
            is_best = validation_bpc < best["validation_bpc"]
            if is_best:
                best = {"validation_bpc": validation_bpc, "step": step, "passes": passes, "seen_bpc": seen_bpc}
            rate_used = optimizer.param_groups[0]["lr"] if step else float("nan")  # the rate of the step just taken, read from the optimizer itself
            row = [step, round(passes, 2), rate_used, round(training_loss, 4), round(gradient_length, 4), round(share_clipped, 4), round(seen_bpc, 4),
                   round(validation_bpc, 4), round(float(surprise.mean()), 4), tokens_per_second if math.isnan(tokens_per_second) else round(tokens_per_second), round(minutes, 2)]
            heading = f"===== step {step}, {passes:.1f} passes, validation {validation_bpc:.3f} bits per character ====="
            facts = {"best": best, "validation_bpc": validation_bpc, "seen_bpc": seen_bpc, "minutes": minutes, "row": row, "heading": heading}
            save_checkpoint([folder / "best.pt", folder / "last.pt"] if is_best else [folder / "last.pt"], model, optimizer, run, step, batches, facts, stamp)
            with open(folder / "log.csv", "a", newline="") as f:
                csv.writer(f).writerow(row)
            sample = generate(model, tokenizer, PROMPT, 200 if run.tokenizer == "char" else 80)
            with open(folder / "samples.txt", "a", encoding="utf-8") as f:
                f.write(f"{heading}\n{sample}\n\n")
            say(f"  step {step:>6,}  {passes:5.1f} passes  training loss {training_loss:6.3f}  seen {seen_bpc:.3f}  validation {validation_bpc:.3f} bpc"
                f"{'  *' if is_best else ''}  {minutes:5.1f} min")
            leg_started, leg_first_step = time.perf_counter(), step
        if step >= last_step:
            break
        if step == stop_at:
            return None

        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, run.warm_up, last_step, run.peak_rate, run.floor_rate)
        tokens, targets = random_batch(training, run.batch, run.context, batches, run.device)   # take a batch
        with torch.autocast(run.device, dtype=torch.bfloat16, enabled=run.sixteen_bit):
            loss = model(tokens, targets)[1]                                                    # run the model, measure the loss
        optimizer.zero_grad(set_to_none=True)                                                   # forget the last step's gradients (PyTorch adds them up otherwise)
        loss.backward()                                                                         # work out the new ones
        length = torch.nn.utils.clip_grad_norm_(model.parameters(), run.clip)                   # clip them; this hands back their length before clipping
        optimizer.step()                                                                        # nudge the parameters
        losses.append(loss.detach())   # kept on the GPU. Reading a number from the GPU stops Python until the GPU has caught up,
        lengths.append(length.detach())  # and then the GPU idles while Python prepares the next step. So we read them once per evaluation.
        step += 1

    if torch.load(folder / "best.pt", map_location="cpu", weights_only=True)["step"] != best["step"]:
        raise AssertionError(f"runs/{run.name}/best.pt does not hold step {best['step']}, the run's best moment")
    result = {"name": run.name, "best": best, "final_validation_bpc": facts["validation_bpc"], "final_seen_bpc": facts["seen_bpc"],
              "steps": last_step, "minutes": round(facts["minutes"], 2)}
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    say(f"{run.name}: best validation {best['validation_bpc']:.4f} bits per character at step {best['step']:,} ({best['passes']:.1f} passes), {result['minutes']} minutes")
    return result
