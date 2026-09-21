"""Tests for the training library.  Run:  uv run pytest tests/test_train.py

A training loop has few lines and many ways to mislead: a validation number computed the wrong way, a best
checkpoint that is not the best, a resumed run that is quietly a different run. These check the parts that
produce the numbers we will publish.
"""

import csv
import dataclasses
import hashlib
import json
import math
import pickle
import re
import runpy
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import gp_thee.train as T
from gp_thee.data import DATA, load_tokens
from gp_thee.model import GPT, Config
from gp_thee.tokenizer import CharTokenizer, load
from gp_thee.train import RunConfig, bits_per_character, characters_in, evaluate, generate, learning_rate, plan_windows, seen_sample

QUIET = lambda *_: None
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
LETTERS = CharTokenizer(sorted("abcdefghijklmnopqrstuvwxyz .,"))   # 29 characters + START = the tiny model's vocabulary of 30

TINY = Config(vocab_size=30, context=16, layers=2, heads=2, width=32, dropout=0.0)


def tiny_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(TINY)


# ------------------------------------------------------------------------------------------ the schedule
def test_the_learning_rate_climbs_then_follows_a_cosine_down_to_the_floor():
    rate = lambda step: learning_rate(step, warm_up=100, last_step=1000, peak=1e-3, floor=1e-4)
    assert rate(0) == pytest.approx(1e-5) and rate(49) == pytest.approx(5e-4) and rate(99) == pytest.approx(1e-3)
    assert rate(100) == pytest.approx(1e-3)
    assert rate(550) == pytest.approx((1e-3 + 1e-4) / 2)            # halfway down the cosine
    assert rate(1000) == rate(5000) == pytest.approx(1e-4)
    assert all(rate(s) >= rate(s + 1) for s in range(100, 1000))    # never rises after the warm-up


def test_a_budget_in_passes_becomes_the_right_number_of_steps():
    run = RunConfig(name="x", passes=17.0)
    assert run.steps(4_811_375) == round(17 * 4_811_375 / (64 * 256)) == 4992
    assert dataclasses.replace(run, passes=0.001).steps(1000) == 1


def test_every_run_stops_to_be_scored_the_same_number_of_times():
    run = RunConfig(name="x", evaluations=40)
    for last_step in (1_564, 4_992, 19_970):                        # a short word-fragment run, and two character runs
        stops = run.evaluation_steps(last_step)
        assert len(stops) == 41 and stops[0] == 0 and stops[-1] == last_step
        gaps = np.diff(stops)
        assert gaps.max() - gaps.min() <= 1                         # evenly spread
    assert dataclasses.replace(run, evaluations=3).evaluation_steps(78) == [0, 26, 52, 78]
    assert dataclasses.replace(run, evaluations=40).evaluation_steps(5) == [0, 1, 2, 3, 4, 5]   # more stops than steps: every step, once


# ------------------------------------------------------------------------------------------ evaluation
@pytest.mark.parametrize("targets", [1, 5, 16, 17, 24, 25, 31, 32, 33, 100, 257, 1000])
@pytest.mark.parametrize("stride", [8, 4, 16])
def test_every_token_is_scored_exactly_once_with_enough_text_behind_it(targets, stride):
    context = 16
    times_scored, text_behind = np.zeros(targets, dtype=int), np.zeros(targets, dtype=int)
    plan = plan_windows(targets, context, stride)
    for begin, first in plan:
        assert 0 <= begin and begin + min(context, targets) <= targets       # the window lies inside the stream
        for position in range(first, min(context, targets)):
            times_scored[begin + position] += 1
            text_behind[begin + position] = position + 1                     # tokens the model could see, itself included
    assert (times_scored == 1).all()
    in_first_window = np.arange(targets) < context
    assert (text_behind[in_first_window] == np.arange(1, targets + 1)[in_first_window]).all()  # the opening has what there is
    assert (text_behind[~in_first_window] > context - stride).all()                               # everything later has plenty


def test_evaluation_matches_the_slow_obvious_way():
    # The slow way: for every token, hand the model the (up to) 16 tokens before it and read off one number.
    # With a stride of 1 the fast way must give exactly the same context to every token, so the same surprise.
    model, stream = tiny_model().eval(), np.random.default_rng(0).integers(0, 30, size=60)
    fast = evaluate(model, stream, "cpu", stride=1, batch=7)
    slow = []
    with torch.no_grad():
        for t in range(1, len(stream)):
            window = torch.tensor(stream[max(0, t - 16):t])[None]
            slow.append(F.cross_entropy(model(window)[0][0, -1][None], torch.tensor([stream[t]])).item())
    assert np.allclose(fast, slow, atol=1e-5)


def test_evaluation_does_not_depend_on_how_it_is_batched_and_leaves_the_model_in_training_mode():
    model, stream = tiny_model().train(), np.random.default_rng(1).integers(0, 30, size=200)
    a, b = evaluate(model, stream, "cpu", batch=64), evaluate(model, stream, "cpu", batch=3)
    assert np.allclose(a, b, atol=1e-6) and len(a) == 199 and model.training


def test_the_default_stride_is_half_the_context():
    # Every published number uses the default. Nothing else in this file would notice if it became "no overlap at all".
    model, stream = tiny_model(), np.random.default_rng(3).integers(0, 30, size=200)
    assert np.array_equal(evaluate(model, stream, "cpu"), evaluate(model, stream, "cpu", stride=8))
    assert not np.allclose(evaluate(model, stream, "cpu"), evaluate(model, stream, "cpu", stride=16), atol=1e-3)  # and the check can tell strides apart


@pytest.mark.parametrize("stride", [0, -1, 17, 20])
def test_evaluation_refuses_a_stride_that_makes_no_sense(stride):
    # `match`, because with a stride of 20 numpy raises a ValueError of its own further down ("could not broadcast"),
    # and any ValueError would do: the check could be deleted and that case would still pass.
    with pytest.raises(ValueError, match="makes no sense"):
        evaluate(tiny_model(), np.arange(120) % 30, "cpu", stride=stride)


def test_evaluation_refuses_a_plan_that_scores_a_token_twice_or_not_at_all(monkeypatch):
    # The guard behind plan_windows. Without it a token scored twice would quietly keep the later window's number, and one
    # never scored would count as no surprise at all: a wrong sum either way, and no error.
    model, stream = tiny_model(), np.arange(41) % 30                               # 40 targets, context 16
    monkeypatch.setattr(T, "plan_windows", lambda targets, context, stride: [(0, 0), (8, 7), (16, 8), (24, 8)])   # target 15 twice
    with pytest.raises(AssertionError, match="exactly once"):
        evaluate(model, stream, "cpu")
    monkeypatch.setattr(T, "plan_windows", lambda targets, context, stride: [(0, 0), (8, 9), (16, 8), (24, 8)])   # target 16 never
    with pytest.raises(AssertionError, match="exactly once"):
        evaluate(model, stream, "cpu")


def test_evaluation_leaves_the_model_in_the_mode_it_found_it_even_after_an_error():
    model = tiny_model().eval()
    evaluate(model, np.arange(40) % 30, "cpu")
    assert not model.training
    with pytest.raises(IndexError):
        evaluate(model.train(), np.array([0, 1, 99]), "cpu")        # 99 is not in a vocabulary of 30
    assert model.training


def test_evaluation_is_32_bit_even_inside_a_16_bit_block():
    model, stream = tiny_model(), np.random.default_rng(4).integers(0, 30, size=300)
    plain = evaluate(model, stream, "cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert np.array_equal(evaluate(model, stream, "cpu"), plain)


def test_a_model_that_has_blown_up_is_reported_as_such():
    model = tiny_model()
    with torch.no_grad():
        model.final_norm.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        evaluate(model, np.arange(40) % 30, "cpu")


def test_evaluation_switches_dropout_off():
    torch.manual_seed(0)
    model = GPT(dataclasses.replace(TINY, dropout=0.5)).train()
    stream = np.random.default_rng(2).integers(0, 30, size=100)
    assert np.array_equal(evaluate(model, stream, "cpu"), evaluate(model, stream, "cpu"))


def test_bits_per_character_counts_characters_not_tokens():
    tokenizer = CharTokenizer(sorted("abc"))
    stream = np.array(tokenizer.encode_works(["abca", "bc"]))       # START a b c a START b c
    assert characters_in(stream, tokenizer) == 6                     # the scored tokens are a b c a START b c; START is no characters
    surprise = np.full(len(stream) - 1, math.log(2))                 # one bit of surprise at each of the 7 scored tokens
    assert bits_per_character(surprise, 6) == pytest.approx(7 / 6)   # START's surprise is counted; it adds no characters


def test_the_first_token_is_not_scored_so_its_characters_are_not_counted():
    tokenizer = CharTokenizer(sorted("abc"))
    assert characters_in(np.array(tokenizer.encode("abca")), tokenizer) == 3      # a|bca: only b, c and a are predicted


def test_the_seen_sample_is_slices_of_the_stream_each_with_a_run_up():
    stream = np.arange(10_000)
    sample = seen_sample(stream, tokens=1_600, context=32, pieces=16)
    assert len(sample) == 16 and sample[0][1] == 0 and all(run_up == 32 for _, run_up in sample[1:])
    begins = [int(part[run_up]) for part, run_up in sample]                       # in this stream a token IS its position
    assert begins[0] == 0 and begins[-1] == 10_000 - 100 - 1 and np.allclose(np.diff(begins), 9_899 / 15, atol=1)
    assert all(len(part) == run_up + 100 and np.array_equal(part, stream[b - run_up:b + 100]) for (part, run_up), b in zip(sample, begins))
    for too_short in (np.arange(400), np.arange(1_599)):                           # slices would overlap, or run off the front
        with pytest.raises(ValueError):
            seen_sample(too_short, tokens=1_600, context=32)


# ------------------------------------------------------------------------------------------ sampling
def test_the_same_weights_and_seed_give_the_same_text_and_start_is_never_written():
    model = tiny_model()
    once, twice = generate(model, LETTERS, "to be", 40, seed=3), generate(model, LETTERS, "to be", 40, seed=3)
    assert once == twice and once.startswith("to be") and len(once) == 45  # 40 new characters, none of them START
    assert generate(model, LETTERS, "to be", 40, seed=4) != once


def test_a_tiny_temperature_always_picks_the_likeliest_token():
    model, ids = tiny_model(), LETTERS.encode("to be")
    with torch.no_grad():
        for _ in range(30):
            scores = model(torch.tensor([ids[-16:]]))[0][0, -1].clone()
            scores[LETTERS.start_id] = float("-inf")
            ids.append(int(scores.argmax()))
    assert generate(model, LETTERS, "to be", 30, temperature=1e-4, seed=0) == LETTERS.decode(ids)
    assert generate(model, LETTERS, "to be", 30, temperature=1e-4, seed=1) == LETTERS.decode(ids)   # no dice left to roll
    assert generate(model, LETTERS, "to be", 30, temperature=1.0, seed=0) != LETTERS.decode(ids)


def test_sampling_reads_the_most_recent_tokens_once_the_text_is_longer_than_the_context():
    model, tail = tiny_model(), " to be or not to be"                              # 19 characters: more than the context of 16
    one = generate(model, LETTERS, "a" * 16 + tail, 10, temperature=1e-4)
    two = generate(model, LETTERS, "b" * 16 + tail, 10, temperature=1e-4)
    assert one[16:] == two[16:]                                                     # what has scrolled out of the window cannot matter


def test_sampling_leaves_the_model_in_the_mode_it_found_it_and_refuses_nonsense():
    model = tiny_model().train()
    generate(model, LETTERS, "to be", 3)
    assert model.training
    generate(model.eval(), LETTERS, "to be", 3)
    assert not model.training
    for prompt, temperature in (("", 0.8), ("to be", 0.0), ("to be", -1.0)):
        with pytest.raises(ValueError):
            generate(model, LETTERS, prompt, 3, temperature=temperature)
    with pytest.raises(ValueError):
        generate(model.train(), LETTERS, "don't", 3)                                # a character the tokenizer has never seen
    assert model.training


def test_sampling_restores_the_mode_even_if_the_model_fails_midway(monkeypatch):
    # Every error in the test above is raised before the model is touched. This one strikes after eval(), in the middle of the loop.
    # Whoever catches such an error (in a notebook, say) and trains on would otherwise do so with dropout switched off, unannounced.
    model, calls = tiny_model().train(), []
    real = model.forward
    def fails_on_the_third_token(*a, **k):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("out of memory, say")
        return real(*a, **k)
    monkeypatch.setattr(model, "forward", fails_on_the_third_token)
    with pytest.raises(RuntimeError):
        generate(model, LETTERS, "to be", 5)
    assert model.training


# ------------------------------------------------------------------------------------------ a whole run, tiny, on the CPU
@pytest.fixture()
def tiny_run(tmp_path, monkeypatch):
    """A whole run in about a second: 78 steps of a two-layer model on the first 40,000 training tokens,
    scored at steps 0, 26, 52 and 78 on the first 8,000 validation tokens. Runs land in a temporary folder."""
    monkeypatch.setattr(T, "ROOT", tmp_path)
    real = T.load_tokens
    monkeypatch.setattr(T, "load_tokens", lambda name, which: real(name, which)[:40_000 if which == "train" else 8_000])
    return RunConfig(name="t", passes=0.5, batch=8, context=32, layers=2, heads=2, width=32, dropout=0.1, evaluations=3, warm_up=5, device="cpu")


def log_of(folder) -> list[dict]:
    with open(folder / "log.csv", newline="") as f:
        return list(csv.DictReader(f))


def weights_of(path) -> dict:
    return torch.load(path, weights_only=True)["model"]


def test_a_run_learns_and_the_saved_weights_are_the_ones_that_scored(tiny_run, tmp_path):
    result = T.train(tiny_run, say=QUIET)
    rows = log_of(tmp_path / "runs/t")
    assert [int(row["step"]) for row in rows] == [0, 26, 52, 78] == tiny_run.evaluation_steps(result["steps"])
    assert float(rows[-1]["validation_bpc"]) < float(rows[0]["validation_bpc"]) - 1.0     # validation improved by over a bit per character
    assert result["best"]["validation_bpc"] == pytest.approx(min(float(row["validation_bpc"]) for row in rows), abs=1e-4)  # the log is rounded
    model, saved = T.load_checkpoint(tmp_path / "runs/t/best.pt", "cpu")
    assert saved["step"] == result["best"]["step"] and saved["run_config"] == dataclasses.asdict(tiny_run)
    validation, tokenizer = T.load_tokens("char", "validation"), load(DATA / "tokenizers/char.json")
    again = bits_per_character(evaluate(model, validation, "cpu"), characters_in(validation, tokenizer))
    assert again == pytest.approx(result["best"]["validation_bpc"], abs=1e-6)             # the saved weights ARE the ones that scored it
    assert json.loads((tmp_path / "runs/t/result.json").read_text()) == result


def test_a_checkpoint_says_what_made_it_and_holds_nothing_that_can_run(tiny_run, tmp_path):
    T.train(tiny_run, stop_at=0, say=QUIET)
    saved = torch.load(tmp_path / "runs/t/last.pt", weights_only=True)                    # weights_only: numbers, text and lists, no code
    training, validation = T.load_tokens("char", "train"), T.load_tokens("char", "validation")   # the 40,000 and 8,000 tokens this run really read
    assert saved["fingerprints"] == {"tokenizer": hashlib.sha256((DATA / "tokenizers/char.json").read_bytes()).hexdigest(),
                                     "training_stream": hashlib.sha256(np.ascontiguousarray(training)).hexdigest(),
                                     "validation_stream": hashlib.sha256(np.ascontiguousarray(validation)).hexdigest()}
    assert type(saved["torch"]) is str
    config = json.loads((tmp_path / "runs/t/config.json").read_text())
    assert config["fingerprints"] == saved["fingerprints"] and config["git_commit"] == saved["git_commit"] and config["steps"] == 78
    assert not list((tmp_path / "runs/t").glob("*.tmp"))                                  # no half-made file is left lying about (that a save goes through one
                                                                                          # at all is shown among the killed runs, further down)


def test_config_json_counts_the_parameters(tiny_run, tmp_path):
    # Counted here from the tensors themselves, the tied table once. scripts/summarise_runs.py publishes this number, and breaks
    # ties between settings with it ("the smallest model within reach of the best").
    T.train(tiny_run, stop_at=0, say=QUIET)
    model, _ = T.load_checkpoint(tmp_path / "runs/t/last.pt", "cpu")
    assert json.loads((tmp_path / "runs/t/config.json").read_text())["parameters"] == sum(p.numel() for p in model.parameters()) > 10_000


def test_best_means_lowest_validation_even_when_later_is_worse_seen_is_better_or_the_run_was_resumed(tiny_run, tmp_path, monkeypatch):
    # A scripted over-fitting run: validation is best at step 26 and then gets worse, while the seen score keeps improving.
    script = {"validation": iter([5.0, 3.0, 4.0, 3.5]), "seen": iter([5.0, 2.5, 2.0, 1.5])}
    validation_characters = characters_in(T.load_tokens("char", "validation"), load(DATA / "tokenizers/char.json"))
    monkeypatch.setattr(T, "bits_per_character", lambda surprise, characters: next(script["validation" if characters == validation_characters else "seen"]))
    assert T.train(tiny_run, stop_at=52, say=QUIET) is None                                # scored at 0, 26, 52; then the power fails
    result = T.train(tiny_run, resume=True, say=QUIET)                                     # scored at 78 only
    assert result["best"] == {"validation_bpc": 3.0, "step": 26, "passes": pytest.approx(26 * 256 / 40_000), "seen_bpc": 2.5}
    assert result["final_validation_bpc"] == 3.5 and result["final_seen_bpc"] == 1.5
    best, last = (torch.load(tmp_path / f"runs/t/{name}.pt", weights_only=True) for name in ("best", "last"))
    assert (best["step"], last["step"]) == (26, 78) and best["facts"]["validation_bpc"] == 3.0
    assert not all(torch.equal(best["model"][k], last["model"][k]) for k in best["model"])
    assert [(row["step"], row["validation_bpc"], row["seen_bpc"]) for row in log_of(tmp_path / "runs/t")] == \
        [("0", "5.0", "5.0"), ("26", "3.0", "2.5"), ("52", "4.0", "2.0"), ("78", "3.5", "1.5")]


def test_a_tie_for_best_keeps_the_earlier_checkpoint(tiny_run, tmp_path, monkeypatch):
    # Of two equally good moments the earlier one has read the training text fewer times. "Better" means strictly better.
    script = iter([5.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.5, 3.5])                                # validation and seen alike, at steps 0, 26, 52, 78
    monkeypatch.setattr(T, "bits_per_character", lambda surprise, characters: next(script))
    assert T.train(tiny_run, say=QUIET)["best"]["step"] == 26
    assert torch.load(tmp_path / "runs/t/best.pt", weights_only=True)["step"] == 26


def test_the_loop_matches_the_same_loop_written_out_by_hand(tiny_run, tmp_path):
    # Every setting but one differs from its default (16-bit stays off: it has a test of its own), and the tokens are word
    # fragments, so tokens are not characters.
    # If any setting failed to reach the optimizer, or the loop did its six lines in another order, the weights would differ.
    run = dataclasses.replace(tiny_run, name="byhand", tokenizer="bpe-1024", passes=0.1, seed=3, peak_rate=2e-3, floor_rate=3e-4, warm_up=4,
                              weight_decay=0.05, beta_2=0.95, clip=0.25, evaluations=2)
    result = T.train(run, say=QUIET)
    training, validation, tokenizer = T.load_tokens("bpe-1024", "train"), T.load_tokens("bpe-1024", "validation"), load(DATA / "tokenizers/bpe-1024.json")
    steps = run.steps(len(training))
    assert steps == 16

    torch.manual_seed(3)
    dice = np.random.default_rng(3)
    model = GPT(Config(vocab_size=tokenizer.vocab_size, context=32, layers=2, heads=2, width=32, dropout=0.1)).train()
    matrices, scales = [p for p in model.parameters() if p.dim() >= 2], [p for p in model.parameters() if p.dim() < 2]
    assert len(scales) == 5 and len(matrices) == 10
    optimizer = torch.optim.AdamW([{"params": matrices, "weight_decay": 0.05}, {"params": scales, "weight_decay": 0.0}], lr=123.0, betas=(0.9, 0.95))
    losses, lengths = [], []
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, 4, steps, 2e-3, 3e-4)
        begins = dice.integers(0, len(training) - 32, size=8)
        tokens = torch.from_numpy(np.stack([training[b:b + 32] for b in begins]).astype(np.int64))
        targets = torch.from_numpy(np.stack([training[b + 1:b + 33] for b in begins]).astype(np.int64))
        loss = model(tokens, targets)[1]
        optimizer.zero_grad()
        loss.backward()
        lengths.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)))
        optimizer.step()
        losses.append(loss.item())
    assert min(lengths) > 0.25                                                            # the clip really was doing something

    saved = torch.load(tmp_path / "runs/byhand/last.pt", weights_only=True)
    assert saved["step"] == steps and all(torch.equal(saved["model"][k], v) for k, v in model.state_dict().items())
    rows = log_of(tmp_path / "runs/byhand")
    last = rows[-1]
    assert [int(row["step"]) for row in rows] == [0, 8, 16]
    assert float(last["passes"]) == round(16 * 8 * 32 / len(training), 2)
    assert float(last["training_loss"]) == pytest.approx(np.mean(losses[8:]), abs=1e-4)
    assert float(last["gradient_length"]) == pytest.approx(np.mean(lengths[8:]), abs=1e-4) and float(last["share_clipped"]) == 1.0
    assert float(last["learning_rate"]) == pytest.approx(learning_rate(15, 4, 16, 2e-3, 3e-4))   # the rate of the last step taken
    assert float(last["validation_loss"]) == pytest.approx(evaluate(model, validation, "cpu").mean(), abs=1e-4)   # nats per TOKEN
    validation_bits = evaluate(model, validation, "cpu").sum() / math.log(2)
    assert result["final_validation_bpc"] == pytest.approx(validation_bits / len(tokenizer.decode(validation[1:].tolist())), abs=1e-9)
    # the seen sample, by hand: 16 slices of TRAINING text, each read with up to 32 tokens of run-up that are not scored
    piece, nats, characters = len(validation) // 16, 0.0, 0
    for begin in np.linspace(0, len(training) - piece - 1, 16).astype(int):
        run_up = min(32, begin)
        nats += evaluate(model, np.asarray(training[begin - run_up:begin + piece]), "cpu")[run_up:].sum()
        characters += len(tokenizer.decode(np.asarray(training[begin + 1:begin + piece]).tolist()))
    assert result["final_seen_bpc"] == pytest.approx(nats / math.log(2) / characters, abs=1e-9)
    assert float(last["seen_bpc"]) == round(result["final_seen_bpc"], 4)


def test_batches_and_the_seen_sample_come_from_training_and_validation_from_validation(tiny_run, monkeypatch):
    tokenizer = load(DATA / "tokenizers/char.json")
    a, b = tokenizer.encode("ab")
    streams = {"train": np.array([tokenizer.start_id] + [a] * 20_000), "validation": np.array([tokenizer.start_id] + [b] * 2_000)}
    monkeypatch.setattr(T, "load_tokens", lambda name, which: streams[which])
    result = T.train(dataclasses.replace(tiny_run, peak_rate=1e-2, evaluations=1), say=QUIET)     # 39 steps on "aaaa..."
    assert result["final_seen_bpc"] < 0.5 < 3.0 < result["final_validation_bpc"]                   # it knows "a" by heart and has never met "b"


def test_dropout_is_on_while_training(tiny_run, tmp_path):
    for dropout in (0.0, 0.5):
        T.train(dataclasses.replace(tiny_run, name=f"dropout-{dropout}", passes=0.05, dropout=dropout, evaluations=1), say=QUIET)   # 8 steps
    a, b = weights_of(tmp_path / "runs/dropout-0.0/last.pt"), weights_of(tmp_path / "runs/dropout-0.5/last.pt")
    assert not all(torch.equal(a[k], b[k]) for k in a)


def test_sixteen_bit_means_bfloat16_on_the_runs_device_and_is_used_only_when_asked_for(tiny_run, monkeypatch):
    asked, real = [], torch.autocast
    def spy(*a, **k):
        if "dtype" in k:                                   # the training loop's block; evaluate() opens its own, always switched off
            asked.append((a, k["dtype"], k["enabled"]))
        return real(*a, **k)
    monkeypatch.setattr(torch, "autocast", spy)
    T.train(dataclasses.replace(tiny_run, name="32", passes=0.05, evaluations=1), say=QUIET)      # 8 steps
    T.train(dataclasses.replace(tiny_run, name="16", passes=0.05, evaluations=1, sixteen_bit=True), say=QUIET)
    # bfloat16, not float16: float16 runs out of range unless the loss is scaled up first, which this loop does not do
    assert asked == [(("cpu",), torch.bfloat16, False)] * 8 + [(("cpu",), torch.bfloat16, True)] * 8


def test_every_evaluation_leaves_a_sample_from_the_same_prompt(tiny_run, tmp_path):
    # The resume tests only compare one run's samples.txt with another's, and two empty files are equal too.
    T.train(tiny_run, say=QUIET)
    text = (tmp_path / "runs/t/samples.txt").read_text(encoding="utf-8")
    assert [int(step) for step in re.findall(r"^===== step (\d+), ", text, re.M)] == [0, 26, 52, 78]
    samples = re.split(r"^===== step .*=====\n", text, flags=re.M)[1:]
    assert all(sample.startswith(T.PROMPT) and len(sample) == len(T.PROMPT) + 200 + 2 for sample in samples)   # the prompt, 200 new characters, a blank line
    assert len(set(samples)) == 4                                                                  # same prompt, same dice: they differ because the model does


def test_a_different_seed_is_a_different_start_and_different_batches(tiny_run, tmp_path, monkeypatch):
    first_batches, real_batch = [], T.random_batch
    def recording(*a):
        tokens, targets = real_batch(*a)
        first_batches.append(tokens)
        return tokens, targets
    monkeypatch.setattr(T, "random_batch", recording)
    for seed in (1, 2):
        T.train(dataclasses.replace(tiny_run, name=f"seed-{seed}", seed=seed, passes=0.05, evaluations=1), say=QUIET)
    assert not torch.equal(first_batches[0], first_batches[8])                                    # different batches
    assert log_of(tmp_path / "runs/seed-1")[0]["validation_bpc"] != log_of(tmp_path / "runs/seed-2")[0]["validation_bpc"]   # different starting weights


# ------------------------------------------------------------------------------------------ stopping, resuming, starting again
def test_a_run_that_is_cut_off_and_resumed_is_the_same_run(tiny_run, tmp_path, monkeypatch):
    batches_drawn, real_batch = [], T.random_batch
    whole = T.train(dataclasses.replace(tiny_run, name="whole"), say=QUIET)
    monkeypatch.setattr(T, "random_batch", lambda *a: batches_drawn.append(1) or real_batch(*a))
    assert T.train(dataclasses.replace(tiny_run, name="halves"), stop_at=52, say=QUIET) is None
    assert len(batches_drawn) == 52 and not (tmp_path / "runs/halves/result.json").exists()
    resumed = T.train(dataclasses.replace(tiny_run, name="halves"), resume=True, say=QUIET)
    assert len(batches_drawn) == 78                                                                # 26 more steps, not 78 more
    a, b = weights_of(tmp_path / "runs/whole/last.pt"), weights_of(tmp_path / "runs/halves/last.pt")
    assert all(torch.equal(a[k], b[k]) for k in a)                                                 # bit for bit, dropout and all (on the CPU)
    assert {**resumed, "name": "", "minutes": 0} == {**whole, "name": "", "minutes": 0}            # best step and all
    clock = ("tokens_per_second", "minutes")
    logs = [[{k: v for k, v in row.items() if k not in clock} for row in log_of(tmp_path / f"runs/{name}")] for name in ("whole", "halves")]
    assert logs[0] == logs[1] and [row["step"] for row in logs[1]] == ["0", "26", "52", "78"]       # each step logged once
    assert (tmp_path / "runs/halves/samples.txt").read_text() == (tmp_path / "runs/whole/samples.txt").read_text()
    minutes = [float(row["minutes"]) for row in log_of(tmp_path / "runs/halves")]
    # The clock carries on; it does not start again. (A real clock can only show this roughly. The exact check, with a clock we
    # control, is among the log's other columns.)
    assert minutes == sorted(minutes) and resumed["minutes"] >= minutes[2]


def test_a_run_that_died_between_saving_and_logging_gets_the_lost_row_back_from_its_checkpoint(tiny_run, tmp_path):
    T.train(tiny_run, stop_at=52, say=QUIET)
    log = tmp_path / "runs/t/log.csv"
    rows = log.read_bytes().splitlines(keepends=True)
    log.write_bytes(b"".join(rows[:-1]))                                                           # the row for step 52 never made it to disk
    T.train(tiny_run, resume=True, say=QUIET)
    assert [row["step"] for row in log_of(tmp_path / "runs/t")] == ["0", "26", "52", "78"]
    # The very row that was lost, to the byte. Scoring the step again could not bring back its training loss, gradient length,
    # share clipped or speed: the steps they were measured over are gone.
    assert log.read_bytes().splitlines(keepends=True)[:4] == rows and b"nan" not in rows[3]


# One other value for every setting of a run. The test below refuses to run if a setting is missing here, so a new field of RunConfig cannot be forgotten.
OTHER_SETTINGS = {"tokenizer": "bpe-1024", "seed": 9, "passes": 0.25, "batch": 4, "peak_rate": 5e-3, "floor_rate": 5e-5, "warm_up": 7, "weight_decay": 0.0,
                  "beta_2": 0.95, "clip": 0.5, "dropout": 0.5, "layers": 1, "heads": 4, "width": 64, "context": 16, "evaluations": 6, "sixteen_bit": True}


def test_every_setting_but_the_device_is_kept_on_resume(tiny_run, tmp_path):
    # Some settings steer the rest of the run (batch, context, the rates, the clip, 16-bit): changed, the run becomes another run half way.
    # Others are ignored by the loaded weights and optimizer (layers, heads, width, weight decay, beta_2): changed, every later
    # checkpoint would record a setting that made nothing. The device is the one exception, and has a test of its own further down.
    settings = [field.name for field in dataclasses.fields(RunConfig) if field.name not in ("name", "device")]
    assert sorted(settings) == sorted(OTHER_SETTINGS) and all(getattr(tiny_run, setting) != OTHER_SETTINGS[setting] for setting in settings)
    T.train(tiny_run, stop_at=26, say=QUIET)
    before = (tmp_path / "runs/t/last.pt").read_bytes()
    def answer_to(setting):
        try:
            T.train(dataclasses.replace(tiny_run, **{setting: OTHER_SETTINGS[setting]}), resume=True, stop_at=26, say=QUIET)
        except ValueError as refusal:
            return str(refusal)
        return "resumed"
    answers = {setting: answer_to(setting) for setting in settings}
    assert [setting for setting in settings if not re.search(f"other settings.*'{setting}'", answers[setting])] == []   # each refused, and by name
    assert (tmp_path / "runs/t/last.pt").read_bytes() == before                                    # refused before anything was touched
    assert T.train(tiny_run, resume=True, say=QUIET)["steps"] == 78                                # and with its own settings the run carries on


@pytest.mark.parametrize("changed", ["the training stream", "the validation stream", "the tokenizer file"])
def test_a_run_is_not_resumed_if_any_one_of_the_three_things_it_reads_has_changed(tiny_run, tmp_path, monkeypatch, changed):
    # One thing at a time, the other two untouched, so each of the three checksums has to do its own work.
    T.train(tiny_run, stop_at=26, say=QUIET)
    before, real = (tmp_path / "runs/t/last.pt").read_bytes(), T.load_tokens
    if changed == "the tokenizer file":
        (tmp_path / "tokenizers").mkdir()                                                          # the same tokenizer to the last id, another file by one byte
        (tmp_path / "tokenizers/char.json").write_bytes((DATA / "tokenizers/char.json").read_bytes() + b"\n")
        monkeypatch.setattr(T, "DATA", tmp_path)
    else:
        shorter = "train" if changed == "the training stream" else "validation"                   # the same stream, one token shorter
        monkeypatch.setattr(T, "load_tokens", lambda name, which: real(name, which)[:-1] if which == shorter else real(name, which))
    with pytest.raises(ValueError, match="have changed"):
        T.train(tiny_run, resume=True, say=QUIET)
    assert (tmp_path / "runs/t/last.pt").read_bytes() == before


def test_nothing_to_resume_is_said_so_and_leaves_no_folder_behind(tiny_run, tmp_path):
    # A mistyped name with --resume must not leave an empty runs/<mistyped>/ behind to be mistaken for a run.
    with pytest.raises(FileNotFoundError, match="nothing to resume"):
        T.train(dataclasses.replace(tiny_run, name="never-started"), resume=True, say=QUIET)
    assert list(tmp_path.iterdir()) == []


def test_resuming_a_finished_run_changes_nothing(tiny_run, tmp_path):
    result = T.train(tiny_run, say=QUIET)
    before = (tmp_path / "runs/t/log.csv").read_text()
    assert T.train(tiny_run, resume=True, say=QUIET) == result
    assert (tmp_path / "runs/t/log.csv").read_text() == before


def test_a_new_run_does_not_silently_replace_an_old_one(tiny_run, tmp_path):
    T.train(tiny_run, say=QUIET)
    best = (tmp_path / "runs/t/best.pt").read_bytes()
    with pytest.raises(FileExistsError):
        T.train(dataclasses.replace(tiny_run, seed=1), say=QUIET)
    assert (tmp_path / "runs/t/best.pt").read_bytes() == best
    (tmp_path / "runs/t/evaluation-best.json").write_text("{}")                                    # what scripts/evaluate.py leaves behind
    T.train(dataclasses.replace(tiny_run, seed=1), overwrite=True, stop_at=26, say=QUIET)          # asked for: allowed
    assert sorted(f.name for f in (tmp_path / "runs/t").iterdir()) == ["best.pt", "config.json", "last.pt", "log.csv", "samples.txt"]   # nothing of the old run lingers


def test_overwriting_clears_the_old_checkpoints_before_the_new_run_has_any(tiny_run, tmp_path):
    # The test above looks once the new run has saved checkpoints of its own, which would have replaced the old ones anyway. If the
    # new run dies before its first save, the old last.pt must not be lying beside the new config.json for a --resume to pick up.
    T.train(tiny_run, say=QUIET)
    class Stop(Exception):
        pass
    def dies_at_the_first_line(line):
        raise Stop
    with pytest.raises(Stop):
        T.train(dataclasses.replace(tiny_run, seed=1), overwrite=True, say=dies_at_the_first_line)   # the folder is ready, step 0 is not yet scored
    assert sorted(f.name for f in (tmp_path / "runs/t").iterdir()) == ["config.json", "log.csv", "samples.txt"]
    assert json.loads((tmp_path / "runs/t/config.json").read_text())["seed"] == 1 and log_of(tmp_path / "runs/t") == []


@pytest.mark.parametrize("bad", [{"name": "../outside"}, {"name": ""}, {"name": ".."}, {"name": "."}, {"name": "seed-1 --seed 1"}, {"name": "-x"}, {"evaluations": 0}, {"warm_up": 78}, {"warm_up": -5}, {"passes": -1.0}, {"passes": 0.0}, {"peak_rate": 0.0}, {"clip": 0.0}, {"floor_rate": -1e-4},
                                 {"batch": 0}, {"context": 0}, {"heads": 3}, {"tokenizer": "no-such-tokenizer"}])
def test_settings_that_make_no_sense_are_refused_before_anything_is_written(tiny_run, tmp_path, bad):
    with pytest.raises((ValueError, FileNotFoundError)):
        T.train(dataclasses.replace(tiny_run, **bad), say=QUIET)
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError):
        T.train(tiny_run, resume=True, overwrite=True, say=QUIET)


@pytest.mark.parametrize("stream", ["train", "validation"])
def test_token_ids_outside_the_vocabulary_are_refused_before_anything_is_written(tiny_run, tmp_path, monkeypatch, stream):
    # A stream built by another tokenizer. On the CPU the embedding table would complain later, at the first batch that holds
    # the id; the GPU may look up a row that is not there and say nothing at all.
    real, size = T.load_tokens, load(DATA / "tokenizers/char.json").vocab_size
    def one_bad_id(name, which):
        tokens = np.array(real(name, which))
        if which == stream:
            tokens[-1] = size                                                                      # ids run from 0 to size - 1: the first one that does not exist
        return tokens
    monkeypatch.setattr(T, "load_tokens", one_bad_id)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        T.train(tiny_run, say=QUIET)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the Mac's GPU")
def test_the_gpu_dice_can_be_put_back():
    # PyTorch's promise, not ours: dropout's dice live on the GPU, and they can be read and put back. This test runs no line of
    # gp_thee. The two lines of ours that lean on the promise, one in train() and one in save_checkpoint, are tested on the CPU
    # (the two tests about "the dice of the gpu" below). It is also the only test here that touches the GPU, so while a training
    # run has the GPU leave it out:  -k "not gpu_dice"
    state = torch.mps.get_rng_state()
    first = F.dropout(torch.ones(1000, device="mps"), 0.5)
    torch.mps.set_rng_state(state)
    assert torch.equal(first, F.dropout(torch.ones(1000, device="mps"), 0.5))


def test_a_resumed_run_puts_the_dice_of_the_gpu_back(tiny_run, tmp_path, monkeypatch):
    # Without this line a run resumed on the GPU would draw other dropout masks than the run that was never stopped, and nothing
    # would say so. No CPU run reaches the line, so the checkpoint is doctored and torch.mps is only listened to: the GPU is not touched.
    T.train(tiny_run, stop_at=26, say=QUIET)
    path, dice = tmp_path / "runs/t/last.pt", torch.arange(7, dtype=torch.uint8)
    saved = torch.load(path, weights_only=True)
    assert saved["random"]["gpu"] is None                                                          # a CPU run has none to save
    saved["random"]["gpu"] = dice                                                                  # as if it had been running on the GPU
    torch.save(saved, path)
    put_back = []
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda state: put_back.append(state))
    T.train(tiny_run, resume=True, stop_at=26, say=QUIET)                                          # resumes, and stops again at once
    assert len(put_back) == 1 and torch.equal(put_back[0], dice)


def test_an_unfinished_run_and_a_finished_one_without_its_last_checkpoint_are_protected_too(tiny_run, tmp_path):
    T.train(tiny_run, stop_at=26, say=QUIET)                                                       # last.pt, no result.json
    with pytest.raises(FileExistsError):
        T.train(tiny_run, say=QUIET)
    T.train(tiny_run, resume=True, say=QUIET)
    (tmp_path / "runs/t/last.pt").unlink()                                                         # tidied away to save disk; best.pt and result.json remain
    with pytest.raises(FileExistsError):
        T.train(tiny_run, say=QUIET)


# ------------------------------------------------------------------------------------------ a run killed at the worst moments
def test_a_save_that_dies_half_way_leaves_both_good_checkpoints_alone(tiny_run, tmp_path, monkeypatch):
    T.train(tiny_run, stop_at=26, say=QUIET)
    real = torch.save
    def dies_half_way(thing, path, *a, **k):
        real(thing, path, *a, **k)
        Path(path).write_bytes(Path(path).read_bytes()[:1000])
        raise KeyboardInterrupt
    monkeypatch.setattr(torch, "save", dies_half_way)
    with pytest.raises(KeyboardInterrupt):
        T.train(tiny_run, resume=True, say=QUIET)                                                  # dies while writing at step 52
    monkeypatch.setattr(torch, "save", real)
    assert list((tmp_path / "runs/t").glob("*.tmp"))                                               # half a file is lying about...
    assert [torch.load(tmp_path / f"runs/t/{name}.pt", weights_only=True)["step"] for name in ("best", "last")] == [26, 26]   # ...and both good ones are untouched
    assert T.train(tiny_run, resume=True, say=QUIET)["steps"] == 78 and not list((tmp_path / "runs/t").glob("*.tmp"))


def test_a_run_killed_between_the_two_renames_of_its_best_moment_still_ends_with_the_right_best(tiny_run, tmp_path, monkeypatch):
    # The narrowest moment of all: step 52 is the best so far, best.pt already holds it, and last.pt still says "step 26, best so far 26".
    # best.pt is renamed first on purpose. The other way round, last.pt would say "the best was step 52" beside a best.pt that holds
    # step 26; the resumed run would find nothing better at step 78, best.pt would never be put right, and the run could only end
    # by refusing to report a result.
    def script(*scores):                                                                           # validation and seen alike, in the order they are asked for
        scores = iter(scores)
        monkeypatch.setattr(T, "bits_per_character", lambda surprise, characters: next(scores))
    script(5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 3.5, 3.5)                                                 # scored at 0, 26, 52 and 78: the best is step 52
    whole = T.train(dataclasses.replace(tiny_run, name="whole"), say=QUIET)
    real, renames = T.os.replace, []
    def dies_at_the_sixth_rename(source, target):
        renames.append(Path(target).name)
        if len(renames) == 6:                                                                      # steps 0, 26 and 52 were each the best so far: two renames
            raise KeyboardInterrupt                                                                # apiece, and this is the second one at step 52
        real(source, target)
    monkeypatch.setattr(T.os, "replace", dies_at_the_sixth_rename)
    script(5.0, 5.0, 4.0, 4.0, 3.0, 3.0)
    with pytest.raises(KeyboardInterrupt):
        T.train(tiny_run, say=QUIET)
    monkeypatch.setattr(T.os, "replace", real)
    assert sorted(torch.load(tmp_path / f"runs/t/{name}.pt", weights_only=True)["step"] for name in ("best", "last")) == [26, 52]   # the two files disagree
    script(3.0, 3.0, 3.5, 3.5)                                                                     # step 52 again, then step 78
    resumed = T.train(tiny_run, resume=True, say=QUIET)
    assert {**resumed, "name": "", "minutes": 0} == {**whole, "name": "", "minutes": 0} and resumed["best"]["step"] == 52
    assert [torch.load(tmp_path / f"runs/t/{name}.pt", weights_only=True)["step"] for name in ("best", "last")] == [52, 78]
    assert [row["step"] for row in log_of(tmp_path / "runs/t")] == ["0", "26", "52", "78"] and not list((tmp_path / "runs/t").glob("*.tmp"))


def test_half_a_checkpoint_left_by_a_killed_run_is_swept_away_when_the_run_is_picked_up(tiny_run, tmp_path):
    # The two tests above end without a .tmp file whether train() sweeps or not: their resumed runs come to the same stop again, write
    # the same two names again, and rename them. A leftover can outlive that: on the GPU, where a moment that was the best by a hair
    # need not be the best when it is scored again, half a best.pt (129 MB at full size) would lie there for good.
    T.train(tiny_run, stop_at=26, say=QUIET)
    (tmp_path / "runs/t/best.pt.tmp").write_bytes(b"half a checkpoint")
    T.train(tiny_run, resume=True, stop_at=26, say=QUIET)                                          # picks the run up and stops again at once: nothing is saved
    assert sorted(f.name for f in (tmp_path / "runs/t").iterdir()) == ["best.pt", "config.json", "last.pt", "log.csv", "samples.txt"]


@pytest.mark.parametrize("dies", ["before the checkpoint", "after the checkpoint, before the log row", "after the log row, before the sample"])
def test_a_run_killed_inside_an_evaluation_resumes_into_the_same_log_and_samples(tiny_run, tmp_path, monkeypatch, dies):
    whole = T.train(dataclasses.replace(tiny_run, name="whole"), say=QUIET)
    real_save, real_generate, samples = T.save_checkpoint, T.generate, []
    def save_and_die(paths, model, optimizer, run, step, *rest):
        if step == 52 and dies == "before the checkpoint":
            raise KeyboardInterrupt
        real_save(paths, model, optimizer, run, step, *rest)
        if step == 52:
            raise KeyboardInterrupt
    def die_at_the_third_sample(*a, **k):
        samples.append(1)
        if len(samples) == 3:
            raise KeyboardInterrupt
        return real_generate(*a, **k)
    monkeypatch.setattr(T, *(("generate", die_at_the_third_sample) if dies.endswith("sample") else ("save_checkpoint", save_and_die)))
    with pytest.raises(KeyboardInterrupt):
        T.train(tiny_run, say=QUIET)
    monkeypatch.setattr(T, "save_checkpoint", real_save)
    monkeypatch.setattr(T, "generate", real_generate)
    resumed = T.train(tiny_run, resume=True, say=QUIET)
    clock = ("tokens_per_second", "minutes")
    logs = [[{k: v for k, v in row.items() if k not in clock} for row in log_of(tmp_path / f"runs/{name}")] for name in ("whole", "t")]
    assert logs[0] == logs[1] and [row["step"] for row in logs[1]] == ["0", "26", "52", "78"]
    assert (tmp_path / "runs/t/samples.txt").read_text() == (tmp_path / "runs/whole/samples.txt").read_text()
    assert resumed["best"] == whole["best"] and torch.load(tmp_path / "runs/t/best.pt", weights_only=True)["step"] == whole["best"]["step"]


def test_a_run_whose_best_file_is_not_its_best_moment_does_not_report_a_result(tiny_run, tmp_path, monkeypatch):
    T.train(tiny_run, stop_at=52, say=QUIET)
    (tmp_path / "runs/t/best.pt").write_bytes((tmp_path / "runs/t/last.pt").read_bytes())          # as if something had put the wrong file there
    monkeypatch.setattr(T, "bits_per_character", lambda surprise, characters: 99.0)                # the last stop is no better, so best.pt is not rewritten
    saved = torch.load(tmp_path / "runs/t/last.pt", weights_only=True)
    saved["facts"]["best"]["step"] = 26
    torch.save(saved, tmp_path / "runs/t/last.pt")
    with pytest.raises(AssertionError, match="best moment"):
        T.train(tiny_run, resume=True, say=QUIET)
    assert not (tmp_path / "runs/t/result.json").exists()


# ------------------------------------------------------------------------------------------ what made this checkpoint
def test_the_git_stamp_names_the_commit_and_notices_only_changes_that_can_change_a_run(tmp_path, monkeypatch):
    git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *a], cwd=tmp_path,
                                    check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("x = 1\n")
    (tmp_path / "notes.md").write_text("hello\n")
    git("add", ".")
    git("commit", "-qm", "first")
    monkeypatch.setattr(T, "ROOT", tmp_path)
    assert T.git_commit() == git("rev-parse", "HEAD") and len(T.git_commit()) == 40
    (tmp_path / "notes.md").write_text("a note cannot change a run\n")
    assert T.git_commit() == git("rev-parse", "HEAD")
    (tmp_path / "src/a.py").write_text("x = 2\n")
    assert T.git_commit().startswith(git("rev-parse", "HEAD")) and "uncommitted" in T.git_commit()


def test_the_stamp_is_taken_once_when_the_run_starts(tiny_run, tmp_path, monkeypatch):
    stamps = iter("abcdefgh")
    monkeypatch.setattr(T, "git_commit", lambda: next(stamps) * 40)
    T.train(tiny_run, stop_at=26, say=QUIET)
    assert torch.load(tmp_path / "runs/t/last.pt", weights_only=True)["git_commit"] == "a" * 40 == json.loads((tmp_path / "runs/t/config.json").read_text())["git_commit"]


@pytest.mark.filterwarnings("ignore:for .*copying from a non-meta")
def test_loading_a_checkpoint_runs_nothing_and_lands_on_the_cpu(tiny_run, tmp_path, monkeypatch):
    torch.save({"model_config": Path("not a number, a text or a list")}, tmp_path / "stranger.pt")
    with pytest.raises(pickle.UnpicklingError):
        T.load_checkpoint(tmp_path / "stranger.pt", "cpu")
    T.train(tiny_run, stop_at=0, say=QUIET)
    asked, real = {}, torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: asked.update(k) or real(*a, **k))
    T.load_checkpoint(tmp_path / "runs/t/last.pt", "meta")      # "meta" stands in for the GPU: AdamW's counters must not follow the model there
    assert asked["map_location"] == "cpu" and asked["weights_only"] is True


@pytest.mark.filterwarnings("ignore:for .*copying from a non-meta")
def test_a_loaded_checkpoint_keeps_the_optimizer_counters_and_the_dice_on_the_cpu(tiny_run, tmp_path):
    # The test above checks how torch.load is asked; this one checks what comes back, whichever way it was made. AdamW reads its
    # step counters as plain numbers at every step. Counters on the GPU would stop Python once per tensor per step to fetch them,
    # and a resumed run would crawl. The state of PyTorch's own dice belongs on the CPU too, where those dice live.
    T.train(tiny_run, stop_at=26, say=QUIET)                    # 26 steps, so the optimizer has counters to keep
    model, saved = T.load_checkpoint(tmp_path / "runs/t/last.pt", "meta")
    assert next(model.parameters()).device.type == "meta"       # the model goes where it was asked to go...
    counters = [state["step"] for state in saved["optimizer"]["state"].values()]
    assert len(counters) == sum(len(group["params"]) for group in saved["optimizer"]["param_groups"]) == 15   # (one counter for every tensor of weights)
    assert {counter.device.type for counter in counters} == {"cpu"} and saved["random"]["torch"].device.type == "cpu"   # ...and these stay behind


def test_a_checkpoint_made_on_the_gpu_holds_the_dice_of_the_gpu(tmp_path, monkeypatch):
    # The other line no CPU run reaches. A stand-in model that only SAYS it is on the GPU, and a torch.mps that is only listened to.
    model, dice = tiny_model(), torch.arange(5, dtype=torch.uint8)
    on_the_gpu = types.SimpleNamespace(parameters=lambda: iter([types.SimpleNamespace(device=torch.device("mps"))]), state_dict=model.state_dict, config=model.config)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: dice)
    T.save_checkpoint([tmp_path / "x.pt"], on_the_gpu, torch.optim.AdamW(model.parameters()), RunConfig(name="x"), 0, np.random.default_rng(0), {}, {})
    kept = torch.load(tmp_path / "x.pt", weights_only=True)["random"]["gpu"]
    assert kept is not None and torch.equal(kept, dice)


def test_a_run_whose_folder_was_renamed_can_still_be_resumed(tiny_run, tmp_path):
    whole = T.train(dataclasses.replace(tiny_run, name="whole"), say=QUIET)
    T.train(tiny_run, stop_at=26, say=QUIET)
    (tmp_path / "runs/t").rename(tmp_path / "runs/a-better-name")                                  # the name does not steer a run; everything else does
    resumed = T.train(dataclasses.replace(tiny_run, name="a-better-name"), resume=True, say=QUIET)
    assert {**resumed, "name": "", "minutes": 0} == {**whole, "name": "", "minutes": 0}


def test_a_run_begun_on_one_device_may_be_resumed_on_another(tiny_run, tmp_path):
    T.train(tiny_run, stop_at=26, say=QUIET)
    saved = torch.load(tmp_path / "runs/t/last.pt", weights_only=True)
    saved["run_config"]["device"] = "mps"                         # as if it had begun on the GPU
    torch.save(saved, tmp_path / "runs/t/last.pt")
    assert T.train(tiny_run, resume=True, say=QUIET)["steps"] == 78


def test_a_checkpoint_from_the_first_version_of_this_file_can_be_read_but_not_resumed(tiny_run, tmp_path):
    T.train(tiny_run, stop_at=26, say=QUIET)
    saved = torch.load(tmp_path / "runs/t/last.pt", weights_only=True)
    saved["torch"] = torch.__version__                            # the first version stored this object, not its text...
    del saved["fingerprints"]                                     # ...and no checksums
    torch.save(saved, tmp_path / "runs/t/last.pt")
    model, again = T.load_checkpoint(tmp_path / "runs/t/last.pt", "cpu")
    assert again["step"] == 26
    with pytest.raises(ValueError, match="older train.py"):
        T.train(tiny_run, resume=True, say=QUIET)


# ------------------------------------------------------------------------------------------ the log's other columns
def test_the_clock_carries_on_and_tokens_per_second_times_the_training_steps_only(tiny_run, tmp_path, monkeypatch):
    now, real_batch, real_generate = [0.0], T.random_batch, T.generate
    def one_second_per_step(*a):
        now[0] += 1.0
        return real_batch(*a)
    def a_minute_per_sample(*a, **k):
        now[0] += 60.0
        return real_generate(*a, **k)
    def clock():
        now[0] += 1e-6                                           # time never stands still between two readings
        return now[0]
    monkeypatch.setattr(T, "random_batch", one_second_per_step)
    monkeypatch.setattr(T, "generate", a_minute_per_sample)
    monkeypatch.setattr(T.time, "perf_counter", clock)
    T.train(tiny_run, stop_at=52, say=QUIET)
    now[0] = 1_000_000.0                                         # the machine was off for a while
    result = T.train(tiny_run, resume=True, say=QUIET)
    rows = log_of(tmp_path / "runs/t")
    assert [float(row["minutes"]) for row in rows] == [0.0, round(86 / 60, 2), round(172 / 60, 2), round(198 / 60, 2)] and result["minutes"] == 3.3
    assert rows[0]["tokens_per_second"] == "nan" and [int(row["tokens_per_second"]) for row in rows[1:]] == [8 * 32] * 3   # one step a second, whatever the stops cost


def test_share_clipped_is_the_share_of_steps_whose_gradients_were_longer_than_the_clip(tiny_run, tmp_path, monkeypatch):
    lengths, real = [], torch.nn.utils.clip_grad_norm_
    def spy(parameters, clip):
        lengths.append(float(real(parameters, clip)))
        return torch.tensor(lengths[-1])
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    T.train(tiny_run, say=QUIET)
    shares = [float(row["share_clipped"]) for row in log_of(tmp_path / "runs/t")[1:]]
    assert shares == [round(float(np.mean(np.array(lengths[i:i + 26]) > 1.0)), 4) for i in (0, 26, 52)] and 0 < shares[-1] < 1


def test_the_first_row_of_the_log_claims_nothing_about_steps_that_were_never_taken(tiny_run, tmp_path):
    # Before the first step no learning rate has been used, and there is no training loss, no gradient and no speed to report.
    # "nan" says so. The optimizer's 0.001, or a speed of 0, would be a claim about steps that did not happen.
    T.train(tiny_run, stop_at=0, say=QUIET)
    first = log_of(tmp_path / "runs/t")[0]
    no_steps_yet = ("learning_rate", "training_loss", "gradient_length", "share_clipped", "tokens_per_second")
    assert {column: first[column] for column in no_steps_yet} == dict.fromkeys(no_steps_yet, "nan")
    assert all(math.isfinite(float(first[column])) for column in T.LOG_COLUMNS if column not in no_steps_yet)   # what CAN be known at step 0 is there


# ------------------------------------------------------------------------------------------ the command line
def command_line(monkeypatch, *flags):
    """Run scripts/train.py for real, with train() swapped for a recorder, so nothing trains."""
    calls = []
    monkeypatch.setattr(T, "train", lambda run, **options: calls.append((run, options)))
    monkeypatch.setattr(sys, "argv", ["train.py", *flags])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return calls[0]


def test_the_command_line_builds_the_run_and_resume_takes_it_from_the_folder(tiny_run, tmp_path, monkeypatch):
    began = dataclasses.replace(tiny_run, name="x", tokenizer="bpe-1024", seed=3, passes=0.25, sixteen_bit=True)
    flags = ["--name", "x", "--tokenizer", "bpe-1024", "--seed", "3", "--passes", "0.25", "--batch", "8", "--context", "32", "--layers", "2", "--heads", "2",
             "--width", "32", "--dropout", "0.1", "--evaluations", "3", "--warm-up", "5", "--device", "cpu", "--sixteen-bit"]
    assert command_line(monkeypatch, *flags) == (began, {"resume": False, "overwrite": False})
    assert command_line(monkeypatch, "--name", "y", "--overwrite") == (RunConfig(name="y"), {"resume": False, "overwrite": True})
    (tmp_path / "runs/x").mkdir(parents=True)
    (tmp_path / "runs/x/config.json").write_text(json.dumps({**dataclasses.asdict(began), "steps": 39, "parameters": 1, "git_commit": "unknown", "fingerprints": {}}))
    assert command_line(monkeypatch, "--name", "x", "--resume") == (began, {"resume": True, "overwrite": False})
    assert command_line(monkeypatch, "--name", "x", "--resume", "--device", "mps", "--seed", "4")[0] == dataclasses.replace(began, device="mps", seed=4)  # flags survive, for train() to judge
    with pytest.raises(SystemExit):
        command_line(monkeypatch, "--name", "never-started", "--resume")
