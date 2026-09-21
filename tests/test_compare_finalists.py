"""scripts/compare_finalists.py puts two models with DIFFERENT tokenizers on the very same text, character by character. Every choice
in it was written down before the sweep (docs/BUILD_LOG.md, addendum to entry 15, item 7); these tests pin the program to those words.

No model is loaded and no GPU is touched: the "surprise" of a model is a row of random numbers, or one made by hand.
"""

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from gp_thee.data import DATA, load_tokens, load_works
from gp_thee.evaluation import breakdown, piece_lengths
from gp_thee.tokenizer import load

spec = importlib.util.spec_from_file_location("compare_finalists", Path(__file__).resolve().parent.parent / "scripts" / "compare_finalists.py")
finalists = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalists)
bits_by_character, blocks_of, interval = finalists.bits_by_character, finalists.blocks_of, finalists.interval

TOY = types.SimpleNamespace(vocab=["a", "bc", "def", "<START>"], start_id=3)     # all that piece_lengths() asks of a tokenizer


# ------------------------------------------------------------------------------------------ one number per character
def test_a_tokens_bits_are_shared_equally_among_its_characters_and_starts_bits_go_to_the_first_character_of_the_work_it_opens():
    stream = [3, 0, 1, 3, 2, 0]                                                  # START a bc | START def a
    surprise = np.array([1.0, 4.0, 8.0, 6.0, 3.0]) * math.log(2)                 # in nats, as train.evaluate gives them; the first START was given, not predicted
    assert bits_by_character(surprise, stream, TOY).tolist() == [1.0, 2.0, 2.0, 2.0 + 8.0, 2.0, 2.0, 3.0]   # the second START's 8 bits land on the "d" that follows it
    assert bits_by_character(surprise, np.array(stream, dtype=np.uint16), TOY).sum() == 22.0   # the saved streams are 16-bit and unsigned: nothing may wrap round


@pytest.mark.parametrize("name", ["char", "bpe-1024"])
def test_on_the_real_validation_text_the_characters_bits_add_up_work_by_work_to_what_the_breakdown_says(name):
    tokenizer, stream, works = load(DATA / "tokenizers" / f"{name}.json"), load_tokens(name, "validation"), load_works("validation")
    surprise = np.random.default_rng(7).exponential(2.0, size=len(stream) - 1)
    bits, parts = bits_by_character(surprise, stream, tokenizer), breakdown(surprise, stream, tokenizer, works)
    # one number per character of the same text, nothing lost, and every START paid for by the work it opens
    assert len(bits) == sum(map(len, works)) and bits.sum() == pytest.approx(surprise.sum() / math.log(2), rel=1e-12)
    edges = np.cumsum([0] + [len(work) for work in works])
    assert [bits[a:b].sum() for a, b in zip(edges, edges[1:])] == pytest.approx([work["bits"] for work in parts["works"]], rel=1e-12)
    ids = np.asarray(stream).astype(np.int64)
    second = int(np.flatnonzero(ids == tokenizer.start_id)[1])                   # the START that opens the second work, and the piece after it
    share = surprise[second] / piece_lengths(tokenizer)[ids[second + 1]]
    assert bits[edges[1]] == pytest.approx((surprise[second - 1] + share) / math.log(2), rel=1e-12)


# ------------------------------------------------------------------------------------------ blocks
def test_blocks_are_5000_characters_never_cross_a_play_and_cover_every_character_once():
    assert finalists.BLOCK == 5_000
    assert blocks_of(["a" * 12_001, "b" * 5_000, "c" * 3]) == [(0, 0, 5_000), (0, 5_000, 10_000), (0, 10_000, 12_001), (1, 12_001, 17_001), (2, 17_001, 17_004)]
    works = load_works("validation")
    blocks, edges = blocks_of(works), np.cumsum([0] + [len(work) for work in works])
    assert [begin for _, begin, _ in blocks] == [0] + [end for _, _, end in blocks[:-1]] and blocks[-1][2] == edges[-1]   # end to end: every character once
    assert all(edges[play] <= begin < end <= edges[play + 1] and end - begin <= 5_000 for play, begin, end in blocks)     # each block inside its own play
    assert len(blocks) == sum(math.ceil(len(work) / 5_000) for work in works)                                             # only the last block of a play is short


# ------------------------------------------------------------------------------------------ the interval
def by_the_book(difference, blocks, resamples=10_000):
    """The words of the build log, written out slowly: blocks drawn again with replacement, play by play, default_rng(0), 95% by percentiles."""
    dice, draws = np.random.default_rng(0), []
    for _ in range(resamples):
        bits = characters = 0
        for play in sorted({play for play, _, _ in blocks}):
            mine = [(begin, end) for p, begin, end in blocks if p == play]
            for pick in dice.choice(len(mine), size=len(mine)):
                bits, characters = bits + difference[mine[pick][0]:mine[pick][1]].sum(), characters + mine[pick][1] - mine[pick][0]
        draws.append(bits / characters)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def test_the_interval_is_10000_resamples_of_blocks_play_by_play_with_fixed_dice_and_95_percent_by_percentiles(monkeypatch):
    works = ["a" * 23, "b" * 17, "c" * 9]
    monkeypatch.setattr(finalists, "BLOCK", 5)                                   # small blocks, short last ones: the statistic must weigh a block by its size
    difference, blocks = np.random.default_rng(3).normal(0.02, 0.1, size=49), blocks_of(works)
    assert finalists.RESAMPLES == 10_000 and len(blocks) == 5 + 4 + 2
    low, high = interval(difference, blocks)
    assert (low, high) == pytest.approx(by_the_book(difference, blocks), rel=1e-9) and low < difference.mean() < high
    assert interval(difference, blocks) == (low, high)                           # the dice are fixed: the same numbers every time


def test_a_difference_that_is_the_same_everywhere_gets_an_interval_that_is_just_that_difference(monkeypatch):
    monkeypatch.setattr(finalists, "RESAMPLES", 200)
    blocks = blocks_of(["a" * 12_001, "b" * 7_000])
    assert interval(np.full(19_001, 0.0125), blocks) == pytest.approx((0.0125, 0.0125), rel=1e-12)


def test_blocks_are_drawn_within_their_own_play_so_each_play_keeps_its_weight(monkeypatch):
    # One play where the second model is better everywhere, one where it is worse everywhere. Drawn play by play, every resample holds
    # the same blocks of each play as the text does, so nothing varies. Drawn from one pot, the mix of plays would vary, and the interval with it.
    monkeypatch.setattr(finalists, "RESAMPLES", 200)
    blocks = blocks_of(["a" * 20_000, "b" * 10_000])
    difference = np.concatenate([np.full(20_000, 0.03), np.full(10_000, -0.03)])
    assert interval(difference, blocks) == pytest.approx((0.01, 0.01), rel=1e-12)


# ------------------------------------------------------------------------------------------ the whole script
def test_the_finalists_are_characters_and_the_word_fragment_arm_with_the_lowest_mean_each_arm_averaged_over_its_seeds(tmp_path, monkeypatch, capsys):
    works = load_works("validation")
    characters, edges = sum(map(len, works)), np.cumsum([0] + [len(work) for work in works])
    rate = {"char": (1.80, 1.80), "bpe-2048": (1.70, 1.84)}                      # bits per character in the first play and in the second

    def surprise_of(arm, seed):
        """A model that charges every character of a play the same, a little more or less from seed to seed; START costs nothing."""
        ids = np.asarray(load_tokens(arm, "validation")).astype(np.int64)
        lengths = piece_lengths(load(DATA / "tokenizers" / f"{arm}.json"))[ids]
        second_play = np.cumsum(lengths) > edges[1]
        return (lengths * np.where(second_play, rate[arm][1], rate[arm][0]) * (1 + 0.01 * (seed - 2)) * math.log(2))[1:]

    scores = {"char": [1.80, 1.80, 1.80], "bpe-1024": [1.60, 1.95, 1.95], "bpe-2048": [1.75, 1.77, 1.79]}   # bpe-1024 has the best single run, bpe-2048 the lowest mean
    for arm, by_seed in scores.items():
        for seed, score in zip((1, 2, 3), by_seed):
            folder = tmp_path / "runs" / f"sweep-{arm}-seed-{seed}"
            folder.mkdir(parents=True)
            if arm in rate:
                score = surprise_of(arm, seed).sum() / math.log(2) / characters                          # what the run recorded is what its checkpoint scores
            (folder / "config.json").write_text(json.dumps({"tokenizer": arm, "seed": seed}))
            (folder / "result.json").write_text(json.dumps({"best": {"validation_bpc": score}}))
    (tmp_path / "runs" / "sweep-bpe-4096-seed-1").mkdir()                        # unfinished: no part of anything
    (tmp_path / "runs" / "sweep-bpe-4096-seed-1" / "config.json").write_text(json.dumps({"tokenizer": "bpe-4096", "seed": 1}))
    (tmp_path / "docs").mkdir()
    loaded = []
    monkeypatch.setattr(finalists, "ROOT", tmp_path)
    monkeypatch.setattr(finalists, "RESAMPLES", 200)
    monkeypatch.setattr(finalists, "load_checkpoint", lambda path, device: (loaded.append((path.parent.name, path.name, device)) or path.parent.name, None))
    monkeypatch.setattr(finalists, "evaluate", lambda model, stream, device: surprise_of(model.split("-seed-")[0].removeprefix("sweep-"), int(model.rsplit("-", 1)[1])))
    monkeypatch.setattr(sys, "argv", ["compare_finalists.py", "--device", "cpu"])
    finalists.main()
    out = json.loads((tmp_path / "docs" / "finalists.json").read_text())
    assert out["arms"] == ["char", "bpe-2048"] and out["runs"] == {arm: [f"sweep-{arm}-seed-{seed}" for seed in (1, 2, 3)] for arm in ("char", "bpe-2048")}
    assert loaded == [(f"sweep-{arm}-seed-{seed}", "best.pt", "cpu") for arm in ("char", "bpe-2048") for seed in (1, 2, 3)]   # each run at its BEST checkpoint
    first, second = (title.split("\n", 1)[0] for title in works)
    assert out["by_play"] == {first: pytest.approx(-0.10), second: pytest.approx(0.04)}                  # the mean over seeds 1, 2, 3 is the rate itself; a sign per play
    difference = np.concatenate([np.full(len(works[0]), -0.10), np.full(len(works[1]), 0.04)])
    assert out["difference"] == pytest.approx(difference.mean()) and "bpe-2048 minus char" in out["difference_is"]
    assert out["interval_95"] == pytest.approx(by_the_book(difference, blocks_of(works), 200)) and out["interval_95"][0] < out["difference"] < out["interval_95"][1]
    assert (out["blocks"], out["block_characters"]) == (len(blocks_of(works)), 5_000)
    assert out["share_of_blocks_where_the_first_arm_is_better"] == pytest.approx(math.ceil(len(works[1]) / 5_000) / out["blocks"])
    assert "cannot overturn the rule" in capsys.readouterr().out

    # a checkpoint that does not score what its run recorded is not quietly averaged in
    (tmp_path / "runs" / "sweep-char-seed-2" / "result.json").write_text(json.dumps({"best": {"validation_bpc": 1.79}}))
    with pytest.raises(AssertionError, match="sweep-char-seed-2: the per-character bits do not add up"):
        finalists.main()
