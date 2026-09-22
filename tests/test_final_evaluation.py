"""Tests for a script that must never be run in a test.

`scripts/final_evaluation.py` opens the three test works once and can never be run again, so nothing here calls
its `main()`. What can still be pinned is everything that decides whether the one run produces the right number:
that the stream it will build is byte-identical to the streams every published figure was made from, that the bar
it computes reproduces the one part 5 published, that the gate refuses in the right order, and that the arms are
the ones docs/BUILD_LOG.md entry 19 fixed by name.

The rest -- what the test works actually score -- is the one thing in this project no test can ever check.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import final_evaluation as fe  # noqa: E402

from gp_thee.data import DATA, load_tokens, load_works  # noqa: E402
from gp_thee.tokenizer import load as load_tokenizer  # noqa: E402

RELEASE_CHECKPOINT = ROOT / "runs" / "sweep-char-seed-1" / "best.pt"


# ------------------------------------------------------------------------------------ the stream it will build
@pytest.mark.parametrize("kind", ["char", "bpe-1024", "bpe-4096"])
def test_the_stream_it_builds_is_the_stream_every_published_figure_used(kind):
    # If this differs by one token the test score is not comparable with anything, and nothing would say so.
    tokenizer = load_tokenizer(DATA / "tokenizers" / f"{kind}.json")
    mine = fe.stream_of(load_works("validation"), tokenizer)
    assert np.array_equal(mine, np.asarray(load_tokens(kind, "validation")))
    assert mine.dtype == np.uint16


def test_the_character_count_is_characters_not_tokens():
    works = load_works("validation")
    assert fe.characters_of(works) == 274727                    # split.json's figure, published in part 2
    assert len(fe.stream_of(works, load_tokenizer(DATA / "tokenizers" / "char.json"))) == 274727 + len(works)


def test_what_the_test_stream_will_have_to_be():
    # Asserted here from figures published in part 2, so that a mismatch on the night is a refusal and not a result.
    sets = json.loads((ROOT / "data" / "processed" / "split.json").read_text())["sets"]["test"]
    assert sets["chars"] == 233161 and sets["works"] == 3


# ------------------------------------------------------------------------------------ the bar it will compute
@pytest.mark.slow
def test_the_bar_reproduces_the_one_part_5_published():
    # baseline_table is not imported from scripts/baselines.py, which has no main() guard. So it is checked
    # against that script's committed output instead, row by row, on the works both are allowed to see.
    published = json.loads((ROOT / "docs" / "baselines.json").read_text())["results"]
    got = fe.baseline_table(load_works("validation"), load_works("train"))
    checked = 0
    for name, row in published.items():
        assert name in got, name
        assert got[name]["bits_per_character"] == pytest.approx(row["bits_per_character"], abs=1e-9), name
        for slice_name in ("speaker_labels", "everything_else"):
            if slice_name in row:
                assert got[name][slice_name] == pytest.approx(row[slice_name], abs=1e-9), f"{name}/{slice_name}"
        checked += 1
    assert checked >= 11                                        # blind guess, counts, 2..8-gram, bzip2, xz


# ------------------------------------------------------------------------------------ the gate
def test_the_gate_refuses_when_the_measurement_has_already_been_taken(tmp_path, monkeypatch):
    monkeypatch.setattr(fe, "OUT", tmp_path / "final-evaluation.json")
    monkeypatch.setattr(fe, "ATTEMPTS", tmp_path / "attempts.jsonl")
    fe.OUT.write_text("{}")
    with pytest.raises(SystemExit) as refused:
        fe.gate()
    assert "already exists" in str(refused.value)
    assert "not opened" in str(refused.value)
    assert fe.ATTEMPTS.exists() and json.loads(fe.ATTEMPTS.read_text())["outcome"] == "refused"


def test_the_gate_refuses_a_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(fe, "OUT", tmp_path / "nothing.json")
    monkeypatch.setattr(fe, "ATTEMPTS", tmp_path / "attempts.jsonl")
    def fake_git(*args, **kwargs):
        if args[0] == "ls-files":
            return "docs/release.json"
        if args[0] == "status":
            return "" if "release" in args[-1] else " M src/gp_thee/train.py"
        return "abc123"
    monkeypatch.setattr(fe, "git", fake_git)
    with pytest.raises(SystemExit) as refused:
        fe.gate()
    assert "not clean" in str(refused.value) and "train.py" in str(refused.value)


def test_every_refusal_is_recorded_before_it_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(fe, "ATTEMPTS", tmp_path / "attempts.jsonl")
    for why in ("first", "second"):
        with pytest.raises(SystemExit):
            fe.refuse(why)
    lines = [json.loads(line) for line in fe.ATTEMPTS.read_text().splitlines()]
    assert [line["detail"] for line in lines] == ["first", "second"]      # append-only, nothing overwritten
    assert all(line["outcome"] == "refused" and line["at"] for line in lines)


# ------------------------------------------------------------------------------------ what entry 19 fixed
def test_the_arms_are_the_ones_entry_19_fixed_by_name():
    assert fe.RELEASED == ("sweep-char-seed-1", "best")
    assert fe.HEADLINE == [(f"sweep-char-seed-{s}", "best") for s in (1, 2, 3)]
    assert len(fe.CHARACTER_ARMS) == 9 and len(fe.FRAGMENT_ARMS) == 12 and len(fe.ARMS) == 21
    assert len(set(fe.ARMS)) == 21                                        # no arm scored twice
    checkpoints = [(name, which, ROOT / "runs" / name / f"{which}.pt") for name, which in fe.ARMS]
    if not any(path.exists() for _, _, path in checkpoints):
        pytest.skip("local training checkpoints are hosted separately and are not distributed through git")
    missing = [f"{name}/{which}" for name, which, path in checkpoints if not path.exists()]
    assert not missing, f"local checkpoint collection is incomplete: {', '.join(missing)}"


def test_the_released_arm_matches_the_committed_release():
    chosen = json.loads((ROOT / "docs" / "release.json").read_text())
    assert (chosen["run"], chosen["checkpoint"].removesuffix(".pt")) == fe.RELEASED


def test_the_unlock_string_is_the_one_the_lock_wants():
    from gp_thee.data import UNLOCK
    assert fe.UNLOCK == UNLOCK
    with pytest.raises(PermissionError):
        load_works("test")                                                # the lock still holds without it


# ------------------------------------------------------------------------------------ the headline arithmetic
def test_the_headline_ranks_the_released_run_the_right_way_round():
    # Lower bits per character is better, so rank 1 must be the lowest score.
    out = {"about": {"released": {"step": 9236}},
           "arms": {"sweep-char-seed-1/best": {"test_bpc": 1.80, "validation_bpc": 1.7492,
                                               "taken_apart": {"works": [                # a LIST, as breakdown returns
                                                   {"title": "A", "bits": 200.0, "characters": 100},
                                                   {"title": "B", "bits": 300.0, "characters": 100}]}},
                    "sweep-char-seed-2/best": {"test_bpc": 1.70},
                    "sweep-char-seed-3/best": {"test_bpc": 1.75}}}
    head = fe._headline(out)
    assert head["the_recipe"]["released_rank_on_test"] == 3                # 1.80 is the worst of the three
    assert head["the_recipe"]["mean"] == pytest.approx(1.75)
    assert head["the_recipe"]["spread"] == pytest.approx(0.10)
    assert head["the_artifact"]["test_bits_per_character"] == 1.80
    left = head["the_artifact"]["leave_one_work_out"]
    assert left["without A"] == pytest.approx(3.0) and left["without B"] == pytest.approx(2.0)


def test_the_headline_survives_a_block_that_failed():
    # block() records a failure and returns None; the headline must still be writable rather than raising.
    head = fe._headline({"about": {"released": {"step": 9236}}, "arms": {}})
    assert head["the_artifact"]["test_bits_per_character"] is None
    assert head["the_recipe"]["mean"] is None and head["the_recipe"]["released_rank_on_test"] is None


def test_an_arm_saves_its_raw_surprise_before_computing_a_statistic(tmp_path, monkeypatch):
    """Post-run hardening: a failed reduction must never precede preservation of its raw evidence."""
    surprise = np.array([0.25, 0.5, 0.75])
    monkeypatch.setattr(fe, "ARRAYS", tmp_path)
    monkeypatch.setattr(fe, "load_checkpoint", lambda *_: (object(), {"step": 1}))
    monkeypatch.setattr(fe, "evaluate", lambda *_args, **_kwargs: surprise)

    class ReductionFailed(RuntimeError):
        pass

    def fail_after_save(*_):
        assert (tmp_path / "sweep-char-seed-1--best.npy").exists()
        raise ReductionFailed

    monkeypatch.setattr(fe, "bits_per_character", fail_after_save)
    with pytest.raises(ReductionFailed):
        fe._one_arm("sweep-char-seed-1", "best", np.array([1]), object(), ["x"], 1, "cpu")
    assert np.array_equal(np.load(tmp_path / "sweep-char-seed-1--best.npy"), surprise)


# ------------------------------------------------------------------------------------ the third arm's instrument
def test_part_8_uses_part_7s_own_scan_and_not_a_copy():
    import memorisation as part7
    assert fe.scan_one is part7.scan_one
    source = (ROOT / "scripts" / "final_evaluation.py").read_text()
    assert "def agreement" not in source and "candidates(" not in source   # nothing re-implemented here


@pytest.mark.skipif(not RELEASE_CHECKPOINT.exists(),
                    reason="the released checkpoint is hosted on Hugging Face, not distributed through git")
def test_the_headline_reads_breakdowns_real_shape_not_a_dict():
    # Three independent auditors caught _headline calling .items() on breakdown()["works"], which is a LIST.
    # It would have crashed AFTER the works were opened, destroying the one measurement. Pinned here.
    import numpy as np
    from gp_thee.evaluation import breakdown
    from gp_thee.train import evaluate, load_checkpoint
    works = load_works("validation")
    tokenizer = load_tokenizer(DATA / "tokenizers" / "char.json")
    stream = fe.stream_of(works, tokenizer)
    model, _ = load_checkpoint(ROOT / "runs" / "sweep-char-seed-1" / "best.pt", "cpu")
    taken_apart = breakdown(evaluate(model, stream, "cpu"), stream, tokenizer, works)
    assert isinstance(taken_apart["works"], list)
    assert all({"title", "bits", "characters"} <= set(row) for row in taken_apart["works"])
    head = fe._headline({"about": {"released": {"step": 9236}},
                         "arms": {"sweep-char-seed-1/best": {"test_bpc": 1.75, "validation_bpc": 1.75,
                                                             "taken_apart": taken_apart}}})
    left = head["the_artifact"]["leave_one_work_out"]
    assert len(left) == 2 and all(v is not None for v in left.values())
    for row in taken_apart["works"]:
        rest = [r for r in taken_apart["works"] if r["title"] != row["title"]]
        assert left[f"without {row['title']}"] == pytest.approx(
            sum(r["bits"] for r in rest) / sum(r["characters"] for r in rest))
