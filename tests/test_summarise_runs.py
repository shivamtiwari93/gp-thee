"""The script that applies the rule for calling two settings different gets a test of its own: its arithmetic decides which model we keep."""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "summarise_runs.py"


def summarise(tmp_path, runs, *arguments):
    """Build a fake runs/ tree, run a copy of the real script on it, and return what it printed and what it wrote."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "docs").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts")
    for name, tokenizer, parameters, seed, best, extra in runs:
        folder = tmp_path / "runs" / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"name": name, "tokenizer": tokenizer, "seed": seed, "steps": 100, "parameters": parameters,
                                                        "git_commit": extra.get("git_commit", "x"), "torch": "2.14.0", "fingerprints": {"tokenizer": tokenizer}}))
        if not extra.get("unfinished"):
            (folder / "result.json").write_text(json.dumps({"best": {"validation_bpc": best, "step": extra.get("best_step", 50), "passes": 1.0, "seen_bpc": 1.0},
                                                            "final_validation_bpc": best + 0.1, "minutes": 1.0}))
    printed = subprocess.run([sys.executable, str(tmp_path / "scripts" / "summarise_runs.py"), *arguments], capture_output=True, text=True, check=True).stdout
    written = sorted((tmp_path / "docs").glob("results*.json"))
    return printed, json.loads(written[-1].read_text()), written[-1].name


def three_seeds(name, tokenizer, parameters, shift):
    return [(f"{name}-{seed}", tokenizer, parameters, seed, best + shift, {}) for seed, best in ((1, 1.773), (2, 1.765), (3, 1.781))]


def test_repeats_are_averaged_the_spread_is_pooled_and_the_rule_is_applied(tmp_path):
    runs = [("a-1", "char", 10, 1, 1.770, {}), ("a-1-again", "char", 10, 1, 1.776, {}), ("a-2", "char", 10, 2, 1.765, {}), ("a-3", "char", 10, 3, 1.781, {})]  # seed 1 twice: 1.773 once averaged
    runs += three_seeds("b", "bpe-1024", 11, 0.014) + three_seeds("c", "bpe-2048", 12, 0.020) + [("d-unfinished", "bpe-4096", 13, 1, 9.9, {"unfinished": True})]
    printed, written, name = summarise(tmp_path, runs)
    a, b, c = written["groups"]
    assert name == "results.json" and (a["seeds"], b["seeds"], c["seeds"]) == (3, 3, 3)
    assert a["mean_best_validation_bpc"] == pytest.approx(1.773) and a["same_seed_differences"] == {"1": pytest.approx(0.006)}
    assert [g["standard_deviation_over_seeds"] for g in (a, b, c)] == pytest.approx([0.008] * 3)             # n - 1 in the denominator
    assert written["pooled_standard_deviation"] == pytest.approx(0.008) and written["degrees_of_freedom"] == 6
    needed = 2.447 * 0.008 * math.sqrt(2 / 3)                                                                 # t(0.975, 6)
    assert 0.014 < needed < 0.020
    verdicts = [line.rsplit("->", 1)[1].strip() for line in printed.splitlines() if "against" in line]
    assert verdicts == ["a tie", "DIFFERENT", "a tie"] and printed.count(f"{needed:.4f} needed") == 3
    assert written["choice"]["settings"]["tokenizer"] == "char"                                               # the lowest mean, and the smallest model


def test_a_tie_goes_to_the_smaller_model_and_a_real_difference_does_not(tmp_path):
    printed, written, _ = summarise(tmp_path, three_seeds("a", "char", 10, 0.010) + three_seeds("b", "bpe-1024", 11, 0.0))
    assert written["choice"]["settings"]["tokenizer"] == "char"                                               # 0.010 behind, 0.018 needed: a tie, so the smaller model
    (tmp_path / "again").mkdir()
    printed, written, _ = summarise(tmp_path / "again", three_seeds("a", "char", 10, 0.030) + three_seeds("b", "bpe-1024", 11, 0.0))
    assert written["choice"]["settings"]["tokenizer"] == "bpe-1024"                                           # 0.030 behind: really worse


def test_a_prefix_gets_a_record_of_its_own_and_mixed_code_gets_a_warning(tmp_path):
    runs = three_seeds("sweep-a", "char", 10, 0.0) + [("other-1", "bpe-1024", 11, 1, 1.9, {})]
    runs[1] = (*runs[1][:5], {"git_commit": "another commit", "best_step": 100})
    printed, written, name = summarise(tmp_path, runs, "sweep-")
    assert name == "results-sweep.json" and len(written["groups"]) == 1 and "pooled_standard_deviation" not in written
    assert "do not share the same git_commit" in printed and "best at the very end" in printed
