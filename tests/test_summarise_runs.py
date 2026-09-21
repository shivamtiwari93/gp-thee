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
        (folder / "config.json").write_text(json.dumps({"name": name, "tokenizer": tokenizer, "seed": seed, "steps": 100, "parameters": parameters, **extra.get("settings", {}),
                                                        "git_commit": extra.get("git_commit", "x"), "torch": "2.14.0", "fingerprints": {"tokenizer": tokenizer}}))
        if not extra.get("unfinished"):
            (folder / "result.json").write_text(json.dumps({"best": {"validation_bpc": best, "step": extra.get("best_step", 50), "passes": 1.0, "seen_bpc": 1.0},
                                                            "final_validation_bpc": best + 0.1, "minutes": 1.0}))
        if extra.get("diverged"):
            (folder / "DIVERGED").write_text("FloatingPointError: the model's predictions are no longer numbers")
        if "everything_else" in extra:                                                                        # what scripts/evaluate.py leaves beside a run's best checkpoint
            (folder / "evaluation-best.json").write_text(json.dumps({"everything_else": {"bits_per_character": extra["everything_else"]}, "speaker_labels": {"bits_per_character": 2.2}}))
    printed = subprocess.run([sys.executable, str(tmp_path / "scripts" / "summarise_runs.py"), *arguments], capture_output=True, text=True, check=True).stdout
    written = sorted((tmp_path / "docs").glob("results*.json"))
    return printed, json.loads(written[-1].read_text()), written[-1].name


def three_seeds(name, tokenizer, parameters, shift):
    return [(f"{name}-{seed}", tokenizer, parameters, seed, best + shift, {}) for seed, best in ((1, 1.773), (2, 1.765), (3, 1.781))]


def test_repeats_are_averaged_the_spread_is_pooled_and_the_rule_is_applied(tmp_path):
    runs = [("a-1", "char", 10, 1, 1.770, {}), ("a-1-again", "char", 10, 1, 1.776, {}), ("a-2", "char", 10, 2, 1.765, {}), ("a-3", "char", 10, 3, 1.781, {})]  # seed 1 twice: 1.773 once averaged
    runs += three_seeds("b", "bpe-1024", 11, 0.014) + three_seeds("c", "bpe-2048", 12, 0.020)
    printed, written, name = summarise(tmp_path, runs)
    a, b, c = written["groups"]
    assert name == "results.json" and (a["seeds"], b["seeds"], c["seeds"]) == (3, 3, 3)
    assert a["mean_best_validation_bpc"] == pytest.approx(1.773) and a["same_seed_differences"] == {"1": pytest.approx(0.006)}
    assert [g["standard_deviation_over_seeds"] for g in (a, b, c)] == pytest.approx([0.008] * 3)             # n - 1 in the denominator
    assert written["pooled_standard_deviation"] == pytest.approx(0.008) and written["degrees_of_freedom"] == 6
    needed = 2.447 * 0.008 * math.sqrt(2 / 3)                                                                 # t(0.975, 6)
    assert 0.014 < needed < 0.020
    verdicts = [line.rsplit("->", 1)[1].strip() for line in printed.splitlines() if "against" in line]
    assert verdicts == ["a tie", "DIFFERENT", "a tie"] and printed.count(f"give or take {needed:.4f}") == 3
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


# ------------------------------------------------------------------------------------------ the sweep itself: five arms of three seeds
PARAMETERS = {"char": 10_757_760, "bpe-1024": 11_113_344, "bpe-1536": 11_309_952, "bpe-2048": 11_506_560, "bpe-4096": 12_292_992}
PASSES = {"char": 34.0015567275467, "bpe-1024": 42.30000056799317, "bpe-1536": 45.82006563142701, "bpe-2048": 24.166432141354747, "bpe-4096": 27.1766535560545}
BY_VOCABULARY = list(PARAMETERS)                                              # which is also smallest model first. By folder name the character arm comes LAST


def arm(tokenizer, mean, spread=0.008, seeds=(1, 2, 3), **extra):
    """One arm as scripts/sweep.py names it, with its own long-winded `passes`. Its seeds score mean - spread, mean, mean + spread."""
    return [(f"sweep-{tokenizer}-seed-{seed}", tokenizer, PARAMETERS[tokenizer], seed, mean + offset * spread, {"settings": {"passes": PASSES[tokenizer]}, **extra})
            for seed, offset in zip(seeds, (-1, 0, 1))]


def needed_in(printed, marker="against"):
    return [float(line.split("give or take ")[1].split()[0]) for line in printed.splitlines() if marker in line]


def test_five_arms_of_three_seeds_are_judged_at_1_82_pooled_standard_deviations_not_by_the_two_arms_own_spread(tmp_path):
    # Two quiet arms 0.008 apart and three noisy ones far behind. By their OWN spread (0.001) the quiet two would differ, and the larger
    # vocabulary would win. By the POOLED spread (0.0078, so 0.0141 is needed) they tie, and the tie goes to the characters.
    runs = arm("bpe-1024", 1.752, 0.001) + arm("bpe-1536", 1.800, 0.010) + arm("bpe-2048", 1.800, 0.010) + arm("bpe-4096", 1.800, 0.010) + arm("char", 1.760, 0.001)
    printed, written, name = summarise(tmp_path, runs, "sweep-")
    pooled = math.sqrt((2 * 2 * 0.001 ** 2 + 3 * 2 * 0.010 ** 2) / 10)
    assert name == "results-sweep.json" and written["degrees_of_freedom"] == 10 and written["pooled_standard_deviation"] == pytest.approx(pooled)
    assert round(2.228 * math.sqrt(2 / 3), 2) == 1.82                                                         # t(0.975, 10) x sqrt(1/3 + 1/3): the build log's "1.82"
    assert needed_in(printed) == [pytest.approx(1.82 * pooled, abs=5e-5)] * 10                                # ten pairs, one threshold (printed to four places)
    assert f"{pooled:.4f} (10 degrees of freedom)" in printed
    assert written["choice"]["settings"]["tokenizer"] == "char" and written["choice"]["lowest_mean"] == pytest.approx(1.752)
    verdicts = {line.split(":  ")[0].strip(): line.rsplit("->", 1)[1].strip() for line in printed.splitlines() if "against" in line}
    assert verdicts["passes 34.0, tokenizer char  against  passes 42.3, tokenizer bpe-1024"] == "a tie" and list(verdicts.values()).count("DIFFERENT") == 6


def test_the_winner_is_the_smallest_model_within_reach_of_the_lowest_mean_even_when_the_character_arm_sorts_last_by_name(tmp_path):
    shifts = {"bpe-1024": 0.004, "bpe-1536": 0.002, "bpe-2048": 0.0, "bpe-4096": 0.030, "char": 0.006}
    runs = [run for tokenizer, shift in shifts.items() for run in arm(tokenizer, 1.773 + shift)]
    runs += [("pilot-bpe-1024-5000", "bpe-1024", PARAMETERS["bpe-1024"], 0, 1.5, {}), ("char-34-seed-1", "char", PARAMETERS["char"], 1, 1.5, {})]   # no part of the sweep
    printed, written, name = summarise(tmp_path, runs, "sweep-")
    assert name == "results-sweep.json" and written["degrees_of_freedom"] == 10
    assert [group["settings"]["tokenizer"] for group in written["groups"]] == BY_VOCABULARY                   # smallest model first, whatever the folders are called
    # bpe-2048 has the lowest mean; the characters are 0.006 behind it with 0.0146 needed: a tie, and the tie goes to the smallest model
    assert written["choice"]["lowest_mean"] == pytest.approx(1.773) and written["choice"]["settings"]["tokenizer"] == "char"
    assert "lowest mean: passes 24.2, tokenizer bpe-2048 at 1.7730. The smallest model within reach of it: passes 34.0, tokenizer char at 1.7790" in printed
    assert "passes 42.3, tokenizer bpe-1024   (11,113,344 parameters, 100 steps)" in printed and "42.30000056799317" not in printed   # a float is shown to one decimal
    (tmp_path / "again").mkdir()
    shifts["char"] = 0.016                                                                                    # now out of reach: the next smallest within reach takes it
    printed, written, _ = summarise(tmp_path / "again", [run for tokenizer, shift in shifts.items() for run in arm(tokenizer, 1.773 + shift)], "sweep-")
    assert written["choice"]["settings"]["tokenizer"] == "bpe-1024" and written["choice"]["mean_best_validation_bpc"] == pytest.approx(1.777)


def test_a_sweep_that_is_cut_short_gets_no_verdict(tmp_path):
    runs = [run for tokenizer in BY_VOCABULARY for run in arm(tokenizer, 1.773, seeds=(1, 2) if tokenizer == "bpe-2048" else (1, 2, 3))]
    runs += [("sweep-bpe-2048-seed-3", "bpe-2048", PARAMETERS["bpe-2048"], 3, 9.9, {"unfinished": True, "settings": {"passes": PASSES["bpe-2048"]}})]
    printed, written, _ = summarise(tmp_path, runs, "sweep-")
    assert "PARTIAL" in printed and "no verdict" in printed and [group["seeds"] for group in written["groups"]] == [3, 3, 3, 2, 3]
    assert not {"choice", "pooled_standard_deviation", "degrees_of_freedom", "secondary_everything_else"} & set(written)
    assert not any(word in printed for word in ("against", "within reach", "lowest mean", "DIFFERENT", "a tie"))   # no table of pairs either: nothing to read a winner from
    (tmp_path / "whole").mkdir()
    printed, written, _ = summarise(tmp_path / "whole", [run for tokenizer in BY_VOCABULARY for run in arm(tokenizer, 1.773)], "sweep-")
    assert "PARTIAL" not in printed and written["choice"]["settings"]["tokenizer"] == "char"                   # the verdict comes once, when every arm has all its seeds


def test_a_run_that_blew_up_is_listed_and_its_replacement_with_seed_plus_1000_takes_its_place_in_the_arm(tmp_path):
    runs = arm("char", 1.773) + arm("bpe-1536", 1.790) + arm("bpe-2048", 1.790) + arm("bpe-4096", 1.790) + arm("bpe-1024", 1.790, seeds=(1, 1002, 3))
    runs += [("sweep-bpe-1024-seed-2", "bpe-1024", PARAMETERS["bpe-1024"], 2, 9.9, {"unfinished": True, "diverged": True, "settings": {"passes": PASSES["bpe-1024"]}})]
    printed, written, _ = summarise(tmp_path, runs, "sweep-")
    assert written["discarded_because_the_loss_stopped_being_a_number"] == ["sweep-bpe-1024-seed-2"] and "DISCARDED: sweep-bpe-1024-seed-2 " in printed
    assert [group["seeds"] for group in written["groups"]] == [3] * 5 and written["degrees_of_freedom"] == 10   # seed 1002 stands in for seed 2: the verdict is given
    assert sorted(run["seed"] for run in written["groups"][1]["runs"]) == [1, 3, 1002] and "PARTIAL" not in printed
    (tmp_path / "clean").mkdir()
    printed, written, _ = summarise(tmp_path / "clean", arm("char", 1.773) + arm("bpe-1024", 1.790), "sweep-")
    assert written["discarded_because_the_loss_stopped_being_a_number"] == [] and "DISCARDED" not in printed


def test_runs_that_were_not_all_made_by_one_clean_commit_get_a_warning_even_when_each_arm_agrees_with_itself(tmp_path):
    across = "WARNING: these runs were not all made by one clean commit"
    printed, _, _ = summarise(tmp_path, arm("char", 1.773, git_commit="aaaa") + arm("bpe-1024", 1.790, git_commit="bbbb"), "sweep-")
    assert across in printed and "do not share the same git_commit" not in printed                            # every arm is of one piece; the comparison is not
    (tmp_path / "dirty").mkdir()
    dirty = "aaaa (with uncommitted changes to the code or data)"
    printed, _, _ = summarise(tmp_path / "dirty", arm("char", 1.773, git_commit=dirty) + arm("bpe-1024", 1.790, git_commit=dirty), "sweep-")
    assert across in printed                                                                                  # one commit, but not a clean one
    (tmp_path / "clean").mkdir()
    printed, _, _ = summarise(tmp_path / "clean", arm("char", 1.773, git_commit="aaaa") + arm("bpe-1024", 1.790, git_commit="aaaa"), "sweep-")
    assert "WARNING" not in printed


def test_the_secondary_is_the_same_test_on_everything_but_speaker_labels_and_it_decides_nothing(tmp_path):
    def with_secondary(runs, mean, spread=0.001):
        return [(*run[:5], {**run[5], "everything_else": mean + offset * spread}) for run, offset in zip(runs, (-1, 0, 1))]
    primary = arm("char", 1.773) + arm("bpe-1024", 1.769) + arm("bpe-4096", 1.800)                            # char and bpe-1024 tie, so the characters keep the title
    runs = with_secondary(primary[:3], 1.745) + with_secondary(primary[3:6], 1.730) + with_secondary(primary[6:], 1.760, 0.002)
    printed, written, _ = summarise(tmp_path, runs, "sweep-")
    pooled = math.sqrt((2 * 2 * 0.001 ** 2 + 2 * 0.002 ** 2) / 6)                                             # pooled on THAT statistic, not borrowed from the whole text
    secondary = written["secondary_everything_else"]
    assert secondary["pooled_standard_deviation"] == pytest.approx(pooled) and written["pooled_standard_deviation"] == pytest.approx(0.008)
    assert [mean for _, mean in secondary["means"]] == pytest.approx([1.745, 1.730, 1.760])
    assert [group["mean_everything_else_bpc"] for group in written["groups"]] == pytest.approx([1.745, 1.730, 1.760])
    table = printed[printed.index("SECONDARY, decides nothing"):]
    assert needed_in(table) == [pytest.approx(2.447 * pooled * math.sqrt(2 / 3), abs=5e-5)] * 3
    assert [line.rsplit("->", 1)[1].strip() for line in table.splitlines() if "against" in line] == ["different"] * 3
    assert written["choice"]["settings"]["tokenizer"] == "char"                  # on ordinary text bpe-1024 is clearly better, and the rule still gives characters the title
    (tmp_path / "without").mkdir()
    _, without, _ = summarise(tmp_path / "without", primary, "sweep-")
    assert without["choice"] == written["choice"] and without["pooled_standard_deviation"] == written["pooled_standard_deviation"]
    (tmp_path / "one-missing").mkdir()
    printed, written, _ = summarise(tmp_path / "one-missing", runs[:-1] + [primary[-1]], "sweep-")            # one run not yet scored by evaluate.py: no secondary at all
    assert "SECONDARY" not in printed and "secondary_everything_else" not in written and written["choice"] == without["choice"]


def test_a_run_still_in_progress_means_no_verdict_even_if_every_arm_has_the_same_number_of_seeds(tmp_path):
    # The sweep goes seed by seed. After two whole seeds every arm has two, and while the first run of seed three trains they still all have two.
    # Simulated on equal arms, the choice after two seeds differs from the choice after three about one time in eleven. So: no early verdict.
    runs = three_seeds("sweep-a", "char", 10, 0.0) + three_seeds("sweep-b", "bpe-1024", 11, 0.03)
    printed, written, _ = summarise(tmp_path, runs + [("sweep-b-4", "bpe-1024", 11, 4, 9.9, {"unfinished": True})], "sweep-")
    assert "PARTIAL" in printed and "sweep-b-4" in printed and "choice" not in written
    (tmp_path / "two").mkdir()
    two_seeds = [run for run in runs if run[3] in (1, 2)]
    printed, written, _ = summarise(tmp_path / "two", two_seeds, "sweep-")
    assert "PARTIAL" in printed and "choice" not in written                                                   # two seeds each: equal, and still not enough
