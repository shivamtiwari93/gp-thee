"""The rule that chooses how long a tokenizer's model trains was written down in advance. These tests pin the program to the rule.

First the rule itself (two pure functions), then the part that makes and reads the pilot runs. Nothing trains here: the script's
ROOT points at an empty folder, and where it would start scripts/train.py stands a function that records the command and leaves
behind the files a real run would leave. Pilots "made earlier" are folders made by hand.
"""

import dataclasses
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from gp_thee.data import load_tokens
from gp_thee.train import RunConfig

spec = importlib.util.spec_from_file_location("find_length", Path(__file__).resolve().parent.parent / "scripts" / "find_length.py")
find_length = importlib.util.module_from_spec(spec)
spec.loader.exec_module(find_length)
next_to_try, choose = find_length.next_to_try, find_length.choose


def run(best, at, of):
    return {"best": best, "best_step": round(at * of)}


def test_it_starts_near_5000_steps_and_always_tries_double():
    assert next_to_try({}) == 5_000
    assert next_to_try({5_000: run(1.80, 1.0, 5_000)}) == 10_000
    assert next_to_try({5_000: run(1.80, 0.3, 5_000)}) == 10_000            # even a run that over-fitted early: we cannot know without trying


def test_it_doubles_while_doubling_gains_more_than_a_hundredth_and_stops_at_20000():
    tried = {5_000: run(1.800, 1.0, 5_000), 10_000: run(1.780, 0.98, 10_000)}
    assert next_to_try(tried) == 20_000                                     # 0.020 gained: double again
    tried[20_000] = run(1.775, 0.97, 20_000)
    assert next_to_try(tried) is None                                       # 0.005 gained: stop. Nothing over-fitted early, so no halves either
    assert choose(tried) == 10_000                                          # 20,000 is the lowest, 10,000 is within 0.01 of it, and shorter
    tried[20_000] = run(1.700, 0.97, 20_000)
    assert next_to_try(tried) is None and choose(tried) == 20_000           # still gaining, but 20,000 is the cap
    assert next_to_try({5_000: run(1.800, 1.0, 5_000), 10_000: run(1.7901, 1.0, 10_000)}) is None   # 0.0099 is not more than 0.01


def test_a_run_whose_best_moment_comes_early_sends_it_to_try_half_and_the_half_too():
    tried = {5_000: run(1.900, 0.30, 5_000), 10_000: run(1.930, 0.14, 10_000)}
    assert next_to_try(tried) == 2_500                                      # 10,000's half is 5,000, already made; 5,000's half is next
    tried[2_500] = run(1.880, 0.52, 2_500)
    assert next_to_try(tried) == 1_250                                      # 52% is before two thirds
    tried[1_250] = run(1.885, 0.96, 1_250)
    assert next_to_try(tried) is None                                       # its best moment is late: nothing more to try
    assert choose(tried) == 1_250                                           # the lowest is 2,500's 1.880; 1,250 is within 0.01 and shorter
    tried[1_250] = run(1.895, 0.96, 1_250)
    assert choose(tried) == 2_500                                           # 0.015 behind is not within 0.01


def test_two_thirds_is_the_line_and_625_steps_is_the_floor():
    assert next_to_try({5_000: run(1.9, 0.66, 5_000), 10_000: run(1.95, 0.9, 10_000)}) == 2_500
    assert next_to_try({5_000: run(1.9, 0.67, 5_000), 10_000: run(1.95, 0.9, 10_000)}) is None
    tried = {steps: run(1.9, 0.1, steps) for steps in (10_000, 5_000, 2_500, 1_250)}
    assert next_to_try(tried) == 625
    tried[625] = run(1.9, 0.1, 625)
    assert next_to_try(tried) is None                                       # 312 steps would be under the floor


def test_exactly_a_hundredth_is_not_more_than_a_hundredth_and_is_within_a_hundredth():
    assert 1.80 - 0.01 == 1.79 and 1.78 + 0.01 == 1.79                      # these floats are exact, so the boundary itself can be tested
    assert next_to_try({5_000: run(1.80, 1.0, 5_000), 10_000: run(1.79, 1.0, 10_000)}) is None   # gained exactly 0.01: that is not MORE than 0.01
    assert choose({5_000: run(1.79, 1.0, 5_000), 10_000: run(1.78, 1.0, 10_000)}) == 5_000        # exactly 0.01 above the lowest: that is within 0.01


def test_every_early_run_is_halved_not_only_the_length_the_rule_would_pick():
    tried = {5_000: run(1.900, 0.30, 5_000), 10_000: run(1.850, 0.90, 10_000), 20_000: run(1.845, 0.90, 20_000)}
    assert choose(tried) == 10_000 and next_to_try(tried) == 2_500          # 10,000 would be chosen and its best came late, but 5,000's came early: its half is still owed


# ------------------------------------------------------------------------------------------ making and reading the pilots
GRID = (625, 1_250, 2_500, 5_000, 10_000, 20_000)
TOKENS = {name: len(load_tokens(name, "train")) for name in ("char", "bpe-1024", "bpe-1536", "bpe-2048", "bpe-4096")}
NOT_A_NUMBER = "Traceback (most recent call last):\n  ...\nFloatingPointError: the model's predictions are no longer numbers: the run has diverged\n"


def result_of(steps, best, at):
    return json.dumps({"best": {"validation_bpc": best, "step": round(at * steps)}, "final_validation_bpc": best + 0.004, "steps": steps, "minutes": 1.0})


def pilot_by_hand(root, name, steps, state="done", /, best=1.9, at=1.0, **changed):
    """A pilot's folder as an earlier search, or somebody's hand, would have left it. `at` is where its best moment falls, as a share of the run."""
    folder = root / "runs" / name
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({**dataclasses.asdict(RunConfig(name=name, tokenizer="bpe-1024", seed=0)), "steps": steps, **changed}))
    for file in {"done": ("last.pt",), "interrupted": ("last.pt",), "diverged": ("last.pt", "DIVERGED"), "to do": ()}[state]:
        (folder / file).write_bytes(b"")
    if state == "done":
        (folder / "result.json").write_text(result_of(changed.get("steps", steps), best, at))
    return folder


@pytest.fixture
def fake(tmp_path, monkeypatch, capsys):
    """scripts/find_length.py in an empty project folder. `scores` says how a pilot of each length will turn out: (best, where its best moment falls)."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "runs").mkdir()
    world = types.SimpleNamespace(root=tmp_path, calls=[], scores={}, diverges=set(), fails=set())

    def train(command, stderr=None, text=None):
        assert command[:2] == [sys.executable, str(tmp_path / "scripts" / "train.py")]
        assert stderr == subprocess.PIPE and text                           # the pilot's errors are read for one word, so they must come back as text
        flags = command[2:]
        world.calls.append(flags)
        assert len(world.calls) < 40                                        # a search is a handful of pilots: by forty it is going round in circles
        folder = tmp_path / "runs" / flags[1]
        if "--resume" not in flags:                                         # a resumed run takes every setting from its own config.json
            _, name, _, tokenizer, _, seed, _, passes = flags
            folder.mkdir(exist_ok=True)
            steps = RunConfig(name=name, passes=float(passes)).steps(TOKENS[tokenizer])                       # as the run itself will count them
            (folder / "config.json").write_text(json.dumps({**dataclasses.asdict(RunConfig(name=name, tokenizer=tokenizer, seed=int(seed))), "steps": steps}))
        (folder / "last.pt").write_bytes(b"")
        if flags[1] in world.diverges | world.fails:
            return subprocess.CompletedProcess(command, 1, stderr=NOT_A_NUMBER if flags[1] in world.diverges else "KeyboardInterrupt\n")
        steps = json.loads((folder / "config.json").read_text())["steps"]
        (folder / "result.json").write_text(result_of(steps, *world.scores.get(steps, (1.9, 1.0))))
        return subprocess.CompletedProcess(command, 0, stderr="")

    def search(*flags):
        monkeypatch.setattr(sys, "argv", ["find_length.py", *flags])
        before, stopped = len(world.calls), None
        try:
            find_length.main()
        except SystemExit as stop:
            stopped = str(stop.code)
        return types.SimpleNamespace(stopped=stopped, calls=world.calls[before:], printed=capsys.readouterr().out)

    monkeypatch.setattr(find_length, "ROOT", tmp_path)
    monkeypatch.setattr(find_length, "subprocess", types.SimpleNamespace(run=train, PIPE=subprocess.PIPE))
    monkeypatch.setattr(sys, "path", list(sys.path))                        # main() puts ROOT/src on the path; this takes it off again
    world.search = search
    return world


def test_a_pilot_is_an_ordinary_run_with_seed_0_that_changes_nothing_but_the_length_and_it_is_read_at_its_best_moment(fake):
    fake.scores[5_000] = (1.80, 0.6)
    got = find_length.pilot("bpe-1024", 5_000, TOKENS["bpe-1024"])
    passes = 5_000 * 64 * 256 / TOKENS["bpe-1024"]
    # seed 0 takes no part in any comparison, and nothing else is on the command line: every other setting stays at its default
    assert fake.calls == [["--name", "pilot-bpe-1024-5000", "--tokenizer", "bpe-1024", "--seed", "0", "--passes", repr(passes)]]
    assert got == {"best": 1.80, "best_step": 3_000, "final": 1.804, "minutes": 1.0, "run": "pilot-bpe-1024-5000"}        # the BEST moment, not the last
    assert find_length.pilot("bpe-1024", 5_000, TOKENS["bpe-1024"]) == got and len(fake.calls) == 1                       # a finished pilot is read, not made again
    assert find_length.pilot("bpe-1024", 10_000, TOKENS["bpe-1024"], make=False) is None and len(fake.calls) == 1         # and with `make` off nothing is ever started


@pytest.mark.parametrize("tokenizer", sorted(TOKENS))
def test_the_passes_on_the_command_line_are_exactly_the_steps_asked_for(fake, tokenizer):
    for steps in GRID:
        assert find_length.pilot(tokenizer, steps, TOKENS[tokenizer])["run"] == f"pilot-{tokenizer}-{steps}"   # it stops if the pilot "ran" for any other number of steps
    assert [(call[1], call[7]) for call in fake.calls] == [(f"pilot-{tokenizer}-{steps}", repr(steps * 64 * 256 / TOKENS[tokenizer])) for steps in GRID]


def test_an_interrupted_pilot_is_resumed_with_only_its_name_and_one_killed_before_its_first_checkpoint_starts_afresh(fake):
    pilot_by_hand(fake.root, "pilot-bpe-1024-5000", 5_000, "to do")
    pilot_by_hand(fake.root, "pilot-bpe-1024-10000", 10_000, "interrupted")
    assert find_length.pilot("bpe-1024", 5_000, TOKENS["bpe-1024"])["run"] == "pilot-bpe-1024-5000"
    assert find_length.pilot("bpe-1024", 10_000, TOKENS["bpe-1024"])["run"] == "pilot-bpe-1024-10000"
    assert fake.calls[0][2:7] == ["--tokenizer", "bpe-1024", "--seed", "0", "--passes"] and fake.calls[1] == ["--name", "pilot-bpe-1024-10000", "--resume"]


@pytest.mark.parametrize("made_with", [{"steps": 4_999}, {"tokenizer": "bpe-1536"}, {"seed": 1}, {"evaluations": 20}, {"sixteen_bit": True}])
def test_a_pilot_this_search_could_not_have_made_is_refused(fake, made_with):
    # A pilot made by hand with another seed or other settings, after a look at the curves, would be a way of steering the rule.
    pilot_by_hand(fake.root, "pilot-bpe-1024-5000", 5_000, best=1.70, at=0.9, **made_with)
    with pytest.raises(SystemExit, match="pilot-bpe-1024-5000 is not a pilot this search could have made"):
        find_length.pilot("bpe-1024", 5_000, TOKENS["bpe-1024"])
    refused = fake.search("--tokenizer", "bpe-1024", "--dry")
    assert "not a pilot this search could have made" in refused.stopped and fake.calls == []


def test_a_pilot_whose_loss_stops_being_a_number_is_kept_marked_and_made_again_with_seed_1000(fake):
    fake.diverges = {"pilot-bpe-1024-5000"}
    got = find_length.pilot("bpe-1024", 5_000, TOKENS["bpe-1024"])
    passes = repr(5_000 * 64 * 256 / TOKENS["bpe-1024"])
    assert fake.calls == [["--name", "pilot-bpe-1024-5000", "--tokenizer", "bpe-1024", "--seed", "0", "--passes", passes],
                          ["--name", "pilot-bpe-1024-5000-seed-1000", "--tokenizer", "bpe-1024", "--seed", "1000", "--passes", passes]]   # the same run, with seed 1000
    discarded = fake.root / "runs" / "pilot-bpe-1024-5000"
    assert got["run"] == "pilot-bpe-1024-5000-seed-1000" and (discarded / "DIVERGED").read_text() == NOT_A_NUMBER and (discarded / "last.pt").exists()   # kept as evidence
    assert find_length.pilot("bpe-1024", 5_000, TOKENS["bpe-1024"]) == got and len(fake.calls) == 2                       # never resumed, though it has a checkpoint
    fake.diverges = {"pilot-bpe-1024-10000", "pilot-bpe-1024-10000-seed-1000"}
    assert find_length.pilot("bpe-1024", 10_000, TOKENS["bpe-1024"])["run"] == "pilot-bpe-1024-10000-seed-2000"           # the replacement blew up too
    pilot_by_hand(fake.root, "pilot-bpe-1024-20000", 20_000, "diverged")
    assert find_length.pilot("bpe-1024", 20_000, TOKENS["bpe-1024"], make=False) is None and len(fake.calls) == 5         # --dry: the replacement is not there yet, nor made


def test_a_pilot_that_stops_for_any_other_reason_stops_the_search_and_is_not_discarded(fake):
    fake.fails = {"pilot-bpe-1024-5000"}
    stopped = fake.search("--tokenizer", "bpe-1024")
    assert "pilot-bpe-1024-5000 stopped with exit status 1" in stopped.stopped and "again to resume" in stopped.stopped and len(stopped.calls) == 1
    assert not list(fake.root.glob("runs/*/DIVERGED")) and not (fake.root / "docs" / "run_lengths.json").exists()
    fake.fails = set()
    assert fake.search("--tokenizer", "bpe-1024").calls[0] == ["--name", "pilot-bpe-1024-5000", "--resume"]


# ------------------------------------------------------------------------------------------ the whole search
def test_the_character_length_is_not_searched_again(fake):
    record = fake.root / "docs" / "run_lengths.json"
    record.write_text(json.dumps({"char": {"steps": 9_985, "found_by": "three pilots, by hand"}}))
    refused = fake.search("--tokenizer", "char")                                 # it used to make pilot-char-5000 and write a length from the grid over the 9,985
    assert "entry 14" in refused.stopped and refused.calls == [] and not list((fake.root / "runs").iterdir())
    assert json.loads(record.read_text()) == {"char": {"steps": 9_985, "found_by": "three pilots, by hand"}}


def test_dry_reads_the_pilots_that_exist_says_what_would_run_next_and_writes_nothing(fake):
    record = fake.root / "docs" / "run_lengths.json"
    record.write_text(json.dumps({"char": {"steps": 9_985}, "bpe-1024": {"steps": 1, "stale": True}}))
    before = record.read_bytes()
    nothing_yet = fake.search("--tokenizer", "bpe-1024", "--dry")
    assert nothing_yet.stopped is None and "next: 5,000 steps (42.3 passes)" in nothing_yet.printed and "the rule picks" not in nothing_yet.printed
    pilot_by_hand(fake.root, "pilot-bpe-1024-5000", 5_000, best=1.800, at=0.9)
    unfinished = pilot_by_hand(fake.root, "pilot-bpe-1024-10000", 10_000, "interrupted")
    half_way = fake.search("--tokenizer", "bpe-1024", "--dry")
    assert "next: 10,000 steps" in half_way.printed and "the rule picks" not in half_way.printed                # an interrupted pilot is not resumed by --dry either
    (unfinished / "result.json").write_text(result_of(10_000, 1.795, 0.9))
    complete = fake.search("--tokenizer", "bpe-1024", "--dry")
    assert "the rule picks 5,000 steps = 42.3 passes" in complete.printed and "wrote" not in complete.printed
    assert fake.calls == [] and record.read_bytes() == before                    # even a complete search is only read
    assert [path.name for path in (fake.root / "docs").iterdir()] == ["run_lengths.json"]


def test_a_pilot_made_by_hand_at_a_length_the_rule_never_asked_for_is_ignored(fake):
    # An auditor's off-grid 7,500-step pilot became "the rule's pick". Only lengths the rule itself asks for are ever read.
    pilot_by_hand(fake.root, "pilot-bpe-1024-5000", 5_000, best=1.800, at=1.0)
    pilot_by_hand(fake.root, "pilot-bpe-1024-10000", 10_000, best=1.795, at=0.9)
    pilot_by_hand(fake.root, "pilot-bpe-1024-7500", 7_500, best=1.700, at=0.9)                  # off the grid, and far the best
    pilot_by_hand(fake.root, "pilot-bpe-1024-2500", 2_500, best=1.600, at=0.9)                  # on the grid, but no run's best moment came early, so no half is owed
    pilot_by_hand(fake.root, "pilot-bpe-1024-20000", 20_000, best=1.600, at=0.9)                # and doubling to 10,000 gained only 0.005, so 20,000 is not owed either
    done = fake.search("--tokenizer", "bpe-1024")
    written = json.loads((fake.root / "docs" / "run_lengths.json").read_text())["bpe-1024"]
    assert done.stopped is None and done.calls == [] and written["steps"] == 5_000 and sorted(written["tried"]) == ["10000", "5000"]
    assert "the rule picks 5,000 steps" in done.printed and "possibly under-trained" in done.printed   # its best moment was its last step: said, and the choice stands


def test_the_whole_search_makes_the_pilots_the_rule_asks_for_writes_its_own_key_and_leaves_the_others(fake):
    record = fake.root / "docs" / "run_lengths.json"
    record.write_text(json.dumps({"char": {"steps": 9_985}, "bpe-1024": {"steps": 1, "stale": True}}))
    fake.scores.update({5_000: (1.800, 0.5), 10_000: (1.780, 0.9), 20_000: (1.775, 0.9), 2_500: (1.850, 1.0)})
    fake.diverges = {"pilot-bpe-1024-20000"}
    done = fake.search("--tokenizer", "bpe-1024")
    assert [call[1] for call in done.calls] == ["pilot-bpe-1024-5000", "pilot-bpe-1024-10000", "pilot-bpe-1024-20000", "pilot-bpe-1024-20000-seed-1000", "pilot-bpe-1024-2500"]
    assert all(call[0::2] == ["--name", "--tokenizer", "--seed", "--passes"] for call in done.calls) and "wrote docs/run_lengths.json" in done.printed
    written = json.loads(record.read_text())
    assert written["char"] == {"steps": 9_985} and sorted(written) == ["bpe-1024", "char"]                       # the other tokenizers' lengths are kept
    assert written["bpe-1024"]["steps"] == 10_000 and written["bpe-1024"]["passes"] == 10_000 * 64 * 256 / TOKENS["bpe-1024"] and "stale" not in written["bpe-1024"]
    assert written["bpe-1024"]["tried"]["5000"] == {"best": 1.800, "best_step": 2_500, "final": 1.804, "minutes": 1.0, "run": "pilot-bpe-1024-5000"}
    assert sorted(written["bpe-1024"]["tried"], key=int) == ["2500", "5000", "10000", "20000"]
    assert written["bpe-1024"]["tried"]["20000"]["run"] == "pilot-bpe-1024-20000-seed-1000"                      # the record names the run that counted
    assert fake.search("--tokenizer", "bpe-1024").calls == [] and json.loads(record.read_text()) == written      # given again: nothing is made again, the record is the same
