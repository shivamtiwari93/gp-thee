"""scripts/sweep.py makes the fifteen runs of the tokenizer comparison. Every rule it follows was written down before the runs
(docs/BUILD_LOG.md, entries 14 and 15 and the addendum to 15). These tests pin the program to those words.

Nothing trains here and no GPU is touched. The real script is loaded as a module. Its ROOT points at an empty folder, the
commit it sees is one the test controls, and where it would start scripts/train.py stands a function that records the command
and leaves behind the files a real run would leave (config.json, last.pt, result.json). Runs "made earlier" are folders made by hand.
"""

import importlib.util
import dataclasses
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import gp_thee.train as T
from gp_thee.data import load_tokens
from gp_thee.train import RunConfig

spec = importlib.util.spec_from_file_location("sweep", Path(__file__).resolve().parent.parent / "scripts" / "sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

ARMS = ("char", "bpe-1024", "bpe-1536", "bpe-2048", "bpe-4096")
ORDER = [(seed, arm) for seed in (1, 2, 3) for arm in ARMS]                     # seed by seed, and seed 0 takes no part
TOKENS = {arm: len(load_tokens(arm, "train")) for arm in ARMS}
FOUND = {"char": 9_985, "bpe-1024": 5_000, "bpe-4096": 2_500}                   # exactly a factor of two apart: borrowing is still allowed
PLAN = {"char": 9_985, "bpe-1024": 5_000, "bpe-1536": 5_000, "bpe-2048": 2_500, "bpe-4096": 2_500}   # 1536 from 1024, 2048 from 4096, in STEPS
NOT_A_NUMBER = "Traceback (most recent call last):\n  ...\nFloatingPointError: the model's predictions are no longer numbers: the run has diverged\n"
DIRTY = "commit-a (with uncommitted changes to the code or data)"


def found(**steps):
    return {arm: {"steps": length} for arm, length in steps.items()}


def made_by_hand(root, arm, seed, state="done", /, **changed):
    """A run folder as an earlier start of the sweep would have left it. The files are empty: sweep.py reads only config.json."""
    folder = root / "runs" / f"sweep-{arm}-seed-{seed}"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({**dataclasses.asdict(RunConfig(name=folder.name, tokenizer=arm, seed=seed)), "steps": PLAN[arm], "git_commit": "commit-a", **changed}))
    for file in {"done": ("last.pt", "result.json"), "interrupted": ("last.pt",), "diverged": ("last.pt", "DIVERGED"), "to do": ()}[state]:
        (folder / file).write_bytes(b"")
    return folder


@pytest.fixture
def project(tmp_path, monkeypatch, capsys):
    (tmp_path / "docs").mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / "docs" / "run_lengths.json").write_text(json.dumps(found(**FOUND)))
    world = types.SimpleNamespace(root=tmp_path, calls=[], commit="commit-a", diverges=set(), fails=set(), moves_after={})

    def train(command, stderr=None, text=None):
        assert command[:2] == [sys.executable, str(tmp_path / "scripts" / "train.py")]
        assert stderr == subprocess.PIPE and text                              # the run's errors are read for one word, so they must come back as text
        flags = command[2:]
        world.calls.append(flags)
        assert len(world.calls) < 40                                           # a sweep is fifteen runs: by forty it is going round in circles
        folder = tmp_path / "runs" / flags[1]
        if "--resume" not in flags:                                            # a resumed run takes every setting from its own config.json
            _, name, _, arm, _, seed, _, passes = flags
            folder.mkdir(exist_ok=True)
            (folder / "config.json").write_text(json.dumps({**dataclasses.asdict(RunConfig(name=name, tokenizer=arm, seed=int(seed))), "git_commit": world.commit,
                                                            "steps": RunConfig(name=name, passes=float(passes)).steps(TOKENS[arm])}))
        (folder / "last.pt").write_bytes(b"")
        world.commit = world.moves_after.get(flags[1], world.commit)
        if flags[1] in world.diverges | world.fails:
            return subprocess.CompletedProcess(command, 1, stderr=NOT_A_NUMBER if flags[1] in world.diverges else "KeyboardInterrupt\n")
        (folder / "result.json").write_bytes(b"")
        return subprocess.CompletedProcess(command, 0, stderr="")

    def start(*flags):
        monkeypatch.setattr(sys, "argv", ["sweep.py", *flags])
        before, stopped = len(world.calls), None
        try:
            sweep.main()
        except SystemExit as stop:
            stopped = str(stop.code)
        return types.SimpleNamespace(stopped=stopped, calls=world.calls[before:], printed=capsys.readouterr().out)

    monkeypatch.setattr(sweep, "ROOT", tmp_path)
    monkeypatch.setattr(sweep, "subprocess", types.SimpleNamespace(run=train, PIPE=subprocess.PIPE))
    monkeypatch.setattr(T, "git_commit", lambda: world.commit)                 # main() imports it when it is called, so it gets this one
    monkeypatch.setattr(sys, "path", list(sys.path))                           # main() puts ROOT/src on the path; this takes it off again
    world.start = start
    return world


# ------------------------------------------------------------------------------------------ the fifteen runs
def test_fifteen_runs_seed_by_seed_with_seeds_1_2_and_3_each_at_its_own_length_and_nothing_else_on_the_command_line(project):
    first = project.start()
    assert first.stopped is None and "all done" in first.printed
    assert [call[1] for call in first.calls] == [f"sweep-{arm}-seed-{seed}" for seed, arm in ORDER]
    for call, (seed, arm) in zip(first.calls, ORDER):
        passes = PLAN[arm] * 64 * 256 / TOKENS[arm]                            # passes over the TRAINING stream, written out in full
        # nothing else on the command line: 40 stops, 32-bit and every other setting stay at their defaults
        assert call == ["--name", f"sweep-{arm}-seed-{seed}", "--tokenizer", arm, "--seed", str(seed), "--passes", repr(passes)]
        assert RunConfig(name="x", passes=float(call[7])).steps(TOKENS[arm]) == PLAN[arm]   # the flag, turned back into steps the way the run itself will
    again = project.start()
    assert again.stopped is None and again.calls == [] and again.printed.count("   done") == 15   # finished runs are skipped: the command can simply be given again


def test_dry_lists_every_run_with_its_length_and_state_and_starts_nothing(project):
    made_by_hand(project.root, "char", 1)
    made_by_hand(project.root, "bpe-1024", 1, "interrupted")
    made_by_hand(project.root, "bpe-1536", 1, "diverged")
    project.commit = "unknown"                                                 # a listing starts nothing, so it has nothing to refuse
    before = sorted(str(path) for path in project.root.rglob("*"))
    listing = project.start("--dry")
    rows = [line.split() for line in listing.printed.splitlines() if line.startswith("sweep-")]
    assert listing.stopped is None and listing.calls == [] and "all done" not in listing.printed
    assert sorted(str(path) for path in project.root.rglob("*")) == before
    listed = ["char-seed-1", "bpe-1024-seed-1", "bpe-1536-seed-1", "bpe-1536-seed-1001", "bpe-2048-seed-1", "bpe-4096-seed-1"]   # the replacement comes straight after the run it replaces
    assert [row[0] for row in rows[:6]] == [f"sweep-{name}" for name in listed] and len(rows) == 16
    assert [" ".join(row[6:]) for row in rows[:6]] == ["done", "interrupted", "diverged", "to do", "to do", "to do"]
    assert [row[1] for row in rows[:6]] == ["9,985", "5,000", "5,000", "5,000", "2,500", "2,500"]
    assert [row[4] for row in rows[:6]] == [f"{PLAN[arm] * 64 * 256 / TOKENS[arm]:.1f}" for arm in (name.split("-seed-")[0] for name in listed)]
    assert len({row[4] for row in rows[:6]}) == 5                              # a borrowed length is the lender's STEPS, so the borrower's passes are its own


# ------------------------------------------------------------------------------------------ the lengths, and who borrows from whom
def test_a_borrowed_length_is_borrowed_in_steps_and_exactly_a_factor_of_two_still_borrows():
    assert sweep.lengths_for(found(**FOUND)) == PLAN
    the_other_way = {"char": 9_985, "bpe-1024": 2_500, "bpe-1536": 2_500, "bpe-2048": 5_000, "bpe-4096": 5_000}
    assert sweep.lengths_for(found(**{"char": 9_985, "bpe-1024": 2_500, "bpe-4096": 5_000})) == the_other_way
    assert sweep.lengths_for(found(**{"char": 9_985, "bpe-1024": 5_000, "bpe-4096": 5_000})) == {"char": 9_985, **dict.fromkeys(ARMS[1:], 5_000)}


def test_a_length_searched_for_a_borrower_is_ignored_unless_the_lenders_are_more_than_a_factor_of_two_apart():
    # The rule was written before the runs: within a factor of two the borrower BORROWS, even if someone has searched its length "just to see".
    assert sweep.lengths_for(found(**FOUND, **{"bpe-1536": 1_250, "bpe-2048": 20_000})) == PLAN
    far = {"char": 9_985, "bpe-1024": 10_000, "bpe-1536": 5_000, "bpe-2048": 1_250, "bpe-4096": 2_500}
    assert sweep.lengths_for(found(**far)) == far                              # four times apart: now every arm has its own searched length


@pytest.mark.parametrize("lenders", [{"bpe-1024": 5_000, "bpe-4096": 20_000}, {"bpe-1024": 20_000, "bpe-4096": 5_000}, {"bpe-1024": 5_001, "bpe-4096": 2_500}])
def test_lenders_more_than_a_factor_of_two_apart_stop_the_sweep_until_all_four_are_searched(project, lenders):
    for searched, missing in (({}, "bpe-1536 and bpe-2048"), ({"bpe-1536": 10_000}, "bpe-2048"), ({"bpe-2048": 10_000}, "bpe-1536")):
        (project.root / "docs" / "run_lengths.json").write_text(json.dumps(found(char=9_985, **lenders, **searched)))
        refused = project.start()
        assert "factor of two" in refused.stopped and refused.stopped.endswith(f"search {missing} as well") and refused.calls == []


@pytest.mark.parametrize("missing", ["char", "bpe-1024", "bpe-4096"])
def test_no_lengths_no_sweep(project, missing):
    (project.root / "docs" / "run_lengths.json").write_text(json.dumps(found(**{arm: steps for arm, steps in FOUND.items() if arm != missing})))
    refused = project.start()
    assert "find the lengths first" in refused.stopped and refused.stopped.endswith(f"--tokenizer {missing}") and refused.calls == []


# ------------------------------------------------------------------------------------------ runs that exist already
def test_a_run_is_done_diverged_interrupted_or_still_to_do(tmp_path):
    assert sweep.state_of(tmp_path / "runs" / "never-made") == "to do"
    assert sweep.state_of(made_by_hand(tmp_path, "char", 1, "to do")) == "to do"                 # killed before its first checkpoint: there is nothing to resume
    assert sweep.state_of(made_by_hand(tmp_path, "char", 2, "interrupted")) == "interrupted"
    assert sweep.state_of(made_by_hand(tmp_path, "char", 3, "diverged")) == "diverged"           # it has a last.pt as well, from its last good stop, and is NOT "interrupted"
    assert sweep.state_of(made_by_hand(tmp_path, "char", 4, "done")) == "done"


def test_an_interrupted_run_is_resumed_with_nothing_but_its_name_and_one_killed_before_its_first_checkpoint_starts_afresh(project):
    made_by_hand(project.root, "char", 1)
    made_by_hand(project.root, "bpe-1024", 1, "to do", git_commit="an older commit")             # it made nothing, so its commit is no part of the sweep
    made_by_hand(project.root, "bpe-1536", 1, "interrupted")
    rest = project.start()
    assert rest.stopped is None and [call[1] for call in rest.calls] == [f"sweep-{arm}-seed-{seed}" for seed, arm in ORDER[1:]]
    assert rest.calls[0][2:] == ["--tokenizer", "bpe-1024", "--seed", "1", "--passes", repr(5_000 * 64 * 256 / TOKENS["bpe-1024"])]
    assert rest.calls[1] == ["--name", "sweep-bpe-1536-seed-1", "--resume"]                      # the settings come from the run's own config.json


def test_a_run_that_stops_for_any_other_reason_stops_the_sweep_and_is_resumed_when_the_command_is_given_again(project):
    project.fails = {"sweep-bpe-1536-seed-1"}
    cut_short = project.start()
    assert "sweep-bpe-1536-seed-1 stopped with exit status 1" in cut_short.stopped and "again to resume" in cut_short.stopped and len(cut_short.calls) == 3
    assert not list(project.root.glob("runs/*/DIVERGED"))                                        # only a loss that is not a number discards a run
    project.fails = set()
    rest = project.start()
    assert rest.stopped is None and rest.calls[0] == ["--name", "sweep-bpe-1536-seed-1", "--resume"] and len(rest.calls) == 13


@pytest.mark.parametrize("state", ["done", "interrupted", "diverged"])
@pytest.mark.parametrize("made_with", [{"steps": 4_999}, {"tokenizer": "bpe-1536"}, {"seed": 2}])
def test_a_run_made_earlier_must_have_the_length_tokenizer_and_seed_the_plan_asks_for_now(project, made_with, state):
    made_by_hand(project.root, "char", 1)
    made_by_hand(project.root, "bpe-1024", 1, state, **made_with)
    for flags in ((), ("--dry",)):                                                               # the listing would mis-state the run too, so it refuses as well
        refused = project.start(*flags)
        assert "sweep-bpe-1024-seed-1 was made with" in refused.stopped and "One arm, one length" in refused.stopped and refused.calls == []


# ------------------------------------------------------------------------------------------ a loss that is not a number
def test_a_run_whose_loss_stops_being_a_number_is_kept_marked_and_replaced_at_once_by_the_same_run_with_seed_plus_1000(project):
    project.diverges = {"sweep-bpe-1024-seed-2", "sweep-bpe-4096-seed-3", "sweep-bpe-4096-seed-1003"}
    whole = project.start()
    names = [f"sweep-{arm}-seed-{seed}" for seed, arm in ORDER]
    names.insert(names.index("sweep-bpe-1024-seed-2") + 1, "sweep-bpe-1024-seed-1002")
    names += ["sweep-bpe-4096-seed-1003", "sweep-bpe-4096-seed-2003"]                            # the replacement blew up too: seed + 2000
    assert whole.stopped is None and "all done" in whole.printed and [call[1] for call in whole.calls] == names
    replacement = whole.calls[names.index("sweep-bpe-1024-seed-1002")]
    assert replacement[2:] == ["--tokenizer", "bpe-1024", "--seed", "1002", "--passes", repr(5_000 * 64 * 256 / TOKENS["bpe-1024"])]   # the same arm at the same length
    discarded = project.root / "runs" / "sweep-bpe-1024-seed-2"
    assert (discarded / "DIVERGED").read_text() == NOT_A_NUMBER and (discarded / "last.pt").exists() and not (discarded / "result.json").exists()   # kept as evidence
    assert sorted(path.parent.name for path in project.root.glob("runs/*/DIVERGED")) == sorted(project.diverges)
    again = project.start()
    assert again.stopped is None and again.calls == []                                           # nothing is left to do, and the discarded runs are left alone


def test_a_diverged_run_is_never_resumed_although_it_has_a_checkpoint(project):
    # On the GPU a resumed leg is not bit for bit the first one: a resumed diverged run might get through, and be counted as that seed's run.
    made_by_hand(project.root, "char", 1, "diverged")
    made_by_hand(project.root, "char", 1001, "interrupted")
    rest = project.start()
    assert rest.stopped is None and rest.calls[0] == ["--name", "sweep-char-seed-1001", "--resume"] and len(rest.calls) == 15
    assert not any(call[1] == "sweep-char-seed-1" for call in rest.calls)


# ------------------------------------------------------------------------------------------ one clean commit
@pytest.mark.parametrize("now", ["unknown", DIRTY])
def test_a_sweep_is_not_begun_by_uncommitted_code_or_by_code_git_cannot_name(project, now):
    project.commit = now
    refused = project.start()
    assert "ONE clean commit" in refused.stopped and refused.calls == [] and not list((project.root / "runs").iterdir())


@pytest.mark.parametrize("missing", [("char", 1), ("bpe-2048", 2), ("bpe-4096", 3)])
def test_every_run_of_the_sweep_counts_not_only_the_ones_before_the_run_to_be_made(project, missing):
    # An auditor got "all done" with fourteen runs from one commit and the FIRST of the order made again by another.
    for seed, arm in ORDER:
        if (arm, seed) != missing:
            made_by_hand(project.root, arm, seed)
    project.commit = "commit-b"
    refused = project.start()
    assert refused.calls == [] and "'commit-b'" in refused.stopped and "['commit-a']" in refused.stopped
    assert "git checkout" in refused.stopped and "give this command again" in refused.stopped   # the message says how to carry on
    project.commit = "commit-a"                                                                  # back at the commit of the runs so far
    rest = project.start()
    assert rest.stopped is None and [call[1] for call in rest.calls] == [f"sweep-{missing[0]}-seed-{missing[1]}"]


@pytest.mark.parametrize("state", ["done", "interrupted", "diverged"])
def test_a_run_made_earlier_by_uncommitted_code_stops_the_sweep_whatever_became_of_it(project, state):
    made_by_hand(project.root, "bpe-4096", 3, state, git_commit=DIRTY)
    refused = project.start()
    assert "ONE clean commit" in refused.stopped and refused.calls == []


def test_the_code_is_checked_before_each_run_and_the_sweep_stops_if_it_has_moved(project):
    # Entry 15: "the script checks before each run and stops if the code has moved". Here a commit lands while the second run trains.
    project.moves_after = {"sweep-bpe-1024-seed-1": "commit-b"}
    stopped = project.start()
    assert "ONE clean commit" in (stopped.stopped or "") and [call[1] for call in stopped.calls] == ["sweep-char-seed-1", "sweep-bpe-1024-seed-1"]
