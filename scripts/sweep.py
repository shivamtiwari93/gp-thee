"""Compare the five tokenizers: three seeds each, every run at its tokenizer's own length, all from one commit.

Run:  uv run python scripts/sweep.py          (about five hours on an M5 Max; plug it in)
      uv run python scripts/sweep.py --dry    (lists the runs and what is left to do)

This script decides nothing. Everything it does was fixed in docs/BUILD_LOG.md (entries 14 and 15) before the
runs: seeds 1, 2 and 3 (seed 0 found the lengths and takes no part); the lengths in docs/run_lengths.json; 40
evaluations per run; 32-bit; every other setting at its default. It only makes the runs, one after another.

The order is seed by seed, not tokenizer by tokenizer. If the machine warms up over the hours, or the sweep is
cut short, every tokenizer has been treated alike.

Two tokenizers borrow a length (in steps) from a neighbour, to save pilot runs: bpe-1536 from bpe-1024, and
bpe-2048 from bpe-4096. That was allowed only if the two lengths found are within a factor of two of each
other. If they are not, all four have to be searched with scripts/find_length.py, and this script says so.
A borrower borrows even if someone has searched its length as well: the rule was written before the runs.

Three things it refuses to do, each learned the hard way:

  * Mix commits. On this GPU an edit to the program re-rolls the score of a run as surely as a new seed does,
    even when the edit changes no arithmetic (entry 14). So every run of the comparison must come from one clean
    commit, and the script checks all of them, not only the ones it made today. COMMIT NOTHING WHILE IT RUNS.
  * Mix lengths. A run that exists already must have the length, tokenizer and seed the plan asks for now.
  * Resume a run that blew up. A run whose loss stops being a number is the one kind of run we discard
    (entry 14). It is kept as evidence, marked DIVERGED, and replaced by the same run with seed + 1000.

Finished runs are skipped and an interrupted run is resumed, so the command can simply be given again.
Afterwards:  uv run python scripts/summarise_runs.py sweep-
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (1, 2, 3)
ARMS = ("char", "bpe-1024", "bpe-1536", "bpe-2048", "bpe-4096")
BORROWS = {"bpe-1536": "bpe-1024", "bpe-2048": "bpe-4096"}


def lengths_for(found: dict) -> dict[str, int]:
    """Every arm's length in steps, from the lengths that were searched. Borrowers borrow unless the two searched lengths are too far apart."""
    missing = [arm for arm in ("char", "bpe-1024", "bpe-4096") if arm not in found]
    if missing:
        raise SystemExit(f"find the lengths first: uv run python scripts/find_length.py --tokenizer {missing[0]}")
    searched = (found["bpe-1024"]["steps"], found["bpe-4096"]["steps"])
    if max(searched) > 2 * min(searched):
        missing = [arm for arm in BORROWS if arm not in found]
        if missing:
            raise SystemExit(f"the lengths found for bpe-1024 and bpe-4096 differ by more than a factor of two: search {' and '.join(missing)} as well")
        return {arm: found[arm]["steps"] for arm in ARMS}
    return {arm: found[BORROWS.get(arm, arm)]["steps"] for arm in ARMS}


def state_of(folder: Path) -> str:
    if (folder / "result.json").exists():
        return "done"
    if (folder / "DIVERGED").exists():
        return "diverged"
    return "interrupted" if (folder / "last.pt").exists() else "to do"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry", action="store_true", help="list the runs and stop")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from gp_thee.data import load_tokens
    from gp_thee.train import git_commit

    steps_for = lengths_for(json.loads((ROOT / "docs" / "run_lengths.json").read_text()))
    now = git_commit()
    started = [folder for folder in sorted((ROOT / "runs").glob("sweep-*")) if state_of(folder) != "to do"]   # EVERY run of the sweep, not only the ones before this one
    commits = {json.loads((folder / "config.json").read_text())["git_commit"] for folder in started}
    print(f"the code is at {now}; runs of this sweep so far were made by: {sorted(commits) or 'none yet'}\n", flush=True)

    for seed in SEEDS:
        for arm in ARMS:
            steps = steps_for[arm]
            passes = steps * 64 * 256 / len(load_tokens(arm, "train"))
            attempt = seed
            while True:                                                   # comes round again only if a run diverges
                name = f"sweep-{arm}-seed-{attempt}"
                folder, state = ROOT / "runs" / name, state_of(ROOT / "runs" / name)
                print(f"{name:<28}{steps:>7,} steps = {passes:5.1f} passes   {state}", flush=True)
                if state != "to do":
                    made = json.loads((folder / "config.json").read_text())
                    if (made["steps"], made["tokenizer"], made["seed"], made["evaluations"], made["sixteen_bit"]) != (steps, arm, attempt, 40, False):
                        sys.exit(f"{name} was made with {made['steps']:,} steps of {made['tokenizer']}, seed {made['seed']}; the plan asks for {steps:,} steps of {arm}, "
                                 f"seed {attempt}. One arm, one length: put docs/run_lengths.json back as it was, or move that run out of runs/.")
                if state == "diverged":
                    attempt += 1000                                       # the discarded run stays where it is; its replacement is the same run with seed + 1000
                    continue
                if args.dry or state == "done":
                    break
                now = git_commit()                                        # asked again before EVERY run: the code can move while the sweep is running
                if now == "unknown" or "uncommitted" in now or commits - {now}:
                    sys.exit(f"the code is at {now!r}, and runs of this sweep were made by {sorted(commits) or 'nothing yet'}. Every run of a comparison must come from ONE clean "
                             "commit.\nTo carry on: `git stash` any changes to src/, then `git checkout <the commit the runs were made by>` (a detached HEAD is fine), "
                             "give this command again, and return to your branch when the sweep is over.")
                command = [sys.executable, str(ROOT / "scripts" / "train.py"), "--name", name]
                command += ["--resume"] if state == "interrupted" else ["--tokenizer", arm, "--seed", str(attempt), "--passes", repr(passes)]
                child = subprocess.run(command, stderr=subprocess.PIPE, text=True)   # what the run prints streams through; its errors are shown below, and read for one word
                sys.stderr.write(child.stderr)
                commits.add(now)
                if child.returncode and "FloatingPointError" in child.stderr:
                    (folder / "DIVERGED").write_text(child.stderr)
                    continue
                if child.returncode:
                    sys.exit(f"{name} stopped with exit status {child.returncode} (see above). Give the command again to resume it.")
                break
    print("" if args.dry else "\nall done. Next: uv run python scripts/summarise_runs.py sweep-")


if __name__ == "__main__":
    main()
