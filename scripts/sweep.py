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
other; otherwise this script refuses, and all four have to be searched with scripts/find_length.py.

One more rule, learned the hard way (entry 14): on this GPU an edit to the program re-rolls the score of a run
as surely as a new seed does, even when the edit changes no arithmetic. So every run of a comparison must come
from the same commit. The script checks that before it starts a run, and stops if the code has changed.

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

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--dry", action="store_true", help="list the runs and stop")
args = parser.parse_args()
sys.path.insert(0, str(ROOT / "src"))
from gp_thee.data import load_tokens  # noqa: E402
from gp_thee.train import git_commit  # noqa: E402

found = json.loads((ROOT / "docs" / "run_lengths.json").read_text())
if not all(arm in found for arm in ("char", "bpe-1024", "bpe-4096")):
    sys.exit("find the lengths first: uv run python scripts/find_length.py --tokenizer bpe-1024   (and bpe-4096)")
if max(found["bpe-1024"]["steps"], found["bpe-4096"]["steps"]) > 2 * min(found["bpe-1024"]["steps"], found["bpe-4096"]["steps"]):
    if not all(arm in found for arm in ARMS):
        sys.exit("the lengths found for bpe-1024 and bpe-4096 differ by more than a factor of two: search bpe-1536 and bpe-2048 as well")
steps_for = {arm: found[arm]["steps"] if arm in found else found[BORROWS[arm]]["steps"] for arm in ARMS}

runs = [(f"sweep-{arm}-seed-{seed}", arm, seed) for seed in SEEDS for arm in ARMS]
commits = set()
for name, arm, seed in runs:
    folder, steps = ROOT / "runs" / name, steps_for[arm]
    state = "done" if (folder / "result.json").exists() else "interrupted" if (folder / "last.pt").exists() else "to do"
    passes = steps * 64 * 256 / len(load_tokens(arm, "train"))
    print(f"{name:<26}{steps:>7,} steps = {passes:5.1f} passes   {state}")
    if (folder / "config.json").exists():
        commits.add(json.loads((folder / "config.json").read_text())["git_commit"])
    if args.dry or state == "done":
        continue
    now = git_commit()
    if "uncommitted" in now or (commits and commits != {now}):
        sys.exit(f"the code is at {now!r}, but the runs so far were made by {sorted(commits)}. Every run of a comparison must come from one clean commit.")
    command = [sys.executable, str(ROOT / "scripts" / "train.py"), "--name", name]
    command += ["--resume"] if state == "interrupted" else ["--tokenizer", arm, "--seed", str(seed), "--passes", repr(passes)]
    subprocess.run(command, check=True)
    commits.add(now)
print("\nall done. Next: uv run python scripts/summarise_runs.py sweep-" if not args.dry else "")
