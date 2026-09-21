"""How long should one tokenizer's model train? Find out by the rule that was written down before any of these runs.

Run:  uv run python scripts/find_length.py --tokenizer bpe-1024        (one to two hours on an M5 Max)
      uv run python scripts/find_length.py --tokenizer bpe-1024 --dry   (says what it would run next, and runs nothing)

The rule (docs/BUILD_LOG.md, entry 14, item 2), with seed 0, which takes no part in any comparison:

    Start near 5,000 steps. While doubling the run improves the best validation score by more than 0.01 bits
    per character, double again, up to 20,000 steps. If a run's best moment falls before two thirds of its
    length, also try half. Keep the SHORTEST length within 0.01 of the lowest score.

Two readings had to be settled to turn that into a program, and both were written into the build log
(entry 15) before the first run. "Also try half" applies to every run that is tried, the halves included, down
to 625 steps. And the run of twice the starting length is always made: we cannot know that doubling does not
help without trying it.

Why a rule at all? Because the length of a run is a setting, and a setting chosen by looking at results can be
chosen to flatter. A run that is too short looks finished in its own log (its best moment is its last, and
nothing says how much was left). A run that is too long over-fits, and keeping the best checkpoint only partly
repairs that, because the learning rate is still high when the best moment comes.

Every run is an ordinary training run, started through scripts/train.py in a process of its own, so a pilot is
made exactly the way a real run is. Finished pilots are not repeated, and an interrupted one is resumed.

Writes docs/run_lengths.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, CAP, FLOOR, ENOUGH = 5_000, 20_000, 625, 0.01
BATCH, CONTEXT, SEED = 64, 256, 0  # the defaults of RunConfig: the search changes nothing but the length


def next_to_try(tried: dict[int, dict]) -> int | None:
    """The next run length the rule asks for, in steps, or None when it has all it needs.

    `tried` maps a length to {"best": that run's best validation score, "best_step": where it fell}.
    """
    if START not in tried:
        return START
    length = START
    while 2 * length <= CAP:                                   # while doubling improves the best score by enough, double again
        if 2 * length not in tried:
            return 2 * length
        if tried[2 * length]["best"] >= tried[length]["best"] - ENOUGH:
            break
        length *= 2
    for steps in sorted(tried, reverse=True):                  # a best moment before two thirds of the run: also try half
        if tried[steps]["best_step"] < 2 / 3 * steps and steps // 2 >= FLOOR and steps // 2 not in tried:
            return steps // 2
    return None


def choose(tried: dict[int, dict]) -> int:
    """The shortest length whose best score is within ENOUGH of the lowest."""
    lowest = min(run["best"] for run in tried.values())
    return min(steps for steps, run in tried.items() if run["best"] <= lowest + ENOUGH)


def pilot(tokenizer: str, steps: int, training_tokens: int) -> dict:
    """Make (or finish, or just read) the pilot run of this length, and say how it went."""
    name = f"pilot-{tokenizer}-{steps}"
    folder = ROOT / "runs" / name
    if not (folder / "result.json").exists():
        passes = steps * BATCH * CONTEXT / training_tokens     # RunConfig counts in passes; this many passes IS this many steps
        command = [sys.executable, str(ROOT / "scripts" / "train.py"), "--name", name]
        command += ["--resume"] if (folder / "last.pt").exists() else ["--tokenizer", tokenizer, "--seed", str(SEED), "--passes", repr(passes)]
        subprocess.run(command, check=True)
    result = json.loads((folder / "result.json").read_text())
    if result["steps"] != steps:
        raise ValueError(f"{name} ran for {result['steps']} steps, not {steps}")
    return {"best": result["best"]["validation_bpc"], "best_step": result["best"]["step"], "final": result["final_validation_bpc"],
            "minutes": result["minutes"], "run": name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dry", action="store_true", help="read the pilots that exist, say what would run next, and stop")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from gp_thee.data import load_tokens
    training_tokens = len(load_tokens(args.tokenizer, "train"))

    tried = {}
    for folder in sorted((ROOT / "runs").glob(f"pilot-{args.tokenizer}-*")):   # pilots already made count, whoever made them
        length = folder.name.rsplit("-", 1)[1]
        if (folder / "result.json").exists() and length.isdigit() and json.loads((folder / "result.json").read_text())["steps"] == int(length):
            tried[int(length)] = pilot(args.tokenizer, int(length), training_tokens)
    while (pending := next_to_try(tried)) is not None and not args.dry:
        tried[pending] = pilot(args.tokenizer, pending, training_tokens)

    print(f"\n{args.tokenizer}: one pass is {training_tokens / (BATCH * CONTEXT):.1f} steps")
    print(f"{'steps':>8}{'passes':>8}{'best':>9}{'at step':>9}{'that is':>9}{'final':>9}{'minutes':>9}")
    for steps in sorted(tried):
        run = tried[steps]
        print(f"{steps:>8,}{steps * BATCH * CONTEXT / training_tokens:>8.1f}{run['best']:>9.4f}{run['best_step']:>9,}{run['best_step'] / steps:>8.0%} {run['final']:>9.4f}{run['minutes']:>9.1f}")
    if pending is not None:
        print(f"\nnext: {pending:,} steps ({pending * BATCH * CONTEXT / training_tokens:.1f} passes)")
        return
    chosen = choose(tried)
    print(f"\nthe rule picks {chosen:,} steps = {chosen * BATCH * CONTEXT / training_tokens:.1f} passes")
    record = ROOT / "docs" / "run_lengths.json"
    lengths = json.loads(record.read_text()) if record.exists() else {}
    lengths[args.tokenizer] = {"steps": chosen, "passes": chosen * BATCH * CONTEXT / training_tokens, "found_by": "scripts/find_length.py, seed 0",
                               "tried": {str(steps): {k: run[k] for k in ("best", "best_step", "final", "minutes", "run")} for steps, run in sorted(tried.items())}}
    record.write_text(json.dumps(lengths, indent=2) + "\n")
    print("wrote docs/run_lengths.json")


if __name__ == "__main__":
    main()
