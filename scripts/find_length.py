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

Only the lengths the rule itself asks for can count (5,000 doubled or halved), and only pilots made with seed 0
and the default settings. A pilot of some other length, made by hand after a look at the curves, would be a way
of steering the rule, so it is ignored. A pilot whose loss stops being a number is kept, marked DIVERGED, and made
again with seed 1000 (the one ground for discarding a run: entry 14).

Writes docs/run_lengths.json, once the search is complete. With --dry it writes nothing.
"""

import argparse
import dataclasses
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


def pilot(tokenizer: str, steps: int, training_tokens: int, make: bool = True) -> dict | None:
    """Make (or finish, or just read) the pilot run of this length, and say how it went. None if it does not exist and `make` is off."""
    seed = SEED
    while True:                                                    # comes round again only if a pilot diverges
        name = f"pilot-{tokenizer}-{steps}" + (f"-seed-{seed}" if seed != SEED else "")
        folder = ROOT / "runs" / name
        if (folder / "DIVERGED").exists():
            seed += 1000
            continue
        if not (folder / "result.json").exists():
            if not make:
                return None
            passes = steps * BATCH * CONTEXT / training_tokens     # RunConfig counts in passes; this many passes IS this many steps
            command = [sys.executable, str(ROOT / "scripts" / "train.py"), "--name", name]
            command += ["--resume"] if (folder / "last.pt").exists() else ["--tokenizer", tokenizer, "--seed", str(seed), "--passes", repr(passes)]
            child = subprocess.run(command, stderr=subprocess.PIPE, text=True)
            sys.stderr.write(child.stderr)
            if child.returncode and "FloatingPointError" in child.stderr:
                (folder / "DIVERGED").write_text(child.stderr)
                continue
            if child.returncode:
                sys.exit(f"{name} stopped with exit status {child.returncode} (see above). Give the command again to resume it.")
        result, made = json.loads((folder / "result.json").read_text()), json.loads((folder / "config.json").read_text())
        from gp_thee.train import RunConfig                        # every setting but these five must be the default: the search changes nothing but the length
        defaults = {k: v for k, v in dataclasses.asdict(RunConfig(name=name)).items() if k not in ("name", "tokenizer", "seed", "passes", "device")}
        if (result["steps"], made["tokenizer"], made["seed"]) != (steps, tokenizer, seed) or any(made.get(k) != v for k, v in defaults.items()):
            sys.exit(f"{name} is not a pilot this search could have made (its length, tokenizer, seed or settings differ). Move it out of runs/.")
        return {"best": result["best"]["validation_bpc"], "best_step": result["best"]["step"], "final": result["final_validation_bpc"],
                "minutes": result["minutes"], "run": name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dry", action="store_true", help="read the pilots that exist, say what would run next, and stop")
    args = parser.parse_args()
    if args.tokenizer == "char":
        sys.exit("the character length was found before this script existed (docs/BUILD_LOG.md, entry 14) and is recorded in docs/run_lengths.json by hand")
    sys.path.insert(0, str(ROOT / "src"))
    from gp_thee.data import load_tokens
    training_tokens = len(load_tokens(args.tokenizer, "train"))

    tried = {}
    while (pending := next_to_try(tried)) is not None:             # walk the rule from the start: only lengths it asks for are ever read
        done = pilot(args.tokenizer, pending, training_tokens, make=not args.dry)
        if done is None:
            break
        tried[pending] = done

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
    if tried[chosen]["best_step"] == chosen:
        print("  (that run's best moment was its last step. The rule's choice stands; the write-up will call this arm possibly under-trained.)")
    if args.dry:
        return
    record = ROOT / "docs" / "run_lengths.json"
    lengths = json.loads(record.read_text()) if record.exists() else {}
    lengths[args.tokenizer] = {"steps": chosen, "passes": chosen * BATCH * CONTEXT / training_tokens, "found_by": "scripts/find_length.py, seed 0",
                               "tried": {str(steps): {k: run[k] for k in ("best", "best_step", "final", "minutes", "run")} for steps, run in sorted(tried.items())}}
    record.write_text(json.dumps(lengths, indent=2) + "\n")
    print("wrote docs/run_lengths.json")


if __name__ == "__main__":
    main()
