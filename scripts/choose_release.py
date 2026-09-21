"""Which of the trained runs is GP-Thee-11M? The rule from docs/BUILD_LOG.md entry 17, applied by a program.

Run:  uv run python scripts/choose_release.py        (a second; writes docs/release.json)

Eleven character runs at 34 passes exist. Choosing among them by hand, after seeing their scores, would be the
last and easiest place in this project to flatter the result: the highest-scoring run of the eleven is a pilot
that part 5 showed was a lucky draw, and releasing it would publish a number that luck produced.

So the rule was written down first, and this script is all of it:

  * ELIGIBLE means made by commit 0f72600 (the commit that decided the tokenizer) with a clean tree, seed 1, 2
    or 3, 32-bit, and every other setting at its default. Three runs qualify, and they are the three the
    published tokenizer comparison used.
  * The released run is the eligible run with the LOWEST best-validation bits per character, as its own
    result.json recorded it during training. Never a re-score. Ties break by smallest seed, then by name.

Both layers of selection are counted here and written into the record, because they will be the first thing a
careful reader asks about: the best of 41 stops within the run, and the best of three runs. Neither of them
touches the test works, which have still never been opened.

Writes docs/release.json, which scripts/final_evaluation.py refuses to run without.
"""

import dataclasses
import hashlib
import json
import statistics as st
from pathlib import Path

from gp_thee.train import RunConfig, git_commit

ROOT = Path(__file__).resolve().parent.parent
COMMIT = "0f726008af749e027022fcc0286ec703c47136fc"   # the commit that decided the tokenizer (entry 16)
SEEDS = (1, 2, 3)


def why_not(config: dict) -> str | None:
    """Why this run cannot be the released one, or None if it can."""
    if config["git_commit"] != COMMIT:
        return f"made by {config['git_commit'][:7]}, not the tokenizer commit {COMMIT[:7]}"
    if config["seed"] not in SEEDS:
        return f"seed {config['seed']}, and the comparison seeds were fixed as 1, 2 and 3"
    if config["sixteen_bit"]:
        return "16-bit, which the pre-registered rule declined to adopt (entry 14)"
    defaults = dataclasses.asdict(RunConfig(name="x"))
    odd = [k for k, v in defaults.items() if k not in ("name", "seed", "device", "passes") and config.get(k) != v]
    return f"not the default settings: {odd}" if odd else None


def main() -> None:
    eligible, excluded = [], []
    for folder in sorted((ROOT / "runs").glob("*/config.json")):
        config = json.loads(folder.read_text())
        result = json.loads((folder.parent / "result.json").read_text()) if (folder.parent / "result.json").exists() else None
        if config["tokenizer"] != "char" or config["steps"] != 9985 or result is None:
            continue
        row = {"run": config["name"], "seed": config["seed"], "commit": config["git_commit"][:7],
               "validation_bpc": result["best"]["validation_bpc"], "best_step": result["best"]["step"]}
        (excluded.append({**row, "excluded_because": why_not(config)}) if why_not(config) else eligible.append(row))
    if not eligible:
        raise SystemExit("no run is eligible: see docs/BUILD_LOG.md entry 17")
    eligible.sort(key=lambda r: (r["validation_bpc"], r["seed"], r["run"]))
    chosen = eligible[0]

    folder = ROOT / "runs" / chosen["run"]
    saved = json.loads((folder / "config.json").read_text())
    scores = [r["validation_bpc"] for r in eligible]
    release = {
        "model": "GP-Thee-11M",
        "run": chosen["run"], "checkpoint": "best.pt", "step": chosen["best_step"],
        "checkpoint_sha256": hashlib.sha256((folder / "best.pt").read_bytes()).hexdigest(),
        "validation_bpc": chosen["validation_bpc"], "parameters": saved["parameters"],
        "tokenizer": saved["tokenizer"], "fingerprints": saved["fingerprints"], "trained_by_commit": saved["git_commit"],
        "chosen_by": "docs/BUILD_LOG.md entry 17: the lowest best-validation score among the runs from the tokenizer commit, seeds 1 to 3, 32-bit, default settings",
        "eligible": eligible, "excluded": excluded,
        "what_the_choice_costs": {
            "the_three_eligible_runs": {"mean": st.mean(scores), "spread": max(scores) - min(scores),
                                        "the_rule_calls_them": "a tie: the sweep's own pooled test needed 0.0130 to call two arms different"},
            "the_released_run_is_below_their_mean_by": st.mean(scores) - chosen["validation_bpc"],
            "note": ("Two layers of selection sit on this validation score and neither sits on the test score: the best of 41 stops "
                     "within the run, and the best of three runs. Simulated with this project's own run-to-run spread, picking the "
                     "best of three flatters a score by about 0.005. The test works chose nothing, so the test score is not flattered."),
        },
        "chosen_at_commit": git_commit(),
    }
    (ROOT / "docs" / "release.json").write_text(json.dumps(release, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'eligible':<24}{'seed':>5}{'commit':>9}{'validation':>12}{'at step':>9}")
    for row in eligible:
        print(f"  {row['run']:<22}{row['seed']:>5}{row['commit']:>9}{row['validation_bpc']:>12.4f}{row['best_step']:>9,}"
              + ("   <- GP-Thee-11M" if row is chosen else ""))
    print(f"\n{len(excluded)} runs are not eligible:")
    for row in sorted(excluded, key=lambda r: r["validation_bpc"]):
        print(f"  {row['run']:<22}{row['validation_bpc']:>9.4f}   {row['excluded_because']}")
    print(f"\nThe three eligible runs are {release['what_the_choice_costs']['the_three_eligible_runs']['spread']:.4f} apart, which this "
          f"project's own test calls a tie.\nThe released run sits {release['what_the_choice_costs']['the_released_run_is_below_their_mean_by']:.4f} "
          "below their mean on validation, and that edge is selection, not quality.\n\nwrote docs/release.json")


if __name__ == "__main__":
    main()
