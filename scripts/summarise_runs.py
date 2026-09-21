"""Gather finished training runs into one table: the mean over seeds, and how far the seeds disagree.

Run:  uv run python scripts/summarise_runs.py                 (every finished run under runs/)
      uv run python scripts/summarise_runs.py baseline-       (only runs whose name starts with this)

One run is one roll of the dice: the starting weights, the batches and the dropout masks all come from the
seed. Two runs that differ only in their seed end a little apart, and a comparison between two settings
means nothing unless the gap between them is clearly bigger than that. So runs that share every setting
except the seed are grouped, and each group gets a mean and a standard deviation over its seeds.

On this GPU even two runs with the SAME seed end slightly apart (docs/BUILD_LOG.md, entry 13). Where a seed
was run twice, the two are averaged before the seeds are compared, and their difference is shown.

Each run is judged at its best validation checkpoint, as decided before any run was made.

When there are several groups, the rule for calling two of them different (fixed in BUILD_LOG entry 14, before
the runs): pool the seed-to-seed standard deviation over all groups; two groups differ only if their means are
further apart than t x pooled sd x sqrt(1/n1 + 1/n2), with t the usual 95% value for the pooled degrees of
freedom. For five groups of three seeds that is 1.82 x the pooled sd. Anything closer is a tie, and a tie goes
to the smaller model. So the choice is: the smallest model whose mean is within that distance of the best mean.

The table of all pairs is information, not the decision. Each of its verdicts has a 5% chance of a false alarm, so
among the ten pairs of five truly equal groups, one sweep in four shows at least one "DIFFERENT".

Use a prefix that selects exactly the groups being compared: the pooled spread is taken over every run it finds.

A verdict is given once, when every group has the same number of seeds (three, for the tokenizer comparison). Before
that the table is marked PARTIAL and no choice is named: simulated on equal arms, the choice after two seeds differs
from the choice after three about one time in eleven. A run that blew up and was replaced (a DIVERGED file in its
folder) is listed, not hidden.

If every run also has an evaluation-best.json (from scripts/evaluate.py), the same test is shown for the score on
everything except speaker-label lines. That table is information and decides nothing: it was named in advance as a
secondary (BUILD_LOG entry 15), because most of the run-to-run noise sits in the labels.

Writes docs/results.json, or docs/results-<prefix>.json when a prefix is given. (The runs themselves are too big
for git; these files are the record.)
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
prefix = sys.argv[1] if len(sys.argv) > 1 else ""
NOT_A_SETTING = {"name", "seed", "steps", "parameters", "git_commit", "torch", "device", "fingerprints"}
T_95 = [None, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110,
        2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]  # two-sided 95% Student t, by degrees of freedom

groups, discarded, unfinished = {}, [], []
for folder in sorted((ROOT / "runs").glob(f"{prefix}*")):
    if (folder / "DIVERGED").exists():
        discarded.append(folder.name)
    if not (folder / "result.json").exists():
        if not (folder / "DIVERGED").exists():
            unfinished.append(folder.name)
        continue
    config, result = json.loads((folder / "config.json").read_text()), json.loads((folder / "result.json").read_text())
    settings = json.dumps({k: v for k, v in config.items() if k not in NOT_A_SETTING}, sort_keys=True)
    groups.setdefault(settings, []).append({"name": config["name"], "seed": config["seed"], "steps": config["steps"], "parameters": config["parameters"],
                                            "git_commit": config["git_commit"], "torch": config.get("torch"), "fingerprints": config.get("fingerprints"),
                                            "minutes": result["minutes"],
                                            "best_validation_bpc": result["best"]["validation_bpc"], "best_step": result["best"]["step"],
                                            "best_passes": result["best"]["passes"], "seen_bpc_at_best": result["best"]["seen_bpc"],
                                            "final_validation_bpc": result["final_validation_bpc"],
                                            "everything_else_bpc": json.loads((folder / "evaluation-best.json").read_text())["everything_else"]["bits_per_character"]
                                            if (folder / "evaluation-best.json").exists() else None})
if not groups:
    sys.exit("no finished runs found")

everything = [json.loads(settings) for settings in groups]
varying = sorted(k for k in set().union(*everything) if len({json.dumps(s.get(k)) for s in everything}) > 1)
out, squares, freedom = [], 0.0, 0
show = lambda value: f"{value:.1f}" if isinstance(value, float) else str(value)
for settings, runs in sorted(groups.items(), key=lambda item: item[1][0]["parameters"]):   # smallest model first
    settings = json.loads(settings)
    by_seed = {}
    for run in runs:
        by_seed.setdefault(run["seed"], []).append(run["best_validation_bpc"])
    seed_means = [sum(scores) / len(scores) for scores in by_seed.values()]
    mean = sum(seed_means) / len(seed_means)
    deviation = math.sqrt(sum((m - mean) ** 2 for m in seed_means) / (len(seed_means) - 1)) if len(seed_means) > 1 else None
    squares, freedom = squares + sum((m - mean) ** 2 for m in seed_means), freedom + len(seed_means) - 1
    repeats = {seed: max(scores) - min(scores) for seed, scores in by_seed.items() if len(scores) > 1}

    label = ", ".join(f"{k} {show(settings.get(k))}" for k in varying) or "all settings the same"
    print(f"{label}   ({runs[0]['parameters']:,} parameters, {runs[0]['steps']:,} steps)")
    for run in runs:
        at_the_end = "   <- best at the very end: a longer run might do better" if run["best_step"] == run["steps"] else ""
        print(f"    {run['name']:<28} seed {run['seed']}   best {run['best_validation_bpc']:.4f} at step {run['best_step']:>6,} ({run['best_passes']:4.1f} passes)"
              f"   seen {run['seen_bpc_at_best']:.4f}   final {run['final_validation_bpc']:.4f}   {run['minutes']:5.1f} min{at_the_end}")
    print(f"    mean over {len(seed_means)} seed(s): {mean:.4f} bits per character" + (f", standard deviation {deviation:.4f}" if deviation is not None else ""))
    for seed, gap in repeats.items():
        print(f"    seed {seed} was run {len(by_seed[seed])} times: those runs differ by {gap:.4f}")
    for what in ("git_commit", "torch", "fingerprints"):
        if len({json.dumps(run[what], sort_keys=True) for run in runs}) > 1:
            print(f"    WARNING: these runs do not share the same {what}. They may not be the same experiment.")
    print()
    out.append({"settings": settings, "seeds": len(seed_means), "mean_best_validation_bpc": mean, "standard_deviation_over_seeds": deviation,
                "same_seed_differences": {str(seed): gap for seed, gap in repeats.items()}, "runs": runs})

summary = {"groups": out, "discarded_because_the_loss_stopped_being_a_number": discarded}
for name in discarded:
    print(f"DISCARDED: {name} (its loss stopped being a number; kept as evidence, replaced by the same run with seed + 1000)")
everyone = [run for group in out for run in group["runs"]]
if len({run["git_commit"] for run in everyone}) > 1 or any("uncommitted" in run["git_commit"] for run in everyone):
    print("WARNING: these runs were not all made by one clean commit. On this GPU an edit re-rolls a score as surely as a new seed (BUILD_LOG entry 14).\n")
complete = len({group["seeds"] for group in out}) == 1 and out[0]["seeds"] >= 3 and not unfinished
if len(out) > 1 and freedom and not complete:
    print("PARTIAL: no verdict yet. It is given once, when every group has the same number of seeds (three or more) and no run is still unfinished"
          + (f" ({', '.join(unfinished)})." if unfinished else "."))
if len(out) > 1 and freedom and complete:
    pooled = math.sqrt(squares / freedom)
    t = T_95[min(freedom, len(T_95) - 1)]
    summary["pooled_standard_deviation"], summary["degrees_of_freedom"] = pooled, freedom
    print(f"pooled standard deviation over seeds: {pooled:.4f} ({freedom} degrees of freedom)")
    for i, a in enumerate(out):
        for b in out[i + 1:]:
            needed = t * pooled * math.sqrt(1 / a["seeds"] + 1 / b["seeds"])
            gap = abs(a["mean_best_validation_bpc"] - b["mean_best_validation_bpc"])
            name = lambda group: ", ".join(f"{k} {show(group['settings'].get(k))}" for k in varying)
            print(f"    {name(a)}  against  {name(b)}:  {gap:.4f} apart, give or take {needed:.4f}  ->  {'DIFFERENT' if gap > needed else 'a tie'}")
    print("    (each verdict above has a 5% false-alarm rate of its own; the choice below is what the rule decides)")
    best = min(out, key=lambda group: group["mean_best_validation_bpc"])
    within = [g for g in out if g["mean_best_validation_bpc"] - best["mean_best_validation_bpc"] <= t * pooled * math.sqrt(1 / g["seeds"] + 1 / best["seeds"])]
    choice = min(within, key=lambda group: group["runs"][0]["parameters"])
    summary["choice"] = {"settings": choice["settings"], "mean_best_validation_bpc": choice["mean_best_validation_bpc"], "lowest_mean": best["mean_best_validation_bpc"]}
    print(f"\nlowest mean: {name(best)} at {best['mean_best_validation_bpc']:.4f}. The smallest model within reach of it: {name(choice)} at {choice['mean_best_validation_bpc']:.4f}")

    if all(run["everything_else_bpc"] is not None for run in everyone):   # the named secondary: same test, on everything but speaker-label lines. It decides nothing.
        squares_else = 0.0
        for group in out:
            by_seed = {}
            for run in group["runs"]:
                by_seed.setdefault(run["seed"], []).append(run["everything_else_bpc"])
            per_seed = [sum(v) / len(v) for v in by_seed.values()]
            group["mean_everything_else_bpc"] = sum(per_seed) / len(per_seed)
            squares_else += sum((m - group["mean_everything_else_bpc"]) ** 2 for m in per_seed)
        pooled_else = math.sqrt(squares_else / freedom)
        summary["secondary_everything_else"] = {"pooled_standard_deviation": pooled_else, "means": [[name(group), group["mean_everything_else_bpc"]] for group in out]}
        print(f"\nSECONDARY, decides nothing: the same test on everything except speaker-label lines (pooled standard deviation {pooled_else:.4f})")
        for group in out:
            print(f"    {name(group):<44}{group['mean_everything_else_bpc']:.4f}")
        for i, a in enumerate(out):
            for b in out[i + 1:]:
                needed = t * pooled_else * math.sqrt(1 / a["seeds"] + 1 / b["seeds"])
                gap = abs(a["mean_everything_else_bpc"] - b["mean_everything_else_bpc"])
                print(f"    {name(a)}  against  {name(b)}:  {gap:.4f} apart, give or take {needed:.4f}  ->  {'different' if gap > needed else 'a tie'}")

record = ROOT / "docs" / (f"results-{prefix.strip('-_.')}.json" if prefix else "results.json")
record.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote docs/{record.name}")
