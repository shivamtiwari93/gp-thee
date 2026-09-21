"""Score a trained model on the validation works, and take the score apart.

Run:  uv run python scripts/evaluate.py --run pilot-char-17            (about ten seconds on the GPU)
      uv run python scripts/evaluate.py --run pilot-char-17 --which last

Loads runs/<name>/best.pt (the moment of the run with the best validation score) and reports bits per
character: over everything, per work, and for speaker-label lines against everything else. Then it sets
the result beside the baselines from scripts/baselines.py, if that has been run.

Why per work and per kind of line? The validation works are two plays whose characters the model has never
met. "ROMEO." is a name it must spell out from nothing, 162 times in the play. So part of the validation
score measures "can you guess the names in a play you have not read", which no amount of skill could do.
Splitting the score shows how much.

This script reads the validation works only. The test works stay closed until the final evaluation.

Writes runs/<name>/evaluation-<which>.json.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from gp_thee.data import DATA, load_tokens, load_works
from gp_thee.evaluation import breakdown
from gp_thee.tokenizer import load as load_tokenizer
from gp_thee.train import evaluate, fingerprints, load_checkpoint

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--run", required=True, help="the run's folder name under runs/")
parser.add_argument("--which", default="best", choices=["best", "last"], help="which checkpoint of the run")
parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
args = parser.parse_args()

folder = ROOT / "runs" / args.run
model, saved = load_checkpoint(folder / f"{args.which}.pt", args.device)
name = saved["run_config"]["tokenizer"]
tokenizer, stream, works = load_tokenizer(DATA / "tokenizers" / f"{name}.json"), load_tokens(name, "validation"), load_works("validation")
if "fingerprints" in saved:  # is this the tokenizer, and the validation text, that the run was scored on while it trained?
    today = fingerprints(name, load_tokens(name, "train"), stream)
    changed = [what for what in ("tokenizer", "validation_stream") if saved["fingerprints"][what] != today[what]]
    if changed:
        sys.exit(f"this checkpoint was trained with a different {' and '.join(changed)} from the one on disk now; its scores would mean nothing")
else:
    print("(a checkpoint from the first version of train.py: it carries no checksums, so the tokenizer cannot be verified)\n")
parts = breakdown(evaluate(model, stream, args.device), stream, tokenizer, works)

print(f"{args.run} ({args.which} checkpoint: step {saved['step']:,}, {name} tokenizer, {model.config.parameter_count():,} parameters)\n")
print(f"{'':<36}{'characters':>12}{'tokens':>10}{'bits per character':>20}")
rows = [("validation works, everything", parts["all"])] + [("  " + work["title"], work) for work in parts["works"]] \
     + [("  speaker-label lines", parts["speaker_labels"]), ("  everything else", parts["everything_else"])]
for label, part in rows:
    shown = "n/a" if part["bits_per_character"] is None else f"{part['bits_per_character']:.4f}"  # a set of poems has no speaker labels
    print(f"{label:<36}{part['characters']:>12,}{part['tokens']:>10,}{shown:>20}")
recorded = saved["facts"]["validation_bpc"]
print(f"\n(the training run recorded {recorded:.4f} for this checkpoint; computed again just now: {parts['all']['bits_per_character']:.4f})")
if abs(recorded - parts["all"]["bits_per_character"]) > 1e-3:
    sys.exit("those two should agree. Something differs between the training run and today: the code, the data or the device.")

baselines = ROOT / "docs" / "baselines.json"
if baselines.exists():
    print("\nagainst predictors with no neural network (scripts/baselines.py), same works, same measure:")
    for label, result in json.loads(baselines.read_text(encoding="utf-8"))["results"].items():
        primed = "after_reading_the_training_works" in result  # the compressors get to read the training works first, as the model did
        print(f"  {label + (', after reading the training works' if primed else ''):<58}{result['after_reading_the_training_works' if primed else 'bits_per_character']:>8.3f}")
    print(f"  {'this model':<58}{parts['all']['bits_per_character']:>8.3f}")

out = {"run": args.run, "checkpoint": args.which, "step": saved["step"], "tokenizer": name, "git_commit_of_training": saved["git_commit"],
       "scored_on": args.device, "validation_bpc_recorded_while_training": recorded, **parts}
(folder / f"evaluation-{args.which}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"\nwrote runs/{args.run}/evaluation-{args.which}.json")
