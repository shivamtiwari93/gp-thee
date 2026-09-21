"""Train a model.

Run:  uv run python scripts/train.py --name first-run
      uv run python scripts/train.py --name bpe-2048-seed-1 --tokenizer bpe-2048 --seed 1 --passes 34
      uv run python scripts/train.py --name first-run --resume        (carry on after an interruption)

Everything lands in runs/<name>/ (not committed): config.json, log.csv, samples.txt, best.pt, last.pt, result.json.
Every setting is a field of RunConfig in src/gp_thee/train.py, and every field is a flag here.

--resume needs only the name: the settings come from the run's own config.json, because a resumed run must be
the run that was started. A flag that contradicts them is an error, except --device.

Plug the laptop in. A run on battery is slower, and a run that drains the battery does not finish.
"""

import argparse
import dataclasses
import inspect
import json
import re
import subprocess

from gp_thee.train import ROOT, RunConfig, train

# each field's comment in RunConfig doubles as its help text
comments = dict(re.findall(r"^\s+(\w+):[^#\n]*#\s*(.+)$", inspect.getsource(RunConfig), re.M))
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
for field in dataclasses.fields(RunConfig):
    default = "" if field.default is dataclasses.MISSING else f" (default: {field.default})"
    options = {"action": argparse.BooleanOptionalAction} if field.type is bool else {"type": field.type}
    parser.add_argument(f"--{field.name.replace('_', '-')}", required=field.default is dataclasses.MISSING, default=argparse.SUPPRESS,
                        help=comments.get(field.name, "") + default, **options)
parser.add_argument("--resume", action="store_true", help="carry on from runs/<name>/last.pt, with the settings the run began with")
parser.add_argument("--overwrite", action="store_true", help="replace an existing run of the same name")
given = vars(parser.parse_args())
resume, overwrite = given.pop("resume"), given.pop("overwrite")

began_with = {}
if resume:
    config = ROOT / "runs" / given["name"] / "config.json"
    if not config.exists():
        parser.error(f"nothing to resume: {config} does not exist")
    fields = {field.name for field in dataclasses.fields(RunConfig)}
    began_with = {k: v for k, v in json.loads(config.read_text()).items() if k in fields}
run = RunConfig(**{**began_with, **given})  # a flag that contradicts the run's settings survives to here, and train() refuses it

try:
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()
    if power and "Battery Power" in power[0]:
        print("WARNING: running on battery. Plug in: a long run on battery is slower and may not finish.\n")
except FileNotFoundError:
    pass  # not a Mac

train(run, resume=resume, overwrite=overwrite)
