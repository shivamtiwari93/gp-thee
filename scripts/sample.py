"""Ask a trained model for text.

Run:  uv run python scripts/sample.py --run sweep-char-seed-1 --prompt $'\\n\\nHAMLET.\\n'
      uv run python scripts/sample.py --run sweep-char-seed-1 --suite          (the ten fixed prompts, written to runs/<name>/)
      uv run python scripts/sample.py --run sweep-char-seed-1 --scene          (two turns of the "speak as a character" wrapper)
      uv run python scripts/sample.py --compare sweep-char-seed-1 pilot-char-68

The prompt passes through the normaliser in src/gp_thee/sampling.py first, and everything it changed is printed
above the sample. A character this universe does not contain is refused by name rather than quietly dropped.

Defaults, and what everything published uses: temperature 0.8, no top-k, no top-p, stop when the model says the
work has ended. Anything else is recorded in the header.

The suite is ten fixed prompts, the same ten for every checkpoint for the rest of the project, so that two
checkpoints can be set side by side. It writes runs/<name>/suite.txt to read and suite.json to compare.
"""

import argparse
import dataclasses
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from gp_thee.data import DATA
from gp_thee.sampling import SCENE, SUITE, Refused, as_a_character, generate
from gp_thee.tokenizer import load as load_tokenizer
from gp_thee.train import load_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def open_run(name: str, which: str, device: str):
    folder = ROOT / "runs" / name
    model, saved = load_checkpoint(folder / f"{which}.pt", device)
    tokenizer = load_tokenizer(DATA / "tokenizers" / f"{saved['run_config']['tokenizer']}.json")
    facts = {"run": name, "checkpoint": f"{which}.pt", "step": saved["step"],
             "checkpoint_sha256": hashlib.sha256((folder / f"{which}.pt").read_bytes()).hexdigest(),
             "validation_bpc": saved["facts"]["validation_bpc"], "tokenizer": saved["run_config"]["tokenizer"],
             "tokenizer_sha256": saved["fingerprints"]["tokenizer"] if "fingerprints" in saved else None,
             "trained_by_commit": saved["git_commit"], "torch": saved["torch"], "device": device,
             "parameters": model.config.parameter_count()}
    return model, tokenizer, facts


def show(block: dict) -> str:
    """One prompt and what the model wrote, as it goes into a suite file."""
    lines = [f"prompt as typed: {block['typed']!r}"]
    lines.append(f"prompt as fed:   {block['prompt']!r}" if block["prompt"] != block["typed"] else "prompt as fed:   [unchanged]")
    lines += [f"  note: {note}" for note in block["notes"]]
    lines.append(f"stopped because: {block['why']}" + (f"; {block['dropped_from_the_window']} tokens of the prompt fell out of the window"
                                                       if block["dropped_from_the_window"] else ""))
    return "\n".join(lines) + "\n" + "-" * 100 + "\n" + block["prompt"] + block["continuation"] + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--run", help="a folder under runs/")
    parser.add_argument("--which", default="best", choices=["best", "last"])
    parser.add_argument("--prompt", help="what to continue. Use $'...' in a shell to type a line break")
    parser.add_argument("--characters", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 takes the likeliest character every time")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--never-stop", action="store_true", help="forbid START, so the model cannot end the work")
    parser.add_argument("--suite", action="store_true", help="the ten fixed prompts; writes runs/<name>/suite.txt and suite.json")
    parser.add_argument("--scene", action="store_true", help="the 'speak as a character' wrapper on a fixed two-turn scene")
    parser.add_argument("--as", dest="answerer", help="who answers in --scene. Without it the model casts the part itself")
    parser.add_argument("--compare", nargs=2, metavar="RUN", help="set two runs' suites side by side")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    if args.compare:
        suites = [json.loads((ROOT / "runs" / name / "suite.json").read_text()) for name in args.compare]
        print(f"{'':<34}" + "".join(f"{name:<44}" for name in args.compare))
        for field in ("step", "validation_bpc", "trained_by_commit", "checkpoint_sha256"):
            print(f"  {field:<32}" + "".join(f"{str(s['about'][field])[:42]:<44}" for s in suites))
        for i, blocks in enumerate(zip(*[s["prompts"] for s in suites])):
            print(f"\n{'=' * 100}\n{i + 1}. {blocks[0]['kind']}: {blocks[0]['typed']!r}")
            for name, block in zip(args.compare, blocks):
                print(f"\n--- {name}\n{block['continuation']}")
        return

    settings = dict(temperature=args.temperature, seed=args.seed, top_k=args.top_k, top_p=args.top_p,
                    on_start="mask" if args.never_stop else "stop")
    model, tokenizer, facts = open_run(args.run, args.which, args.device)
    facts |= {"sampler": settings, "written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": platform.platform()}

    if args.suite:
        blocks = []
        for i, (kind, typed) in enumerate(SUITE):
            block = generate(model, tokenizer, typed, characters=400, **(settings | {"seed": args.seed + i}))
            blocks.append({"kind": kind, "typed": typed, **block})
            print(f"{i + 1:>3}. {kind}")
        scene = as_a_character(model, tokenizer, SCENE, temperature=args.temperature, seed=args.seed)
        folder = ROOT / "runs" / args.run
        header = "\n".join(f"{k:<22}{v}" for k, v in facts.items() if k != "sampler") + f"\n{'sampler':<22}{settings}\n{'seed rule':<22}{args.seed} + the prompt's number\n"
        (folder / "suite.txt").write_text(header + "\n" + "\n".join(
            f"\n{'=' * 100}\n{i + 1}. {b['kind']}\n{'=' * 100}\n" + show(b) for i, b in enumerate(blocks))
            + f"\n{'=' * 100}\nthe wrapper, not one of the ten: two turns of a scene\n{'=' * 100}\n"
            + f"answered by {scene['answerer']}" + (" (the model cast the part)" if scene["cast_by_the_model"] else "")
            + f"; stopped because {scene['why']}\n" + "-" * 100 + "\n" + scene["transcript"] + "\n", encoding="utf-8")
        (folder / "suite.json").write_text(json.dumps({"about": facts, "prompts": blocks, "scene": scene}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote runs/{args.run}/suite.txt and suite.json")
        return

    if args.scene:
        scene = as_a_character(model, tokenizer, SCENE, answerer=args.answerer, characters=args.characters,
                               temperature=args.temperature, seed=args.seed)
        print(f"answered by {scene['answerer']}" + (" (the model cast the part)" if scene["cast_by_the_model"] else ""))
        print(f"stopped because {scene['why']}\n{'-' * 100}\n{scene['transcript']}")
        return

    if not args.prompt:
        parser.error("say what to continue with --prompt, or ask for --suite or --scene")
    try:
        block = generate(model, tokenizer, args.prompt, characters=args.characters, **settings)
    except Refused as refusal:
        raise SystemExit(f"that prompt cannot be spelled in this universe.\n{refusal}")
    print(show({"typed": args.prompt, **block}))


if __name__ == "__main__":
    main()
