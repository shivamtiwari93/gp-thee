"""Package GP-Thee-11M for Hugging Face: safetensors weights, a config, the tokenizer, and a model card.

Run:  uv run python scripts/export_release.py        (writes release/gp-thee-11m/; uploads nothing)

The released run is `sweep-char-seed-1/best.pt`, chosen by the rule in docs/release.json (entry 17) and scored on
the held-out works in docs/final-evaluation.json (entry 20). This turns that checkpoint into a folder anybody can
download and load, and NOTHING here uploads it: publishing to Hugging Face is a separate, deliberate step that
needs the owner logged in (see the printed instructions at the end).

Two deliberate choices:

  * The weights are saved as safetensors, never as the .pt they were trained in. A .pt is a pickle, which is a
    program; loading one from a stranger runs their code. safetensors is only numbers. This is the same reason
    src/gp_thee/train.py loads checkpoints with weights_only=True.
  * The model ties one embedding table to both the input and the output (model.py:149), so the checkpoint holds
    two names for one tensor. safetensors refuses to store the same tensor twice, so the output copy is dropped
    on export and re-tied on load. scripts/load_release.py does the re-tie and is included in the folder, with a
    test that its output matches the original checkpoint to the bit.

Writes release/gp-thee-11m/ : model.safetensors, config.json, tokenizer.json, load_release.py, README.md.
"""

import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from gp_thee.data import DATA
from gp_thee.train import load_checkpoint

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "release" / "gp-thee-11m"
RUN, WHICH = "sweep-char-seed-1", "best"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    release = json.loads((ROOT / "docs" / "release.json").read_text())
    final = json.loads((ROOT / "docs" / "final-evaluation.json").read_text())
    assert release["run"] == RUN, "docs/release.json names a different run; the rule, not this script, decides"

    model, saved = load_checkpoint(ROOT / "runs" / RUN / f"{WHICH}.pt", "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    # The weights, with the tied output copy dropped. state_dict() lists token_embedding.weight and to_scores.weight
    # as the same tensor; keeping both would make safetensors refuse (shared storage) and would waste 150k numbers.
    weights = {name: tensor.clone() for name, tensor in model.state_dict().items() if name != "to_scores.weight"}
    assert "token_embedding.weight" in weights and "to_scores.weight" not in weights
    save_file(weights, OUT / "model.safetensors",
              metadata={"format": "pt", "tied": "to_scores.weight = token_embedding.weight"})

    config = {
        "model_type": "gp-thee",
        "name": "GP-Thee-11M",
        "description": "A GPT trained from scratch on nothing but the works of Shakespeare. Character-level.",
        "architecture": {**saved["model_config"], "parameters": model.config.parameter_count(),
                         "tied_embeddings": True, "note": "to_scores.weight is token_embedding.weight; re-tie on load"},
        "tokenizer": {"kind": "char", "file": "tokenizer.json", "vocab_size": saved["model_config"]["vocab_size"],
                      "sha256": sha256_of(DATA / "tokenizers" / "char.json")},
        "trained_by_commit": saved["git_commit"],
        "scores_bits_per_character": {
            "validation": release["validation_bpc"],
            "test": final["headline"]["the_artifact"]["test_bits_per_character"],
            "note": "validation is the set the run was selected on (2 plays); test is 3 works it never saw."},
        "provenance": {"source_checkpoint": f"runs/{RUN}/{WHICH}.pt",
                       "source_checkpoint_sha256": release["checkpoint_sha256"],
                       "step": saved["step"], "torch": saved.get("torch")},
        "license": "MIT",
        "repository": "https://github.com/shivamtiwari93/gp-thee",
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy(DATA / "tokenizers" / "char.json", OUT / "tokenizer.json")
    _write_loader(OUT / "load_release.py")
    _write_card(OUT / "README.md", config)

    print(f"wrote {OUT.relative_to(ROOT)}/")
    for path in sorted(OUT.iterdir()):
        print(f"  {path.name:20} {path.stat().st_size:>10,} bytes")
    print("\nNothing was uploaded. To publish it yourself (the `hf` CLI, logged in as you):")
    print("  uv run hf auth login                  # once; opens your browser, or takes a token")
    print(f"  uv run hf upload <your-username>/gp-thee-11m {OUT.relative_to(ROOT)} . --commit-message 'GP-Thee-11M'")
    print("\n(`huggingface-cli` is deprecated and no longer works; the command is `hf`.)")
    print("Or create the model at https://huggingface.co/new and drag the folder in.")


def _write_loader(path: Path) -> None:
    path.write_text('''"""Load GP-Thee-11M from this folder. Standalone: needs only torch, safetensors, and gp_thee.model.

    from load_release import load
    model, tokenizer_json = load()          # model is in eval mode, on the CPU

The output projection is tied to the input embedding, so it is not stored in the file; it is re-tied here.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent


def load(device: str = "cpu"):
    from gp_thee.model import GPT, Config
    config = json.loads((HERE / "config.json").read_text())
    architecture = {k: v for k, v in config["architecture"].items()
                    if k in Config.__dataclass_fields__}
    model = GPT(Config(**architecture))
    weights = load_file(HERE / "model.safetensors")
    weights["to_scores.weight"] = weights["token_embedding.weight"]     # re-tie the dropped output copy
    model.load_state_dict(weights)
    model.to(device).eval()
    return model, json.loads((HERE / "tokenizer.json").read_text())


if __name__ == "__main__":
    model, _ = load()
    print(f"loaded GP-Thee-11M: {sum(p.numel() for p in model.parameters()):,} parameters")
''', encoding="utf-8")


def _write_card(path: Path, config: dict) -> None:
    v = config["scores_bits_per_character"]["validation"]
    t = config["scores_bits_per_character"]["test"]
    path.write_text(f'''---
license: mit
tags:
  - shakespeare
  - character-level
  - gpt
  - educational
  - from-scratch
library_name: safetensors
---

# GP-Thee-11M

A GPT trained from scratch on nothing but the complete works of Shakespeare — an educational project. No
pretrained weights, no borrowed tokenizer, no outside text. Everything it knows, it learned from about 5 MB of
Elizabethan English on one laptop.

- **{config["architecture"]["parameters"]:,} parameters** — {config["architecture"]["layers"]} layers,
  {config["architecture"]["heads"]} heads, width {config["architecture"]["width"]}, context
  {config["architecture"]["context"]}.
- **Character-level.** A vocabulary of {config["architecture"]["vocab_size"] - 1} characters plus a start token;
  the tokenizer was fitted on the training works alone and is in `tokenizer.json`.
- **Scores** (bits per character; lower is better): **{v:.4f}** on the two validation plays it was selected on,
  **{t:.4f}** on three works it never saw (a history, a romance and a poem). For scale, the best predictor built
  without a neural network scores 2.32 on the same held-out text, and a general-purpose compressor 2.58.

## What it is, and is not

GP-Thee is an **in-character autocomplete** for a universe that contains only Shakespeare. Give it a speaker label
and a line and it continues the scene. It cannot follow instructions, has never been asked a question, and — as
[part 7](https://github.com/shivamtiwari93/gp-thee/blob/main/blog/07-does-it-recite.md) measures — reproduces
almost none of its training text verbatim: the longest passage of Shakespeare's own words it can be made to write
back is 25 characters. What it memorised is the *shape* of an edition — cast lists, scene headings, speaker
labels — not the verse.

## Use

```python
from load_release import load          # in this folder; needs torch, safetensors, and gp_thee.model
model, tokenizer = load()
```

The output projection is tied to the input embedding, so it is not stored in the safetensors file; `load_release.py`
re-ties it. The full sampler, the tokenizer code, every training log and the complete build history are in the
repository.

## Provenance

Released run `{config["provenance"]["source_checkpoint"]}` at step {config["provenance"]["step"]}, chosen by a rule
fixed before the runs (`docs/release.json`) and scored on the held-out works exactly once (`docs/final-evaluation
.json`). Trained by commit `{config["trained_by_commit"][:12]}`. Built and documented step by step at
**{config["repository"]}**.

## License

MIT. The corpus is Project Gutenberg eBook #100, public domain in the United States. This project is not
affiliated with Project Gutenberg.
''', encoding="utf-8")


if __name__ == "__main__":
    main()
