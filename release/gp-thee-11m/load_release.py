"""Load GP-Thee-11M from this folder. Standalone: needs only torch, safetensors, and gp_thee.model.

    from load_release import load
    model, tokenizer_json = load()          # model is in eval mode, on the CPU

The output projection is tied to the input embedding, so it is not stored in the file; it is re-tied here.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

# .absolute(), NOT .resolve(): huggingface_hub's snapshot_download lays a folder out as symlinks into a shared
# blobs/ store, so resolving would follow this file's link into blobs/ and look for config.json beside the hashed
# objects, where it does not exist. .absolute() keeps us in the snapshot folder the caller actually downloaded.
HERE = Path(__file__).absolute().parent


def load(device: str = "cpu", folder: "Path | str | None" = None):
    """Load the model and the tokenizer description. `folder` defaults to the one this file sits in."""
    from gp_thee.model import GPT, Config
    here = Path(folder).absolute() if folder is not None else HERE
    config = json.loads((here / "config.json").read_text())
    architecture = {k: v for k, v in config["architecture"].items()
                    if k in Config.__dataclass_fields__}
    model = GPT(Config(**architecture))
    weights = load_file(here / "model.safetensors")
    weights["to_scores.weight"] = weights["token_embedding.weight"]     # re-tie the dropped output copy
    model.load_state_dict(weights)
    model.to(device).eval()
    return model, json.loads((here / "tokenizer.json").read_text())


if __name__ == "__main__":
    model, _ = load()
    print(f"loaded GP-Thee-11M: {sum(p.numel() for p in model.parameters()):,} parameters")
