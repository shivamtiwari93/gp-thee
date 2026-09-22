"""Load GP-Thee-11M from this folder. Standalone: needs only torch, safetensors, and gp_thee.model.

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
