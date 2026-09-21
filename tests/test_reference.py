"""The whole model, checked against a second implementation that shares no code with it.

Run:  uv run pytest tests/test_reference.py

The reference below is NumPy, 64-bit, and deliberately slow and plain: loops over heads and positions, the norm
and GELU written out from their definitions. It was written from the DESCRIPTION of a GPT-2-style model, not
from model.py. If the two agree to ten decimal places on random inputs, the architecture is pinned: a stray
change to the order of a block, the scaling of attention, the way heads are split, or how positions are
numbered shows up here, even when both of model.py's attention paths share the fault.

The weights are pushed well away from their small starting values first. At the start, attention is almost
uniform and several real faults change the scores only in the sixth decimal place.
"""

import math

import numpy as np
import pytest
import torch

from gp_thee.model import GPT, Config

erf = np.vectorize(math.erf)


def norm(x, scale):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * scale


def reference_scores(w: dict, tokens: np.ndarray, layers: int, heads: int) -> np.ndarray:
    length, width = len(tokens), w["token_embedding.weight"].shape[1]
    size = width // heads
    x = w["token_embedding.weight"][tokens] + w["position_embedding.weight"][:length]  # position 0 first
    for layer in range(layers):
        name = f"blocks.{layer}."
        h = norm(x, w[name + "norm_1.weight"])
        q, k, v = np.split(h @ w[name + "attention.query_key_value.weight"].T, 3, axis=-1)
        mixed = np.zeros_like(x)
        for head in range(heads):
            mine = slice(head * size, (head + 1) * size)
            for t in range(length):
                match = k[:t + 1, mine] @ q[t, mine] / math.sqrt(size)       # itself and everything before it
                share = np.exp(match - match.max())
                mixed[t, mine] = (share / share.sum()) @ v[:t + 1, mine]
        x = x + mixed @ w[name + "attention.output.weight"].T                # ADD, do not replace
        h = norm(x, w[name + "norm_2.weight"])
        hidden = h @ w[name + "feed_forward.expand.weight"].T
        x = x + (hidden * 0.5 * (1 + erf(hidden / math.sqrt(2)))) @ w[name + "feed_forward.shrink.weight"].T
    return norm(x, w["final_norm.weight"]) @ w["token_embedding.weight"].T   # the same table, coming out


@pytest.mark.parametrize("builtin", [True, False])
def test_the_model_agrees_with_a_reference_written_from_the_description(builtin):
    torch.manual_seed(0)
    config = Config(vocab_size=50, context=32, layers=2, heads=4, width=64, dropout=0.0, builtin_attention=builtin)
    model = GPT(config).double().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(8.0) if p.dim() >= 2 else p.add_(0.3 * torch.randn_like(p))
    w = {name: p.detach().numpy() for name, p in model.named_parameters()}

    tokens = torch.randint(0, 50, (3, 20))  # shorter than the context, on purpose
    targets = torch.randint(0, 50, (3, 20))
    with torch.no_grad():
        scores, loss = model(tokens, targets)
    expected = np.stack([reference_scores(w, row.numpy(), config.layers, config.heads) for row in tokens])
    assert np.abs(scores.numpy() - expected).max() < 1e-10

    log_probability = expected - np.log(np.exp(expected).sum(-1, keepdims=True))
    surprise = -np.take_along_axis(log_probability, targets.numpy()[..., None], axis=-1)
    assert abs(loss.item() - surprise.mean()) < 1e-5  # looser: the model works out its loss in 32-bit, by design
