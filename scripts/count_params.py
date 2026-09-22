"""Check the parameter-count arithmetic against PyTorch.

Run:  uv run python scripts/count_params.py

Builds a skeleton with the shape GP-Thee will have (no forward pass, just the
weight matrices) and compares PyTorch's count with the hand formula

    12*d*d*L  +  (2L+1)*d  +  V*d  +  T*d

where d = width, L = layers, V = vocabulary size, T = context length.
See blog/01-prerequisites-and-setup.md for where each term comes from.
"""
import torch.nn as nn

def build(V, T, d, L):
    ln = lambda: nn.LayerNorm(d, bias=False)
    block = lambda: nn.ModuleDict(dict(
        ln_1=ln(), qkv=nn.Linear(d, 3 * d, bias=False), attn_out=nn.Linear(d, d, bias=False),
        ln_2=ln(), mlp_up=nn.Linear(d, 4 * d, bias=False), mlp_down=nn.Linear(4 * d, d, bias=False)))
    m = nn.ModuleDict(dict(tok=nn.Embedding(V, d), pos=nn.Embedding(T, d),
                           blocks=nn.ModuleList([block() for _ in range(L)]), ln_f=ln(),
                           head=nn.Linear(d, V, bias=False)))
    m["head"].weight = m["tok"].weight  # weight tying: the output layer reuses the token embedding
    return m

# V=100 is the historical raw-file alphabet used for every character-model row in blog part 1.
# Cleaning later removed three characters (part 2), and START added one (part 3), giving the current
# character vocabulary V=98. Both are printed, but the `Part 1` labels reproduce that post's table.
cases = [
    ("current char model: 6L/384d, V=98", 98, 256, 384, 6),
    ("Part 1 base: 6L/384d, V=100", 100, 256, 384, 6),
    ("Part 1 BPE comparison: 6L/384d, V=2048", 2048, 256, 384, 6),
    ("Part 1 smaller: 6L/256d, V=100", 100, 256, 256, 6),
    ("Part 1 larger: 12L/512d, V=100", 100, 256, 512, 12),
]

for name, V, T, d, L in cases:
    m = build(V, T, d, L)
    pytorch = sum(p.numel() for p in m.parameters())   # .parameters() de-duplicates the tied weight
    hand = 12 * d * d * L + (2 * L + 1) * d + V * d + T * d
    print(f"{name:44s} pytorch={pytorch:>11,}  hand={hand:>11,}  match={pytorch == hand}")
