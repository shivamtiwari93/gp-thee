"""WHERE in the text do the tokenizers' models differ? An exploration, made AFTER the verdict. It decides nothing.

Run:  uv run python scripts/where_they_differ.py        (about two minutes on the GPU, after scripts/sweep.py)

The comparison of part 6 was decided by a rule fixed in advance (scripts/summarise_runs.py). This script was written
afterwards, out of curiosity about the answer, so treat what it shows as a description and not as a test.

How can models that cut the text differently be compared piece by piece? Through the one unit they share. Every
tokenizer in this project first cuts the text into CHUNKS by the same rule (a word with its leading space, a run of
punctuation, a run of line breaks: gp_thee.tokenizer.chunks), and no piece ever crosses a chunk's edge. So each
model's bits can be added up exactly within each chunk, whatever its pieces are, and the chunks' totals add up to the
model's whole score. No sharing-out of bits between characters is needed, which would blur everything: a word-fragment
piece carries its leading space, so sharing its bits equally would move cost from words onto spaces.

Chunks are then grouped by what they are, and word chunks by how often the word occurs in the TRAINING works. Each
cell of the table is that group's contribution to the model's bits per character over the whole validation text, mean
of the three seeds, each run at its best checkpoint. The columns add up to the scores in docs/results-sweep.json.

Writes docs/where_they_differ.json.
"""

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import torch

from gp_thee.data import DATA, load_tokens, load_works
from gp_thee.evaluation import piece_lengths, speaker_label_characters
from gp_thee.tokenizer import chunks
from gp_thee.tokenizer import load as load_tokenizer
from gp_thee.train import evaluate, load_checkpoint

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("char", "bpe-1024", "bpe-1536", "bpe-2048", "bpe-4096")
GROUPS = ["word seen 1,000 times or more", "word seen 100 to 999 times", "word seen 10 to 99 times", "word seen 1 to 9 times", "word never seen in training",
          "punctuation", "line breaks", "lone spaces", "speaker-label lines"]

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--prefix", default="sweep-")
parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
args = parser.parse_args()

works = load_works("validation")
characters = sum(map(len, works))
seen = collections.Counter(chunk.strip().lower() for work in load_works("train") for chunk in chunks(work) if chunk.strip()[:1].isalpha())
on_label = np.concatenate([speaker_label_characters(work) for work in works])

# every chunk of the validation text: where it begins and ends (in characters), and which group it belongs to
edges, group_of, offset = [], [], 0
for work in works:
    for chunk in chunks(work):
        word = chunk.strip().lower()
        if on_label[offset]:
            group = "speaker-label lines"
        elif word[:1].isalpha():
            times = seen[word]
            group = GROUPS[4] if times == 0 else GROUPS[3] if times < 10 else GROUPS[2] if times < 100 else GROUPS[1] if times < 1000 else GROUPS[0]
        else:
            group = "line breaks" if "\n" in chunk else "lone spaces" if not word else "punctuation"
        edges.append((offset, offset + len(chunk)))
        group_of.append(GROUPS.index(group))
        offset += len(chunk)
edges, group_of = np.array(edges), np.array(group_of)


def bits_per_chunk(arm: str) -> np.ndarray:
    """Mean over the arm's three runs of the bits each chunk costs. A START token's bits go to the chunk that follows it."""
    tokenizer, stream = load_tokenizer(DATA / "tokenizers" / f"{arm}.json"), np.asarray(load_tokens(arm, "validation")).astype(np.int64)
    lengths = piece_lengths(tokenizer)[stream]
    begins = np.concatenate([[0], np.cumsum(lengths)[:-1]])                       # the character at which each token begins
    chunk_of_token = np.searchsorted(edges[:, 1], begins, side="right")            # no piece crosses a chunk's edge, so its first character places it
    if (begins[lengths > 0] + lengths[lengths > 0] > edges[chunk_of_token[lengths > 0], 1]).any():
        raise AssertionError("a piece crosses the edge of a chunk")
    totals = []
    for folder in sorted((ROOT / "runs").glob(f"{args.prefix}{arm}-seed-*")):
        if (folder / "result.json").exists():
            model, _ = load_checkpoint(folder / "best.pt", args.device)
            bits = np.concatenate([[0.0], evaluate(model, stream, args.device)]) / math.log(2)
            totals.append(np.bincount(chunk_of_token, weights=bits, minlength=len(edges)))
    return np.mean(totals, axis=0)


table = {arm: bits_per_chunk(arm) for arm in ARMS}
share = [(edges[group_of == g, 1] - edges[group_of == g, 0]).sum() / characters for g in range(len(GROUPS))]
print(f"{'a word chunk includes its leading space':<42}{'chunks':>8}{'of text':>9}" + "".join(f"{arm:>10}" for arm in ARMS) + "   <- bits per character of the WHOLE text that each group adds")
out = {"note": "exploratory, made after the verdict; decides nothing", "groups": {}}
for g, name in enumerate(GROUPS):
    adds = {arm: float(table[arm][group_of == g].sum() / characters) for arm in ARMS}
    per_chunk = {arm: float(table[arm][group_of == g].mean()) for arm in ARMS}
    out["groups"][name] = {"chunks": int((group_of == g).sum()), "share_of_characters": float(share[g]), "adds_bits_per_character": adds, "bits_per_chunk": per_chunk}
    print(f"{name:<42}{(group_of == g).sum():>8,}{share[g]:>9.1%}" + "".join(f"{adds[arm]:>10.4f}" for arm in ARMS))
out["total"] = {arm: float(table[arm].sum() / characters) for arm in ARMS}
print(f"{'whole text':<59}" + "".join(f"{out['total'][arm]:>10.4f}" for arm in ARMS))
print(f"\n{'bits per chunk':<59}" + "".join(f"{arm:>10}" for arm in ARMS))
for name in GROUPS[:5] + ["speaker-label lines"]:
    print(f"{name:<59}" + "".join(f"{out['groups'][name]['bits_per_chunk'][arm]:>10.2f}" for arm in ARMS))
(ROOT / "docs" / "where_they_differ.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print("\nwrote docs/where_they_differ.json")
