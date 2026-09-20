"""What does the chunk rule buy? Learn BPE on the training works WITHOUT it, and compare.

Run:  uv run python scripts/compare_chunk_rule.py      (about a minute and a half)

The tokenizer in src/gp_thee/tokenizer.py only merges inside chunks (a word with its leading space, a run of
punctuation, a run of newlines). This script learns the same number of merges with no such rule, so a merge
may swallow a space, a newline, or two words at once, and measures two things on the training works:

  * compression: characters per token. The rule LOSES here. Fewer restrictions, better compression.
  * consistency: in how many different ways is the same word cut up, depending on its neighbours?
    This is what the rule buys, and it is the reason we keep it.

Nothing is written. Without chunks the whole text is one long sequence, so the chunk-by-chunk shortcut in
learn_merges does not apply; this version recounts pairs with numpy before every merge instead.
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from gp_thee.tokenizer import load

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
VOCAB = 2048

split = json.loads((PROCESSED / "split.json").read_text(encoding="utf-8"))
works = [(PROCESSED / f).read_text(encoding="utf-8") for f in split["sets"]["train"]["files"]]
text = "".join(works)
ours = load(ROOT / "data" / "tokenizers" / f"bpe-{VOCAB}.json")

# ---------------------------------------------------------------- learn merges with no chunk rule
vocab = list(ours.chars)
WALL = 10 ** 6  # sits between works and is never merged, so no piece spans two works
ids = np.concatenate([np.array([WALL] + [vocab.index(ch) for ch in work], dtype=np.int64) for work in works])
for _ in range(len(ours.merges)):
    left, right = ids[:-1], ids[1:]
    real = (left != WALL) & (right != WALL)
    pairs, counts = np.unique(left[real] * WALL + right[real], return_counts=True)
    tied = pairs[counts == counts.max()]
    a, b = max(((int(p) // WALL, int(p) % WALL) for p in tied), key=lambda pair: (vocab[pair[0]], vocab[pair[1]]))  # same tie rule as ours
    hits = np.flatnonzero((ids[:-1] == a) & (ids[1:] == b))
    if a == b:  # "aaa": merge the first two, not the overlapping second pair
        kept, last = [], -2
        for h in hits:
            if h > last + 1:
                kept.append(h)
                last = h
        hits = np.array(kept)
    ids[hits] = len(vocab)
    ids = np.delete(ids, hits + 1)
    vocab.append(vocab[a] + vocab[b])

without_rule = [vocab[i] for i in ids if i != WALL]
with_rule = [ours.vocab[i] for work in works for i in ours.encode(work)]
assert "".join(without_rule) == "".join(with_rule) == text


# ---------------------------------------------------------------- compare
def ways_to_cut(word: str, tokens: list[str]) -> Counter:
    """Every occurrence of `word` as a whole word: which tokens cover it? Count the distinct answers."""
    offsets = np.cumsum([0] + [len(t) for t in tokens])
    seen = Counter()
    for match in re.finditer(r"(?<![^\W\d_’])" + re.escape(word) + r"(?![^\W\d_’])", text):
        k = int(np.searchsorted(offsets, match.start(), side="right") - 1)
        covering = []
        while offsets[k] < match.end():
            covering.append(tokens[k])
            k += 1
        seen["|".join(covering)] += 1
    return seen


pieces = vocab[len(ours.chars):]
straddle = [p for p in pieces if re.search(r"\S\s+\S", p)]
print(f"{'':<34}{'no chunk rule':>16}{'with the rule':>16}")
print(f"{'characters per token':<34}{len(text) / len(without_rule):>16.3f}{len(text) / len(with_rule):>16.3f}")
print(f"{'pieces that straddle two words':<34}{len(straddle):>16,}{0:>16}")
print(f"{'pieces mixing a newline with text':<34}{sum(1 for p in pieces if chr(10) in p and p.strip(chr(10))):>16,}{0:>16}")
common = [w for w, _ in Counter(re.findall(r"(?<![^\W\d_’])[a-z]{3,}(?![^\W\d_’])", text)).most_common(150)]
average = [np.mean([len(ways_to_cut(w, tokens)) for w in common]) for tokens in (without_rule, with_rule)]
print(f"{'ways to cut a common word, average':<34}{average[0]:>16.1f}{average[1]:>16.1f}")
for word in ("come", "love", "the"):
    a, b = ways_to_cut(word, without_rule), ways_to_cut(word, with_rule)
    print(f"{'  ' + repr(word):<34}{len(a):>16}{len(b):>16}     e.g. {[k for k, _ in a.most_common(4)]}")
print(f"\nfirst merges without the rule: {[repr(p) for p in pieces[:10]]}")
print(f"some straddling pieces: {[repr(p) for p in straddle[:8]]}")
