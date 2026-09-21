"""How well can Shakespeare be predicted WITHOUT a neural network? The numbers our model has to beat.

Run:  uv run python scripts/baselines.py        (under a minute, no GPU)

Everything here is fitted on the 39 training works and scored on the 2 validation works, in bits per
character, exactly like the model. The test works are not opened.

  blind guess        every one of the 97 characters equally likely: log2(97) bits
  character counts   an n-gram of order 1
  n-grams            orders 2 to 8: predict from the last 1 to 7 characters, by counting (see gp_thee.evaluation.NGram)
  bzip2, xz          general-purpose file compressors, alone and after reading the training works

We report every n-gram order rather than choosing one. Choosing the best order by its validation score would
be tuning on the validation works. It would flatter the baseline, not the model, so it would be a safe kind of
cheating, but a table needs no choice at all.

Writes docs/baselines.json.
"""

import json
import math
import time
from pathlib import Path

from gp_thee.data import corpus_alphabet, load_works
from gp_thee.evaluation import NGram, compressed_bits_per_character, speaker_label_characters

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
train, validation = load_works("train"), load_works("validation")
alphabet = corpus_alphabet()
characters = sum(len(work) for work in validation)
on_label = np.concatenate([speaker_label_characters(work) for work in validation])
print(f"fitted on {len(train)} training works ({sum(map(len, train)):,} characters), scored on {len(validation)} validation works ({characters:,} characters)")
print(f"speaker-label lines are {on_label.mean():.2%} of the validation characters\n")

results = {"blind guess": {"bits_per_character": math.log2(len(alphabet))}}
print(f"{'':<28}{'all':>8}{'speaker labels':>16}{'everything else':>17}")
print(f"{'blind guess':<28}{math.log2(len(alphabet)):>8.3f}")
for order in range(1, 9):
    started = time.perf_counter()
    bits = NGram(order, alphabet).fit(train).bits(validation)
    name = "character counts" if order == 1 else f"{order}-gram (looks back {order - 1})"
    results[name] = {"bits_per_character": bits.mean(), "speaker_labels": bits[on_label].mean(), "everything_else": bits[~on_label].mean(),
                     "per_work": {work.split("\n", 1)[0]: part.mean() for work, part in zip(validation, np.split(bits, np.cumsum([len(w) for w in validation])[:-1]))}}
    print(f"{name:<28}{bits.mean():>8.3f}{bits[on_label].mean():>16.3f}{bits[~on_label].mean():>17.3f}   ({time.perf_counter() - started:.1f} s)")

text = "".join(validation)
alone, primed = compressed_bits_per_character(text), compressed_bits_per_character(text, after_reading="".join(train))
print()
for name in alone:
    results[name] = {"bits_per_character": alone[name], "after_reading_the_training_works": primed[name]}
    print(f"{name:<28}{alone[name]:>8.3f}   after reading the training works first: {primed[name]:.3f}")

out = {"fitted_on": "the 39 training works", "scored_on": "the 2 validation works", "characters": characters,
       "speaker_label_share_of_characters": float(on_label.mean()),
       "results": json.loads(json.dumps(results, default=float))}
(ROOT / "docs" / "baselines.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("\nwrote docs/baselines.json")
