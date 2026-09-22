---
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

- **10,757,760 parameters** — 6 layers,
  6 heads, width 384, context
  256.
- **Character-level.** A vocabulary of 97 characters plus a start token;
  the tokenizer was fitted on the training works alone and is in `tokenizer.json`.
- **Scores** (bits per character; lower is better): **1.7492** on the two validation plays it was selected on,
  **1.7958** on three works it never saw (a history, a romance and a poem). For scale, the best predictor built
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

This model lives at **https://huggingface.co/shivamtiwari93/gp-thee-11m**.

Released run `runs/sweep-char-seed-1/best.pt` at step 9236, chosen by a rule
fixed before the runs (`docs/release.json`) and scored on the held-out works exactly once (`docs/final-evaluation
.json`). Trained by commit `0f726008af74`. Built and documented step by step at
**https://github.com/shivamtiwari93/gp-thee**.

## License

MIT. The corpus is Project Gutenberg eBook #100, public domain in the United States. This project is not
affiliated with Project Gutenberg.
