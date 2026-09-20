# GP-Thee

**A GPT trained from scratch on nothing but the complete works of Shakespeare.**

Imagine the only text that ever existed was Shakespeare: 38 plays, 154 sonnets, 5 poems. No internet, no Wikipedia, no other books. GP-Thee is a small language model built under that rule. It has no pretrained weights, no borrowed tokenizer and no outside text. Everything it knows, it learned from about 5 MB of Elizabethan English, on one laptop.

This is an educational project. The point is to show every step of building a language model, small enough that you can read all the code and re-run all of it yourself.

> **Status: work in progress.** The environment is set up and the corpus is downloaded and verified. No model has been trained yet. Progress is tracked below and written up step by step in [docs/BUILD_LOG.md](docs/BUILD_LOG.md).

## What it will and will not be

GP-Thee is an in-character autocomplete, not an assistant. It should write convincing pseudo-Shakespeare with proper play formatting and speaker turns. It will know nothing outside the plays and will not follow instructions. The closest thing to chat is in-universe dialogue: give it a speaker and a line, and it answers as another character.

## Progress

- [x] Research: dataset, training stack, model sizing ([build log](docs/BUILD_LOG.md))
- [x] Project scaffold
- [x] Environment: uv, Python 3.14, PyTorch 2.14 on the Mac GPU
- [x] Corpus downloaded and checksum-verified
- [ ] Clean the corpus
- [ ] Train / validation / test split (by whole works)
- [ ] Tokenizer trained on Shakespeare only
- [ ] The model (~10M parameter GPT)
- [ ] Training, with correctness checks first
- [ ] Evaluation: bits per character, baselines, memorisation report
- [ ] Sampling and the "speak as a character" wrapper
- [ ] Weights on Hugging Face
- [ ] Blog post

## Layout

```
data/        corpus provenance; raw download is git-ignored, cleaned text is committed
docs/        PLAN.md (roadmap) and BUILD_LOG.md (the detailed step-by-step record)
scripts/     one-off commands such as the dataset download
src/gp_thee/ the library: data prep, tokenizer, model, training, sampling
```

## Reproduce

```bash
brew install uv
uv sync
uv run python scripts/download_data.py
```

More steps are added here as they are built.

## Data and licence

The corpus is Project Gutenberg eBook #100, *The Complete Works of William Shakespeare*, which is in the public domain in the United States. Code and weights are released under the [MIT licence](LICENSE).
