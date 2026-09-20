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
- [x] Corpus cleaned into 44 files, one per work, and audited twice ([what was removed and kept](data/processed/README.md))
- [x] Train / validation / test split, by whole works, frozen and leak-checked ([why these five works](scripts/make_split.py))
- [ ] Tokenizer trained on Shakespeare only
- [ ] The model: GP-Thee-11M, a 10.8M parameter GPT
- [ ] Training, with correctness checks first
- [ ] Evaluation: bits per character, baselines, memorisation report
- [ ] Sampling and the "speak as a character" wrapper
- [ ] Weights on Hugging Face
- [ ] Blog post

## Layout

```
data/        the corpus: raw/ is the untouched download, processed/ is generated from it
blog/        the write-up, one section per milestone
docs/        PLAN.md (roadmap) and BUILD_LOG.md (the detailed step-by-step record)
scripts/     one-off commands such as the dataset download
src/gp_thee/ the library: data prep, tokenizer, model, training, sampling
```

## Reproduce

```bash
brew install uv                              # the only thing you install by hand
git clone https://github.com/shivamtiwari93/gp-thee && cd gp-thee
uv sync                                      # Python 3.14 + PyTorch 2.14, from the lockfile
uv run python scripts/download_data.py       # confirms the corpus checksum
uv run python scripts/prepare_data.py        # rebuilds data/processed/ from the raw file, byte for byte
uv run python scripts/make_split.py          # re-checks the split: held-out works share no text with training
uv run python scripts/smoke_test_gpu.py      # Mac only: GPU answers match the CPU's
uv run python scripts/count_params.py        # the parameter arithmetic, checked against PyTorch
```

More steps are added here as they are built.

## Read along

1. [Prerequisites and setup](blog/01-prerequisites-and-setup.md): the starting point, the tools and why, and how we know the model has 10.8 million parameters before training it.

## Data and licence

The corpus is Project Gutenberg eBook #100, *The Complete Works of William Shakespeare*, which is in the public domain in the United States. The untouched file is in [data/raw](data/raw/), with the Project Gutenberg notice that accompanies it. This project is not affiliated with Project Gutenberg. Code and weights are released under the [MIT licence](LICENSE).
