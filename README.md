# GP-Thee

**A GPT trained from scratch on nothing but the complete works of Shakespeare.**

Imagine the only text that ever existed was Shakespeare: 38 plays, 154 sonnets, 5 poems. No internet, no Wikipedia, no other books. GP-Thee is a small language model built under that rule. It has no pretrained weights, no borrowed tokenizer and no outside text. Everything it knows, it learned from about 5 MB of Elizabethan English, on one laptop.

This is an educational project. The point is to show every step of building a language model, small enough that you can read all the code and re-run all of it yourself.

> **Status: the build is done; the weights upload is the last step.** GP-Thee-11M is chosen, measured and written up across eight parts. On the three works it was never trained on and never selected against, it scores **1.80 bits per character** — against 2.32 for the best predictor we could build without a neural network — and it reproduces none of Shakespeare's verse, only his editors' scene headings. Every number was measured before it was written; the full record is in [docs/BUILD_LOG.md](docs/BUILD_LOG.md), and the final test's per-token evidence is in [docs/final-evaluation.json](docs/final-evaluation.json).

## What it will and will not be

GP-Thee is an in-character autocomplete, not an assistant. It should write convincing pseudo-Shakespeare with proper play formatting and speaker turns. It will know nothing outside the plays and will not follow instructions. The closest thing to chat is in-universe dialogue: give it a speaker and a line, and it answers as another character.

## Progress

- [x] Research: dataset, training stack, model sizing ([build log](docs/BUILD_LOG.md))
- [x] Project scaffold
- [x] Environment: uv, Python 3.14, PyTorch 2.14 on the Mac GPU
- [x] Corpus downloaded and checksum-verified
- [x] Corpus cleaned into 44 files, one per work, and audited twice ([what was removed and kept](data/processed/README.md))
- [x] Train / validation / test split, by whole works, frozen and leak-checked ([why these five works](scripts/make_split.py))
- [x] Tokenizers written from scratch and fitted on the training works only: characters, and word fragments at four sizes ([src/gp_thee/tokenizer.py](src/gp_thee/tokenizer.py))
- [x] The model: GP-Thee-11M, a 10.8M parameter GPT, with a correctness gate and an independent reference implementation ([src/gp_thee/model.py](src/gp_thee/model.py))
- [x] Training: the loop, honest evaluation, checkpoints that survive a kill, and rules for comparing runs fixed before the runs ([src/gp_thee/train.py](src/gp_thee/train.py))
- [x] Baselines without a neural network: character n-grams, bzip2, xz ([docs/baselines.json](docs/baselines.json))
- [x] Tokenizer comparison, by the rule written down in advance: characters win, 1.7546 against 1.8012 for the nearest vocabulary ([docs/results-sweep.json](docs/results-sweep.json))
- [x] The sampler: a prompt normaliser that refuses what this universe cannot spell, and a "speak as a character" wrapper ([src/gp_thee/sampling.py](src/gp_thee/sampling.py))
- [x] Which run is released, by a rule fixed in advance ([docs/release.json](docs/release.json)): GP-Thee-11M is `sweep-char-seed-1`
- [x] Memorisation report: it recites its editors, not Shakespeare — 44 characters of the poet's own words from the training works, 0 from works it never read ([docs/memorisation.json](docs/memorisation.json))
- [x] The final evaluation on the test works, opened once: **1.80 bits per character**, beating every baseline by 0.52 ([docs/final-evaluation.json](docs/final-evaluation.json))
- [ ] Weights on Hugging Face
- [x] Blog: eight parts, one per milestone ([blog/](blog/))

## Layout

```
data/        the corpus: raw/ is the untouched download, processed/ is generated from it
blog/        the write-up, one section per milestone
docs/        PLAN.md (roadmap) and BUILD_LOG.md (the detailed step-by-step record)
scripts/     one-off commands such as the dataset download
src/gp_thee/ the library: tokenizer, data loading, the model, training and evaluation
runs/        training runs land here (only the weights are gitignored: 43-141 MB each; everything else is committed)
tests/       run with `uv run pytest`
```

## Reproduce

```bash
brew install uv                              # the only thing you install by hand
git clone https://github.com/shivamtiwari93/gp-thee && cd gp-thee
uv sync                                      # Python 3.14 + PyTorch 2.14, from the lockfile
uv run python scripts/download_data.py       # confirms the corpus checksum
uv run python scripts/prepare_data.py        # rebuilds data/processed/ from the raw file, byte for byte
uv run python scripts/make_split.py          # re-checks the split: held-out works share no text with training
uv run python scripts/build_tokenizers.py    # fits the tokenizers on the training works; writes data/tokens/
uv run pytest                                # 409 tests, about four minutes
uv run python scripts/baselines.py           # how well Shakespeare can be predicted WITHOUT a neural network
uv run python scripts/check_model.py         # Mac only: the full-size correctness gate, on the GPU
uv run python scripts/benchmark.py           # Mac only: how fast this machine trains (plug it in first)
uv run python scripts/smoke_test_gpu.py      # Mac only: GPU answers match the CPU's
uv run python scripts/count_params.py        # the parameter arithmetic, checked against PyTorch

uv run python scripts/train.py --name my-run --passes 34 --seed 1   # one training run: about 36 minutes on an M5 Max
uv run python scripts/evaluate.py --run my-run                      # its score, by play and by kind of line, beside the baselines
uv run python scripts/check_model.py --trained runs/my-run/best.pt  # the no-peeking check again, on trained weights
uv run python scripts/summarise_runs.py                             # several seeds gathered into a mean and a spread

uv run python scripts/find_length.py --tokenizer bpe-1024           # the run-length rule, as a program: about an hour per tokenizer
uv run python scripts/sweep.py                                      # the tokenizer comparison: 15 runs, about four hours. Commit nothing while it runs
uv run python scripts/summarise_runs.py sweep-                      # its verdict
uv run python scripts/compare_finalists.py                          # the two finalists on the same text
uv run python scripts/where_they_differ.py                          # where the models differ, chunk by chunk

uv run python scripts/sample.py --run my-run --prompt $'\n\nHAMLET.\n'   # ask the model for text
uv run python scripts/sample.py --run my-run --suite                # the ten fixed prompts, written to runs/my-run/
uv run python scripts/choose_release.py                             # which run is GP-Thee-11M, by the rule
uv run python scripts/memorisation.py                               # does it recite its training works?
uv run python scripts/demonstrate.py                                # every example in blog part 8, generated and measured
```

`--device cpu` works for every script that takes a device, about eight times slower.

More steps are added here as they are built.

## Read along

1. [Prerequisites and setup](blog/01-prerequisites-and-setup.md): the starting point, the tools and why, and how we know the model has 10.8 million parameters before training it.
2. [The data](blog/02-the-data.md): finding the corpus, the traps hidden in it, cleaning 0.75% of it, auditing the result twice, and why these five works are held out.
3. [The tokenizer](blog/03-the-tokenizer.md): characters, then word fragments learned from the plays themselves; what the chunk rule costs and buys; how the vocabulary size will be chosen; and how the tests turned out weaker than the code.
4. [The model](blog/04-the-model.md): 195 lines of GPT, the four checks that catch a model that is quietly wrong, the planted bug that halved the context while every test stayed green, and what we learned about training on an Apple GPU.
5. [Training](blog/05-training.md): the six lines that learn, how to score a model honestly, what a zip file says a good score is, why the length of a run has to be chosen by a rule, and a training script whose numbers were right while nearly everything around them was wrong.
6. [Which tokenizer?](blog/06-which-tokenizer.md): five ways of cutting the text, three runs each, a winner chosen by a rule fixed before the first run, what that rule could and could not have shown, and why the word-fragment models lost on a text this small.
7. [Does it recite?](blog/07-does-it-recite.md): the memorisation measurement I nearly published and why it was circular, a scan of all 4.8 million positions, the longest passage the model can be made to reproduce, a bug in my own rule that flattered it, a sampler you can talk to, which of eleven runs becomes GP-Thee-11M, and a promise from part 2 withdrawn.
8. [The final test](blog/08-the-final-test.md): the three works held out since part 2, opened once; the one-way door and the dress rehearsal that earn a held-out score; GP-Thee-11M at 1.80 bits per character on Shakespeare it never read, beating every baseline and reciting none of it; and the four pre-registered questions the test works settled.

## Data and licence

The corpus is Project Gutenberg eBook #100, *The Complete Works of William Shakespeare*, which is in the public domain in the United States. The untouched file is in [data/raw](data/raw/), with the Project Gutenberg notice that accompanies it. This project is not affiliated with Project Gutenberg. Code and weights are released under the [MIT licence](LICENSE).
