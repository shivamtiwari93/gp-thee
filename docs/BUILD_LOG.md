# Build log

A chronological record of how GP-Thee was built: what was done, the exact commands, the numbers, the decisions and the reasons, and the mistakes. Newest entries are at the bottom.

Numbers marked *measured* were measured directly. Numbers marked *estimate* were extrapolated and have not been checked on this machine.

---

## Entry 0. The premise (2026-09-20)

Build a language model from scratch where the complete works of Shakespeare are the only text that exists. No pretrained weights, no pretrained tokenizer, no outside text.

Why this is a good learning project: the corpus is about 5 MB, so every experiment takes minutes on a laptop, and the whole pipeline (data, tokenizer, model, training, evaluation, sampling) is small enough to read end to end.

## Entry 1. The hardware (2026-09-20)

Read with `system_profiler SPHardwareDataType SPDisplaysDataType` and `sysctl`.

| | |
|---|---|
| Machine | MacBook Pro (Mac17,7), Apple M5 Max |
| CPU | 18 cores (6 super + 12 performance) |
| GPU | 40 cores, Metal 4 |
| Memory | 128 GB unified (shared by CPU and GPU) |
| Free disk | about 1.6 TB |
| OS | macOS 26.5.2 |
| Python | 3.14.6 (Homebrew); no ML libraries installed |

A 10-million-parameter model with its optimizer state needs well under 1 GB. The machine is not the constraint. The size of the corpus is. *(The first sentence is true but misleading: a training step also holds 1 to 5 GB of intermediate results. Corrected in Entry 8.)*

## Entry 2. Research (2026-09-20)

Before writing code, four questions were researched in parallel, each by one agent, and each agent's findings were then attacked by a separate fact-checking agent. A final agent looked for gaps and contradictions. Nine agents in total. The corrections below came from that checking step, which is why they are worth recording.

### 2a. Is there a dataset?

Yes. **Project Gutenberg eBook #100**, one plain-text file.

- 5,422,721 bytes; about 963,000 words; 5.36 million characters; 100 distinct characters (*measured*).
- 44 works: the Sonnets (all 154 counted), 38 plays including *Pericles* and *The Two Noble Kinsmen*, and 5 poems.
- Public domain in the United States.
- Structure is preserved: `ACT` and `SCENE` headings, a cast list per play, speaker names in capitals on their own line (`HAMLET.`), stage directions in brackets (`[_Exit._]`).

Alternatives that were checked and rejected:

| Source | Why not |
|---|---|
| Karpathy's "tiny shakespeare" (1.1 MB) | About 21% of the works. No sonnets, no *Hamlet*, no *Macbeth*. Useful only as a quick test file. |
| Hugging Face "complete works" datasets | Each had a defect: shuffled rows with wrong labels, a 1990s edition with a copyright notice embedded 221 times, 36 plays only, or 80-character windows. |
| Folger Shakespeare | Cleaner text, but licensed CC BY-NC (non-commercial), and missing two poems. Wrong choice for a project that will be published. |
| Kaggle "Shakespeare plays" CSV | 36 plays, no poems, unknown licence, 6,237 stage directions attributed to the wrong speaker. |

Things inside the Gutenberg file that are not Shakespeare and must be cleaned: the marker lines, a title block, a 44-entry table of contents, 38 per-play contents lists, the word `FINIS` (twice), and verse line numbers stuck to the end of 293 lines of *Venus and Adonis*.

**A bug caught before it happened.** The first proposed regex for those line numbers was `` {2,}\d+$``. The fact-checker ran it and found it matches 447 lines, not 293: it also deletes all 154 sonnet numbers. The correct pattern is `(?<=\S) {2,}\d+$`. Lesson: cleaning scripts need assertions on how many lines each rule touches.

**An authorship wrinkle.** *The Passionate Pilgrim* was published under Shakespeare's name, but only about five of its twenty poems are thought to be his. Two of its poems are also Sonnets 138 and 144, and three come from *Love's Labour's Lost*. So those works must stay together on the same side of the train/validation split, or held-out text leaks into training.

### 2b. Which software?

**PyTorch on the Mac GPU backend (MPS)**, in a `uv`-managed environment. Both PyTorch 2.14 and Apple's MLX 0.32 publish Python 3.14 wheels for this Mac (*verified on PyPI*), so no second Python is needed.

Why PyTorch first: at this scale both frameworks finish in minutes, so speed does not decide it. PyTorch has the canonical line-by-line reference for this exact task (nanoGPT), is easy to debug, and the skills carry over to other hardware. MLX is the planned second pass.

Corrections from fact-checking:

- A widely quoted "MLX is 2.1x faster than PyTorch" benchmark dates from mid-2024 (MLX 0.14 against PyTorch 2.3). Which is faster on an M5 Max today is unknown.
- *(This bullet is partly wrong; see Entry 8 for the verified list.)* Known PyTorch MPS bugs on M5 chips (non-deterministic half-precision matrix multiply, fixed in 2.13; corrupted attention output on macOS 26, fixed in 2.14) mean: pin `torch==2.14.0`, and check CPU against GPU results on one batch before trusting a long run.

Expected time per baseline training run on this machine: about 10 to 30 minutes (*estimate*, extrapolated from a published M3 Max run of 30 to 37 minutes).

### 2c. How big should the model be?

About **10 million parameters**: 6 layers, 6 attention heads, width 384, context of 256 tokens. This is the nanoGPT "baby GPT" shape.

The reasoning: the corpus is tiny, so the risk is memorisation, not lack of capacity. Research on training with repeated data (Muennighoff et al., 2023) supports models far larger than the classic "20 tokens per parameter" rule when you train for many passes with dropout and weight decay, which lands at roughly 5 to 12 million parameters for 5 million characters. Published replications of this shape on the 1 MB subset show validation loss bottoming out early and then getting worse, so the best checkpoint must be kept and training stopped early.

Open points the research did not settle, to be answered by experiment: the best tokenizer vocabulary size (candidates: characters, 1024, 2048, 4096), how many passes over the data help, and whether a 39-million-parameter model gains anything.

### 2d. How do we know it is 10 million before training?

Because the parameter count is arithmetic on the architecture. Nothing about it is learned. For a GPT with width `d`, `L` layers, vocabulary `V` and context `T`:

- Attention, per layer: `4·d²` (three `d×d` matrices for queries, keys and values, one for the output).
- Feed-forward, per layer: `8·d²` (`d → 4d → d`).
- So each layer has about `12·d²` weights.
- Token embeddings: `V·d` (shared with the output layer). Position embeddings: `T·d`.

With `d = 384`, `L = 6`:

```
12 × 384² × 6      = 10,616,832   transformer layers
13 × 384           =      4,992   layer-norm weights (2 per layer + 1 final)
100 × 384          =     38,400   token embeddings (100 characters)
256 × 384          =     98,304   position embeddings
                     ----------
                     10,758,528   ≈ 10.8M
```

Training changes the *values* of these numbers, never how many there are. A BPE tokenizer with 2048 entries would add `2048 × 384 ≈ 0.8M`. So the repo is named `gp-thee`, and each released checkpoint is named by its real count, for example `GP-Thee-10M`. The code will print the count, and a test will check it against this calculation.

## Entry 3. Scaffold (2026-09-20)

```bash
brew install uv                       # uv 0.12.17
cd ~/Documents/GitHub
uv init --package --name gp-thee --python 3.14 --vcs git gp-thee
```

Why `uv` with its own Python rather than Homebrew's: Homebrew will move to Python 3.15, which would break an environment built on it mid-project. `uv` pins the interpreter and writes a lockfile, so the environment is reproducible.

Decisions:

- **Raw data is not committed.** The raw file carries Project Gutenberg marker lines and the name is trademarked. A download script re-creates it and verifies its SHA-256. The cleaned corpus, which is only public-domain Shakespeare, will be committed. *(Reversed the same day; see Entry 6.)*
- **Weights are not committed.** They go on Hugging Face.
- **MIT licence** for code and weights.
- **Public from the first commit**, so the history itself is part of the record.

## Entry 4. The one download (2026-09-20)

```bash
uv run python scripts/download_data.py
```

```
Fetching https://www.gutenberg.org/files/100/100-0.txt
  5,422,721 bytes, sha256 a023115c2d4e2ee12221bdd780fdf2ac5a864fe225948656f51f8be462c7fffb
Saved (read-only): data/raw/100-0.txt
```

Checked independently with `shasum -a 256` and `wc`: same hash, 196,022 lines, 963,478 words, 5,422,721 bytes (*measured*). The first and last lines are the Gutenberg `*** START` and `*** END` markers, as expected. `git status` shows nothing, so the ignore rule works.

The script refuses to save a file whose checksum does not match. That matters because a blocked request can return an HTML error page with a success status, and a silently wrong corpus would poison everything after it.

## Entry 5. PyTorch and the Mac GPU (2026-09-20)

```bash
uv add "torch==2.14.0" numpy
```

The version is pinned on purpose. Entry 2b lists two GPU bugs on M5 chips that were fixed in 2.13 and 2.14, so an accidental upgrade or downgrade mid-project could change results.

Smoke test on the GPU backend (`mps`), all *measured*:

| Check | Result |
|---|---|
| `torch.backends.mps.is_available()` | `True` |
| 2048×2048 fp32 matrix multiply, GPU vs CPU | bit-identical |
| Same multiply vs a float64 reference | max error 5.4e-4 on values averaging 36 (normal for fp32) |
| LayerNorm → Linear → GELU, GPU vs CPU | max difference 1.9e-6 (normal float noise) |
| 50 of those multiplies on the GPU | 69 ms |

**A check on the check.** The first version of this test printed a GPU-vs-CPU difference of exactly `0.0`. Floating-point results from two different devices almost never match exactly, so that looked like a broken test (for example, comparing a tensor with itself). It was re-run against a float64 reference and with a second group of operations. The multiply really is bit-identical on this machine, and the second path shows the small differences you would expect. Worth the two minutes: a test that cannot fail tells you nothing.

This is only a smoke test. The full CPU-vs-GPU parity check on a real training batch comes with the model, in step 6 of the plan.

## Entry 6. A decision reversed: the raw corpus goes in the repo (2026-09-20)

Entry 3 kept the raw file out of git because it carries Project Gutenberg's marker lines and the name is a trademark. That was too cautious, and it cost the project something real. The challenge that prompted the rethink: *why hide the dataset in an educational project? People should be able to see exactly what the model was trained on.*

Reasons to commit it:

- **Transparency.** The whole premise is "this file is the entire universe". A reader should be able to open that universe in one click.
- **Reproducibility.** Project Gutenberg revises its files. Their server reports that this master file was last modified on 2025-08-24, and the auto-generated reader copy of the same eBook on 2026-09-01. When the master copy next changes, the pinned checksum stops matching and the download script can no longer rebuild this project. A committed copy cannot drift.
- **It is 5.4 MB.** Git handles that without any special tooling.

What the licence actually asks: the Project Gutenberg License allows free redistribution of their eBooks. If the name "Project Gutenberg" stays on the file, a specific notice with a link to the full licence must accompany it. So the file is committed byte-for-byte unchanged, and `data/raw/README.md` carries that notice, a link to the licence, and a statement that this project is not affiliated with Project Gutenberg. (This is a good-faith reading of the licence by a non-lawyer.)

Changes: removed the `data/raw/*` rule from `.gitignore`; added `data/raw/README.md`; `scripts/download_data.py` now mainly verifies the committed copy and only fetches if the file is missing.

Lesson: "is this allowed?" and "is this the cautious default?" are different questions. The first one has an answer you can look up.

## Entry 7. The parameter arithmetic, checked against PyTorch (2026-09-20)

Entry 2d computed the model size by hand. To make sure the formula is right, a skeleton with the same shape (embeddings, 6 blocks of attention and feed-forward, layer norms, tied output layer, no biases) was built in PyTorch and its parameters counted with `sum(p.numel() for p in model.parameters())`. Run it with `uv run python scripts/count_params.py`. All *measured*:

| Configuration | PyTorch count | Hand formula | Match |
|---|---|---|---|
| 6 layers, width 384, 100 characters, context 256 | 10,758,528 | 10,758,528 | yes |
| same, with a 2048-entry BPE vocabulary | 11,506,560 | 11,506,560 | yes |
| smaller: 6 layers, width 256 | 4,813,056 | 4,813,056 | yes |
| stretch: 12 layers, width 512, vocabulary 2048, context 512 | 39,072,256 | 39,072,256 | yes |

The formula: `12·d²·L + (2L+1)·d + V·d + T·d`. It is exact here because the design has no bias terms and the output layer shares its weights with the token embedding. nanoGPT reports 10.65M for this shape because it uses a 65-character vocabulary and leaves position embeddings out of its headline count: `10,616,832 + 4,992 + 65×384 = 10,646,784`.

## Entry 8. Blog part 1, and what reviewing it caught (2026-09-20)

Wrote [blog/01-prerequisites-and-setup.md](../blog/01-prerequisites-and-setup.md), then had it reviewed by three independent agents: a fact-checker, a senior-ML-engineer lens, and a newcomer reading it cold. 67 issues came back. Most were missing definitions. Five were real errors, and every one of them was then checked first-hand before the post was changed.

**1. "Everything needed to train it fits in under 1 GB." Wrong by about five times.** The arithmetic for weights, gradients and the two AdamW averages is right: 4 copies × 43 MB = 172 MB (*measured*: 174 MB). But a training step also keeps every intermediate result of the forward pass until the backward pass has used it, and that dominates. A rough probe of this exact shape (fp32, dropout 0.2, AdamW, PyTorch's fused attention), all *measured*:

| Batch | Live after the forward pass | Allocated by the GPU driver |
|---|---|---|
| 16 × 256 | 1.3 GB | 2.3 GB |
| 64 × 256 (nanoGPT's batch) | 4.8 GB | 5.5 GB |

Still trivial on 128 GB, but the claim was wrong, and it was wrong in Entry 1 too.

**2. The PyTorch bug list was partly wrong.** Entry 2b cited "corrupted attention output on macOS 26, fixed in 2.14". That is PR [#191794](https://github.com/pytorch/pytorch/pull/191794), which repaired a feature that had landed in the same development cycle, so it probably never shipped in a release. The verified list, read directly from the issue tracker:

| Issue | What | Affected |
|---|---|---|
| [#193487](https://github.com/pytorch/pytorch/issues/193487) | fp32 matrix multiply with a transposed left side silently wrong by 10 to 30%, depending on allocator state | 2.7 to 2.13.0; masked, not root-fixed, in 2.14; still open |
| [#195910](https://github.com/pytorch/pytorch/issues/195910) | causal attention in fp16/bf16 lets 3 of every 4 positions see future tokens | 2.12.1 and earlier; fixed in 2.13 |
| [#180776](https://github.com/pytorch/pytorch/issues/180776) | `F.linear` in fp16/bf16 gives different results on repeated calls, M5 chips | workaround in 2.13 |

These are better reasons for pinning 2.14.0 than the ones originally given, and the first two hit exactly what a GPT does.

**3. "A 2048-entry vocabulary adds 786,432 parameters."** It *replaces* the 38,400-parameter character table, so the net increase is 748,032. The prose disagreed with its own table.

**4. "`uv sync` gets the same Python."** Only to the minor version. `.python-version` says `3.14`; the patch release (we ran 3.14.7) is not pinned.

**5. "Floating-point arithmetic on two devices almost never agrees to the last bit."** Too broad. Single operations are rounded identically everywhere. Long sums differ when the order of addition differs.

New scripts that came out of this:

- `scripts/smoke_test_gpu.py` replaces the interactive test from Entry 5. It adds the transposed multiply from #193487, causal attention in the model's real shape, a future-leak test (change the last token; no earlier position may change), and a control that must show a difference, so the comparison is proven able to fail. All pass (*measured*): multiplies bit-identical; LayerNorm/Linear/GELU differ by 1.7e-6; attention by 7.2e-7; leak exactly 0; control 0.16; 50 multiplies take 68 ms on the GPU and 479 ms on the CPU.
- `scripts/explain_bit_identical.py` answers why the multiply matches bit for bit. Recomputing 60 outputs one product at a time, in index order, with a fused multiply-add, matches both devices 60 out of 60. Reverse order matches 1 of 60; multiply-then-add matches 5 of 60 (*measured*). So Apple's CPU library and GPU kernel add in the same order with the same instruction.
- `scripts/count_params.py`: the "larger" configuration is now 12 layers, width 512 with the *same* tokenizer and context, 37,943,808 parameters. The earlier 39M row also changed vocabulary and context, which would have confounded the size experiment.

Also checked: of PyTorch's nine dependencies, `import torch` loads only `typing-extensions`; one training step adds `sympy` and `mpmath`; the other six stay idle (*measured*).

**A naming point to settle at release.** 10,758,528 rounds to 11M, not 10M. `GP-Thee-10M` is kept as a round label for now, with the exact count to go on the model card. The final count depends on the tokenizer chosen, so the name is decided when the model is.

Lesson: the errors were not in the code. They were in sentences written from memory of what "should" be true. Every one of them was cheap to check.
