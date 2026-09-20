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

Things inside the Gutenberg file that are not Shakespeare and must be cleaned: the marker lines, a title block, a 44-entry table of contents, 38 per-play contents lists, the word `FINIS` (twice), and verse line numbers stuck to the end of 293 lines of *Venus and Adonis*. *(Three of these figures were later corrected by reading the file itself: see Entry 9.)*

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

## Entry 9. Cleaning the corpus (2026-09-20)

**Result:** `scripts/prepare_data.py` turns the raw file into 44 files, one per work, in `data/processed/works/`, plus a manifest. 0.75% of the file is removed. No line of verse or prose is reworded. The rules and the keep decisions are listed in [data/processed/README.md](../data/processed/README.md).

| | Lines | Words | Characters | Distinct characters |
|---|---|---|---|---|
| Raw | 196,022 | 963,478 | 5,359,444 | 100 |
| Cleaned | 194,293 | 957,139 | 5,319,224 | 97 |

### Step 1. Read the file before writing any rule

Eight agents mapped the raw file without changing it: four analysts (work boundaries, editorial clutter, characters and whitespace, duplicated text), each followed by a verifier told to re-measure everything with its own code and to attack the proposed rules. This corrected three things the Entry 2 research had got from a distance:

| Entry 2 said | The file says |
|---|---|
| `FINIS` appears twice | There are three end markers in all: `THE END` once (after the Sonnets), `FINIS` twice. The other 41 works end with nothing but blank lines. |
| 293 lines of *Venus and Adonis* carry a margin number | 294. Line 195658 has a single space before its number, and every pattern proposed so far required two. |
| Strip them with `(?<=\S) {2,}\d+$` | `(?<=\S) +\d+$`, confined to that one poem. The gap runs from 1 to 22 spaces. Some of the numbers are also wrong (one line labelled 448 is line 450), so match the shape, never the value. |

Other traps the map found, each of which would have damaged the corpus silently:

- **Richard II's heading is not its contents entry.** The contents list says `KING RICHARD THE SECOND`; the play is headed `THE LIFE AND DEATH OF KING RICHARD THE SECOND`. The shorter string *does* appear on a line of its own, 41 lines later, as the first entry of the cast list. A splitter that takes the first exact match passes a naive "found all 44" check while starting the play 41 lines late, gluing its opening onto *Pericles*.
- **Blank lines cannot find the works.** 308 runs of four blank lines exist; only 42 sit between works. 252 lines have the exact shape of a work heading; only 45 are.
- **A play's contents list cannot be told from its real headings by appearance.** 85 of the contents lists' `ACT` lines are byte-identical to the real ones. Position is the only reliable rule: from the line `Contents` to the line before `Dramatis Personæ`.
- **Italic markers cannot be paired line by line.** The 9,702 underscores balance perfectly over the whole file, but 264 italic spans cross line boundaries, one of them 29 lines long.
- **Only four pairs of works share any of Shakespeare's own text of 50+ characters** *(wording corrected in Entry 11: counting editor's headings, 23 works share something)*. *The Passionate Pilgrim* with the Sonnets (its poems I and II are Sonnets 138 and 144) and with *Love's Labour's Lost* (three poems); *2 Henry IV* with *Richard II* (one quotation); *Lucrece* with *Venus* (the dedication header). That decides which works can be held out for validation.

### Step 2. The script

Every rule asserts exactly how many lines it touches, the raw file is pinned by SHA-256, and all line numbers are raw line numbers: lines are marked as dropped or edited in place and only assembled at the end, so no rule can shift another's targets. The checksum guarantees the words; the assertions guarantee the structure.

All assertions passed on the first run.

### Step 3. Audit, round one

Four independent auditors, each writing their own tools:

- **Lost text.** A line-by-line alignment of all 196,022 raw lines against the output, confirmed with the system `diff`. Every dropped and edited line matched a stated rule. No text lost, none invented.
- **Structure.** All 44 boundaries exact; every play has one cast list, five acts and gap-free scene numbers; 154 sonnets with the known irregular ones (99 has 15 lines, 126 has 12); *Lucrece* 265 stanzas, *Venus* 199, *A Lover's Complaint* 47. Every play's length within about 5% of commonly cited figures.
- **Mutation testing.** This found the real bug. The script wrote its 44 files *before* running its final twelve checks, so a late failure left a damaged corpus on disk beside a stale manifest, which is exactly what its own docstring promised could not happen. It also found that the heading finder took the *first* match, and a planted duplicate title silently moved 909 lines of *Cymbeline* into *Hamlet*'s file. And three of the script's checks could never fail.
- **Surviving clutter.** Two editor's bracket insertions, ditto marks in the *Julius Caesar* cast list, 2,516 lines with a stray one-space indent that gave `[_Exeunt._]` two spellings, 46 doubled spaces, and a 7-blank-line run the script itself had created beside the duplicate *Venus* title.

Fixes: build everything in memory and write only after every check; require exactly one heading match per title; delete the checks that could not fail; count the 181 blank lines between works as a rule, so every dropped line is accounted for; and a second round of cleaning rules for the surviving clutter, each pinned and counted.

### Step 4. Audit, round two

Because round two changed the output, it was audited again rather than assumed.

- **Alignment:** 191,267 lines kept unchanged, 2,874 edited, 1,881 dropped, 152 blank lines inserted (4 after each play title). 2,857 of the edits change only spaces, tabs or the margin digits. Across all edits the only other characters changed were 33 removed (27 ditto marks, 5 brackets, 1 straight apostrophe) and 223 added, 219 of them the spelled-out ditto descriptions. All *measured*, independently.
- **Mutation testing:** 110 corruptions of a copy of the raw file. Every structural one was caught, and all 107 failing runs left the existing output byte-identical. Content-level changes (a changed word) pass, as they must: no structural check can know Shakespeare's words. That is the checksum's job.
- It also caught stale wording in the script and README, the same kind of error as in Entry 8: sentences that were true before the second round of rules and not after.

The output was byte-identical before and after these last fixes (*measured* by checksum).

### Decisions taken, all reversible by re-running the script

- **Cast lists are kept.** Editors compiled them, not Shakespeare, but they are how a reader meets a play. 0.60% of the corpus.
- **Everything published under his name is kept**, including *The Passionate Pilgrim* (only 5 of its 20 poems are securely his) and six plays widely thought to be collaborations (about 14% of the words). Pruning by attribution needs scene-level tables that scholars still argue over. Instead, none of those works will be used to *measure* the model.
- **Three editor's marks inside lines of verse are kept**, because removing them would mean rewording a line.

### Proposed split (not yet frozen)

Whole works only, chosen from plays that are his alone and share no passage with any other work:

| Set | Works | Share of characters |
|---|---|---|
| Validation | *All's Well That Ends Well*, *Romeo and Juliet* | 5.2% |
| Test | *King John*, *The Tempest*, *A Lover's Complaint* | 4.4% |
| Train | the other 39 | 90.4% |

The model we *measure* never sees *Romeo and Juliet*. The model we *release* will be retrained on all 44 works, for the number of steps the measured run found best.

Lesson: the most valuable hour of this step was spent reading the file, not writing the script. Every trap above is obvious once seen and invisible until then.

## Entry 10. The split is frozen, and three decisions settled (2026-09-20)

**Decisions by the project owner:**

- The proposed split is approved, with the instruction that the blog must explain *why*.
- *The Passionate Pilgrim* stays in the corpus, all twenty poems. The universe is the canon as published under Shakespeare's name.
- The model is named **GP-Thee-11M**. Its 10,758,528 parameters round to 11 million, not 10. The arithmetic in Entry 2d renamed the model.

**The split,** written by `scripts/make_split.py` to `data/processed/split.json`:

| Set | Works | Characters | Share |
|---|---|---|---|
| Validation | *All's Well That Ends Well*, *Romeo and Juliet* | 274,727 | 5.2% |
| Test | *King John*, *The Tempest*, *A Lover's Complaint* | 233,161 | 4.4% |
| Train | the other 39 | 4,811,336 | 90.5% |

Four criteria: every genre covered across the two held-out sets; works that are his alone; works that share no passage with any other work; about 5% each.

**The third criterion is measured, not assumed.** The script reduces each held-out work to lowercase letters and single spaces, then looks for any 50-character run that also occurs in any of the other 43 works. Result (*measured*, 2.5 seconds): four of the five share nothing. *King John* shares one 63-character run with *The Winter's Tale*, which is a scene heading followed by "Enter" (`...exeunt scene ii the same a room of state in the palace enter`). The script accepts that overlap by name and fails on any other. It also fails if a held-out work is ever one of the seven that share text or the seven of doubtful authorship, and re-verifies every file against the manifest's checksums first.

**A claim caught before it shipped.** The first draft of the script's docstring said the last 10% of the file was "the end of *The Two Noble Kinsmen*, all of *The Winter's Tale* and all five narrative poems". That came from an earlier reviewer's inference, labelled as unmeasured at the time, and it was wrong. Measured on the cleaned corpus, the last 10% by characters is 71% of *The Two Gentlemen of Verona*, all of *The Two Noble Kinsmen*, all of *The Winter's Tale* and all five poems:

| Genre | Whole corpus | Last 10% |
|---|---|---|
| Tragedy | 29.1% | 0% |
| History | 27.1% | 0% |
| Comedy | 26.5% | 13.5% |
| Romance | 12.2% | 53.5% |
| Poetry | 5.1% | 33.0% |

The conclusion survived (a last-10% split is badly lopsided here) and got stronger, but the stated facts were wrong. Same lesson as Entry 8: measure before writing the sentence.

**The cost of this split, and how it is paid once.** The measured model never reads *Romeo and Juliet*. The released model, GP-Thee-11M, will be retrained on all 44 works for the number of steps the measured run finds best. Its own score cannot be measured, since nothing is left to measure it on, and the write-up will say so.

Blog part 2, [the data](../blog/02-the-data.md), covers Entries 2a, 9 and 10.

## Entry 11. Blog part 2, and what reviewing it caught (2026-09-20)

Wrote [blog/02-the-data.md](../blog/02-the-data.md) and put it through the same three reviews as part 1: a fact-checker told to re-measure everything, an ML-methodology lens on the split section, and a newcomer reading it cold. 67 issues. Every disputed number was then re-measured first-hand before the post changed.

**The split section claimed more than the split delivers.** This was the section the project owner had specifically asked to be explained well, and it had three real gaps.

1. *"Cover every genre."* Only across the two sets together. Validation has no history, romance or poetry; test has no comedy or tragedy. The post criticised the last-10% split for having no tragedy or history and then had the same kind of gap in its own test set. The honest framing: every genre is graded somewhere; the final exam is on kinds of play that were never tuned on, which makes it stricter; and validation and test are scores on different texts, so they cannot be compared with each other. Scores will be reported per work.
2. *"His alone."* Too strong for two of the five. Laurie Maguire and Emma Smith (Oxford, 2012) argued for Thomas Middleton's hand in *All's Well That Ends Well*, and the New Oxford Shakespeare (2016) accepted the joint attribution. Brian Vickers (2007) attributed *A Lover's Complaint* to John Davies of Hereford, and the RSC Complete Works of 2007 left it out; MacDonald P. Jackson called that omission a mistake. Both checked against sources. Both are disputed, mainstream editions print both works as his, and the split stands, but the post and the script now say so.
3. *The four criteria do not pick five works.* They leave 11 comedies, 9 tragedies, 6 histories, 3 romances and 2 poems. The post never said how the five were chosen from those, and nor did this log. The real reasons, recorded here:
   - **The poem was forced.** The only other eligible poem, *The Phoenix and the Turtle*, is 2,071 characters.
   - ***King John* is the only eligible history that stands alone.** The other five belong to sequences that share characters with plays in training.
   - ***All's Well* and *Romeo and Juliet* have the cleanest record.** The duplicate-text map (Entry 9, step 1) also ran a stricter search: no 40-character run of dialogue and no run of 8 identical words shared with any other work. Eight works passed. Setting aside the collaborations, that is exactly one comedy, one tragedy, two romances and two poems. Validation is consulted most, so it got the two cleanest plays.
   - ***The Tempest* over *Cymbeline*** (both passed) was a judgement call: the shorter one, leaving more text for training.
   - None of the five shares a character with a training play, though names recur: training has another Helena (*A Midsummer Night's Dream*), another Juliet (*Measure for Measure*) and four Antonios (*measured* from speaker labels).

**Facts corrected, each re-measured:**

| The draft said | Measured |
|---|---|
| 308 runs of four or more blank lines | 311 (308 of exactly four, 2 of five, 1 of six); 42 separate works, and the 43rd gap has three |
| Seven works share text with another work | 23 of the 44 share a 50-character run with some other work. Seven share Shakespeare's own words; the other 16 share only an editor's scene heading or cast-list line. Entry 9's "only four pairs" was true of his text, not of the files. |
| The audit raised the authorship question | Entry 2a raised it at the research stage |
| *King John* shares a 63-character run | 63 as stored, including a stray letter and spaces; the phrase itself is 60 characters, and begins with the `Exeunt` of the previous scene |
| The mutation tests corrupted the input | They did, with the checksum test switched off; otherwise every corruption dies on the first line and the other checks are never exercised |
| "Four of the five share nothing at all" | No run of 50 characters. Shorter windows find stock phrases and headings, as they should. |

**Added because a reader would need them:** a definition of the loss; that the test set is never used to make a choice (not merely "looked at once"); the term *data leakage*; that speaker names never seen in training are 2.98% of validation characters (*measured*; a test-set figure was also measured and published here, which was a mistake: see Entry 12), so held-out loss will be higher than tutorial numbers and is not comparable with nanoGPT's 1.47; the caveats owed on retraining the released model on all 44 works (its quality is an expectation, not a measurement; the same number of steps on 10% more text means slightly fewer passes; nothing is left to catch a bad run); two kinds of noise in a small exam, and that only one of them is fixed by repeating runs.

**The alphabet closes part 1's loop.** Cleaning removed three characters (asterisk, tab, straight apostrophe), so the vocabulary is 97, not 100, and the character-level model has 10,757,376 parameters, not 10,758,528 (`scripts/count_params.py`, checked against PyTorch). Still GP-Thee-11M. All 97 occur in the 39 training works; validation uses 72 (*measured*; the test figure that stood here has been removed, see Entry 12). The 23 characters seen fewer than 50 times total 490 occurrences, 0.009% of the text.

Lesson, third time: the prose is where the errors are. The scripts had assertions and audits. The sentences about them had only the author. Treat a claim in the write-up like a line of code: it does not ship until something has tried to break it.

## Entry 12. The tokenizers (2026-09-20)

**Result:** `src/gp_thee/tokenizer.py`, the project's first library code, holds a character tokenizer and a byte-pair-encoding (BPE) tokenizer written from scratch. `scripts/build_tokenizers.py` fits them on the 39 training works and saves five of them in `data/tokenizers/`. 48 tests.

| Tokenizer | Vocabulary | Training tokens | Characters per token (train / validation) | Context of 256 tokens | Merged pieces seen 100+ times | Parameters |
|---|---|---|---|---|---|---|
| char | 98 | 4,811,375 | 1.00 / 1.00 | 256 chars | n/a | 10,757,760 |
| bpe-1024 | 1,024 | 1,936,643 | 2.48 / 2.49 | 636 | 98.3% | 11,113,344 |
| bpe-1536 | 1,536 | 1,787,863 | 2.69 / 2.66 | 689 | 95.0% | 11,309,952 |
| bpe-2048 | 2,048 | 1,694,913 | 2.84 / 2.79 | 727 | 91.2% | 11,506,560 |
| bpe-4096 | 4,096 | 1,507,176 | 3.19 / 3.07 | 817 | 50.6% | 12,292,992 |

All *measured*. Learning all 3,998 merges takes 5.7 seconds; the whole build, 10.

### Design

- **No unknown token.** `decode(encode(text)) == text` for any text over the alphabet. A character outside it is an error, and so is an id outside the vocabulary.
- **Chunks.** Merges are learned and applied inside chunks only: a word with the space before it, a run of punctuation, a run of newlines, indentation. The apostrophe counts as a letter, because here it nearly always is one (`’tis`, `o’er`, `lov’d`). Measured on the training works: of 23,019 apostrophes, well under 1% are closing quotation marks glued to a word (the auditor counted 75; a cruder upper bound of mine gives 148).
- **START**, one special token placed before every work. It cannot be typed (the alphabet has no `<`, `|` or `>`), so `encode` can never produce it. It makes the character vocabulary 98, not 97, and the model 10,757,760 parameters, 384 more than blog part 2 said. Still GP-Thee-11M.
- **One learning run, several sizes.** Merges are learned most-frequent-first, so the first 926 merges of bpe-4096 are bpe-1024. Tested.
- **Fitted on the training works only.** The test that proves it re-learns the merges from the training works and demands an exact match. It is sensitive: fitting on the 39 training works plus the 2 validation works changes merge number 33, 3,848 of the 3,998 positions, and 169 pieces (*measured*). *(This line first quoted a refit on all 44 works, which meant fitting on the test works. See "The test works, a third time" below.)*

### What the chunk rule buys, measured

The first comment I wrote to justify the chunk rule was wrong: it claimed that without the rule the most frequent pairs would be things like `"e " + "t"`. The auditor trained a tokenizer with no chunk rule and found that pair is never merged. `scripts/compare_chunk_rule.py` now reproduces the comparison (2,048 entries, training works, *measured*):

| | No chunk rule | With the rule |
|---|---|---|
| Characters per token | 3.03 | 2.84 |
| Pieces that straddle two words | 237 (12.2%) | 0 |
| Pieces mixing a newline with text | 204 | 0 |
| Ways a common word gets cut up (average over the 150 most common) | 12.4 | 2.0 |
| `come` | 45 ways | 2 |

By compression, and by the "seen 100+ times" figure, having no rule *wins*. What the rule buys is that a word looks the same to the model wherever it appears. Compression is not the objective.

### The audit

Four independent auditors, as for the cleaning script.

- **Independent reimplementation.** A naive trainer written from the description alone, recounting everything before every merge, reproduces all 3,998 merges exactly and in order. The replay in `encode` matches the training-time segmentation for all 38,398 distinct training chunks at every size. The two shortcuts in `learn_merges` are sound, including the index that goes stale (two thirds of its entries by the end, harmlessly).
- **Fuzzing.** 658,544 strings, including every string of up to three character classes, with no failure.
- **Leakage.** A train-only refit reproduces the saved files byte for byte. Replacing all five held-out works with unrelated text leaves the tokenizer files byte-identical.
- **Mutation testing of my tests** found them weaker than they looked. An encoder that ignored every merge after the first 1,500, or that sprinkled stray START ids, passed all 28. Every round trip decoded with START hidden, and nothing pinned the actual segmentation. Now fixed: round trips decode with START shown, one test pins the exact cuts of a known line, token counts are checked against the published report, a naive reference trainer lives in the tests, and saved files are compared byte for byte with what the code writes.

Two things the audit measured that are worth knowing:

- **Ties decide three merges in four** (2,968 of 3,998). Inside a name such as `SYRACUSE`, every neighbouring pair is exactly as frequent as the name. So the last entries of each vocabulary size are an arbitrary pick among equally frequent pairs. Flipping the tie rule changes 130 of 3,998 pieces and 4 tokens in 1.5 million.
- **Speaker names are 13% of bpe-2048 and 16% of bpe-4096**, and they explain the whole gap between training and validation compression. `HAMLET` is one token. `ROMEO`, which the training works never contain, is `RO|M|E|O` at every size. So when models are compared across tokenizers, bits per character will be reported separately for speaker-label lines. Otherwise the comparison is partly a verdict on how each size spells names it has never seen, which is a property of our split and not of the model.

### A mistake of mine: I looked at the test set

Blog part 2 and Entry 11 published two statistics measured on the test works (the share of their characters in speaker names never seen in training, and how many distinct characters they use), and named two test-work characters as unseen names. Nothing was tuned on those numbers and no model existed. But the rule is that the test works are not examined, and the first of those figures measures exactly what separates the tokenizer candidates. The leakage auditor caught it. The figures are removed from the blog and struck from Entry 11; validation figures carry the argument alone.

Two changes make a repeat harder: the build script no longer saves the test works' token streams (a file's size gives away its token count, and a careless `*.npy` would sweep it into training), and `data/tokenizers/report.json` is tested to contain nothing about the test works.

Also recorded, for honesty: the candidate vocabulary sizes date from the research stage, before the split existed, and came from counts over the whole corpus. The chunk rule was developed by reading training works only. The choice among sizes will be made on validation only.

### Decided now, for later steps

- **If a held-out work ever contains a character the training works lack**, the remedy is a corpus-wide cleaning rule or a split redrawn before any model is trained. The alphabet is never widened from held-out text.
- **Bits per character** = total loss in nats over every target, divided by ln 2 and by the set's character count. Identical for every tokenizer. Per work, and separately for speaker-label lines.
- **Sampling** needs a prompt normaliser outside the tokenizer: straight quotes raise an error today, and a prompt that ends in a space puts the model somewhere it has almost never been, because a space belongs to the *next* token.
- **bpe-1536** was added to the sweep: it is exactly where 95% of merged pieces are still seen 100 times in training, and the original sizes jumped straight over it.

Lesson: my tests passed, and the tests were the weak part. Mutation testing (break the code on purpose, see whether a test notices) found in minutes what re-reading never would have.

### Addendum to Entry 12: what reviewing blog part 3 caught (2026-09-20)

Blog part 3 went through the usual three reviews (fact-check, NLP lens, newcomer). 79 issues. Everything below was re-measured first-hand, on training and validation works only.

**The test works, a third time.** The first draft of part 3, a few paragraphs after confessing to the part 2 mistake, did it twice more. (1) To show the leak test is sensitive, I fitted a tokenizer on all 44 works, that is, on the test works, and quoted two numbers that depend on their text; the same numbers were in this log and in a test comment. (2) I quoted a whole-corpus apostrophe count near the training-works count, so the test works' share fell out by subtraction. Nothing was influenced and no model exists. But a rule kept by intention had now failed three times, so it is enforced in code:

- `src/gp_thee/data.py`: every script and test loads works through `load_works(set)`. Asking for `"test"` raises `PermissionError` unless the caller passes the phrase that says this is the final evaluation.
- `scripts/build_tokenizers.py` no longer opens the test works at all. The alphabet of the whole corpus was recorded in `manifest.json` at cleaning time, before the split existed; the script checks the training works contain all of it, and the tests establish that any string over that alphabet round-trips. So every work is provably encodable, unread.
- The sensitivity check uses the validation works as the stand-in leak (`test_a_leak_would_be_noticed`).
- (`scripts/make_split.py` still reads all 44 works, because comparing held-out works with the rest is what defines the split. It reports nothing about them except overlaps.)

**Other corrections, each measured:**

| The draft said | Measured |
|---|---|
| The Gowda and May rule "lands at about 1,536" | Counted the paper's way, over every vocabulary entry, it lands at 1,200 (95.00%). It is 1,536 (94.99%) only when counting merged pieces, which is defensible (25 rare characters and START can never reach 100) but has to be said. |
| The obvious BPE "takes a minute or two" | That is the speed of a version that already counts distinct chunks. Recounting the running text takes 0.80 s per merge in plain Python, about 53 minutes for 3,998. Shortcut one: an hour to a couple of minutes. Shortcut two: minutes to under 6 seconds. |
| Ties come from names like `SYRACUSE` | Mostly arithmetic. The first tie is at merge 172; 8% of the first 500 merges are ties, 23% of the first 926, 78% of 927 to 1,950, and 96% of 1,951 to 3,998, where winning counts are small (97,647 at merge 1; 389 at 926; 149 at 1,950; 58 at 3,998) and thousands of pairs are in the running. |
| "Nobody told it what a word is" | The chunk rule tells it exactly where a word begins and ends. |
| The chunk rule "is worth it" | A bet, not a result. The no-rule tokenizer also wins on validation compression (3.00 vs 2.79) and on merged pieces seen 100+ times (96.6% vs 91.2%). "Ways a word gets cut" is a test the rule cannot fail, and the four commonest cuts of `come` cover 75% of its occurrences. |
| `HAMLET` is one token | At 1,536 and above. At 1,024 it is `HAM` + `LET` (merge 982; that vocabulary stops at 926). `RO` exists because of `ROSALIND`, `RODERIGO`, `ROSENCRANTZ`. |

**Not logged at the time, logged now:** my first guard on START refused any alphabet containing *any* character of `<|start|>`, including `s`, `t`, `a`, `r`, and so rejected Shakespeare's own alphabet. The build script failed on its first run. The guard only needs START to be unspellable, which `<`, `|` and `>` guarantee.

**Also measured for the post:** 83.7% of words in the training works have a space before them and 73.2% a space after, which is why the space goes at the front of a piece. The training text uses 99 of the 256 byte values and `’` is three bytes, which is why we did not build on bytes. 79.3% of single blank lines are followed by a speaker label. bpe-2048 produces 3,471 lone-space tokens in 1.69 million.

**Pre-registered, before any model exists: how the tokenizer will be chosen.**

1. *The measure:* validation bits per character over all the text = total loss in nats, divided by ln 2 and by the character count.
2. *Training length:* tokenizers are not compared at a fixed step count, because a bigger vocabulary makes the corpus shorter (4.8M tokens as characters, 1.5M at 4,096), so equal steps are unequal passes. Each run is compared at its best validation checkpoint.
3. *The decision:* the mean of three runs from different random seeds. If two tokenizers are closer than the spread between runs, the smaller vocabulary wins.
4. *Reported but not deciding:* the score per play, and separately for speaker-label lines and everything else.

Tests: 51.
