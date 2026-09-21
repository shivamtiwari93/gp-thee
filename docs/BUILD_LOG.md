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

## Entry 13. The model, its correctness gate, and a first benchmark (2026-09-20)

**Result:** `src/gp_thee/model.py` is a plain GPT in the nanoGPT arrangement: learned position table, a norm before each step, residual additions, no bias terms, the output layer tied to the token table, GPT-2's starting values. 6 layers, 6 heads, width 384, context 256. With the character tokenizer it has 10,757,760 parameters, the number derived by hand in blog parts 1 and 3, and it refuses to build if the count disagrees. `src/gp_thee/data.py` gained `load_tokens` and `random_batch`. 104 tests. `scripts/check_model.py` is the full-size gate on the GPU: 36 checks, 37 seconds.

### The gate (all *measured*, M5 Max, PyTorch 2.14.0)

| Check | Result |
|---|---|
| Size | 10,757,760 (characters) and 11,506,560 (bpe-2048), and the output table *is* the input table |
| First loss, random tokens | 4.674 against a predicted 4.662; 7.700 against 7.701. The prediction is ln(V) + 0.02² × width / 2: scores with a spread of 0.02 × √384 cost 0.077 nats |
| No peeking | 0.0 change in the past at all 255 cut points, in all 8 combinations of attention (hand-written, built-in), number format (32-bit, 16-bit) and mode (gradients off, gradients on) |
| Two attentions agree | to 2.1e-6 |
| GPU = CPU | predictions to 2.3e-6, gradients to 7.3e-7 of their size; with memorised weights 1.8e-5 and 2.6e-6 |
| Memorises one batch | loss 4.64 → 0.0054 in 200 steps, 2 seconds |
| Same loss everywhere | with memorised weights, 32-bit results agree to 4e-9 (memorised rows) and 1e-6 (unseen rows) across both attentions, CPU and GPU; 16-bit within 1.4e-4 |

### The audit

Four independent auditors. **The model is correct:** a NumPy float64 reference written from the description alone matches it to 3.8e-15 on predictions at full size on real text; gradients match finite differences at 1,548 sampled entries across every tensor; the tied table's gradient is exactly the sum of its two uses; measured starting values match the recipe; none of the three known PyTorch MPS bugs (Entry 8) reproduces at full size. They also confirmed by measurement the comment about why the residual-writing matrices start smaller: with the 1/√(2 × layers) factor the variance of x stays near 0.018 from 1 to 48 layers, and without it grows from 0.03 to 2.26.

**The safety net was the weak part, again.** Of 61 single faults planted in a copy of the code, 17 passed all 26 tests *and* the whole gate. The worst:

- **Heads split and merged without the transpose.** Still perfectly causal, so no-peeking passed; first loss fine; memorised fine. But in each attention layer the last position could read only positions 213 to 255, 43 of 256. *(Corrected in the addendum below: across six layers the reach is 128 positions, half the window, not 43.)* Invisible because both attention paths share that code, and every attention test compared the two paths *with each other*.
- **`load_tokens("train")` returning the validation stream.** All 77 tests passed. Nothing compared a saved stream with the works it claims to be.
- **Anything that only goes wrong on text shorter than the context.** Every test ran at full length. Prompts will be short.
- **Hand-written attention dropping values during evaluation.** The one dropout test covered one path and asked only whether anything at all was random.
- **A numeric fault that exists only in 16-bit.** The gate checked 16-bit for leaks, never for answers.

Three of the gate's own checks were faulty. The memorise check used a learning rate of 1e-3 and failed the *correct* model for 4 or 5 of 10 batches, so its "catches" were partly luck. The first-loss check asserted `loss >= ln(V)` on real text, which is a theorem only for uniformly random tokens: the correct model failed the check for 6 of 40 seeds (5 by falling below ln(V), 1 by exceeding the upper limit), and passed only because the seed was fixed. And the size check repeated a comparison the constructor had already made, so it could not fail.

**Fixes.** A second implementation of the whole forward pass in NumPy, 64-bit, sharing no code with the model, compared on weights pushed well away from their starting values (`tests/test_reference.py`); attention checked against its definition one position at a time, on a short text; short-text-equals-start-of-long-text; every dropout site tested alone, on both paths; GELU, the block, the starting values and the position table each pinned to their definitions; saved streams compared with the works; a test that no code but the loader opens the works. In the gate: the no-peeking check now sweeps every cut point, with gradients on as well as off; first loss is checked against theory on random tokens; the memorise check uses 3e-4; a new check 7 compares answers across every combination of attention, number format and device with sharp, memorised weights; tolerances tightened from 1e-3 to 1e-4; and the gate runs the unit tests first.

Then I planted 34 faults myself, including every earlier survivor: the 31 that can show up on a small model are all caught by the unit tests, and the 3 that exist only in 16-bit on the GPU are caught by the gate (one of them by the GPU unit tests the gate runs first). *(This line first said 32; see the addendum.)* (My first version of one mutant, the "parallel block", simplified back to the correct code and so "survived". A mutant has to be a real fault.)

### Three things the audit found that I did not know

1. **PyTorch's built-in attention is not fused during training on Apple GPUs.** With gradients on, `scaled_dot_product_attention` runs the same separate steps as the hand-written path (a matrix multiply, a softmax, another multiply). The fast fused kernel is used only with gradients off. So the original no-peeking check, which ran with gradients off, never tested the code that training runs. It also explains the benchmark: the two attentions take the same time and memory in 32-bit, and in 16-bit the built-in one is *slower*, because it quietly does its attention in 32-bit. **Decision: train with the hand-written attention.** Same speed or better, the same code runs in training, evaluation and sampling, and it is the code the blog teaches. The built-in one stays as a cross-check. The flag is renamed `builtin_attention`, default off.
2. **Two runs from the same seed are not bit-identical on this GPU.** The only cause is the backward pass of the token-table lookup (`nn.Embedding`), which adds up gradients in a varying order; every other operation, dropout included, is repeatable. The difference starts at 2e-7 of one tensor's gradient and grows: by step 300 two same-seed runs differ by 0.013 to 0.016 nats of validation loss. Writing the lookup as a one-hot matrix multiply would make runs bit-identical. **Decision: keep the ordinary lookup**, say plainly that a seed fixes the starting weights, the batches and the dropout masks but not the last digits, and report the same-seed spread next to the between-seed spread.
3. **16-bit is not yet shown to learn the same way.** Over 300 steps it trails 32-bit by 0.02 to 0.03 nats of validation loss, about twice the same-seed spread. **Decision: the first real runs are 32-bit.** At 13 minutes per 5,000 steps the saving is not worth an unexplained difference.

### Benchmark (first run, on battery power)

Batch 64 × 256 = 16,384 tokens per step, one full training step, *measured*:

| | ms per step | Tokens per second | 5,000 steps |
|---|---|---|---|
| GPU, 32-bit | 158 to 165 | about 100,000 | 13 to 14 minutes |
| GPU, 16-bit, hand-written attention | 96 | 171,000 | 8 minutes |
| GPU, 16-bit, built-in attention | 105 | 156,000 | 9 minutes |
| CPU, 32-bit | 1,795 | 9,100 | 2.5 hours |

An auditor re-timed the four GPU rows in fresh processes and got 160.6, 94.1, 105.7 and 160.0 ms. The estimate in Entry 2 (10 to 30 minutes per run) was right at its fast end. Memory in use at the peak, measured properly by the auditor: 4.79 GB in 32-bit, 3.33 GB in 16-bit with hand-written attention. (The script's first memory column used a driver figure that moves in 1 GiB steps and reported both as equal.)

The benchmark script has since been rewritten: three interleaved rounds with the median and range, the power source recorded, memory measured at its peak, the optimizer's decay groups as training will have them, and the 32-bit against 16-bit comparison run for 300 steps with a learning-rate warm-up (without one, a rate of 1e-3 from the first step sends the loss from 3.4 to 7.2 at step 5 in every run). **It has not been re-run yet: the audit's GPU jobs took the laptop's battery from 100% to 28%, and the numbers to publish should be measured on mains power.** `docs/benchmark.json` still holds the first run.

### Decided now, for the training step

- **Attention:** hand-written. **Number format:** 32-bit.
- **Weight decay:** every matrix and table decays; norm scales do not (`GPT.parameter_groups`). Decaying a scale pulls it towards 0, not towards its neutral value of 1: at a rate of 1e-3 and decay 0.1 an idle scale would fall from 1.0 to 0.61 in 5,000 steps.
- **Count passes, not steps or epochs.** With random windows there are no epochs. One pass is 294 steps for characters and 92 to 118 for the BPE sizes, so 5,000 steps is 17 passes over the characters and 42 to 54 over the BPE streams. Budgets and the learning-rate schedule will be stated in passes.
- **Schedule:** a short warm-up, then cosine decay; gradients clipped at 1.0.
- **Evaluation:** `model.eval()`, no gradients, 32-bit, one deterministic pass over the *whole* validation stream with overlapping windows so every token is scored once with at least 128 tokens of context; sum the nats, never average batch averages; bits per character = total / ln 2 / 274,727.
- **Checkpoints** carry the model and optimizer state, the config, the tokenizer's name and checksum, the step, the random generators' states and the git commit.
- **Add dropout 0.0 to the sweep** as the control, so the write-up can show what dropout buys.

Housekeeping: an auditor's script was shadowed by a stale scratch file named `numbers.py` from the pre-lock days, which read every work from disk and printed one line naming a test work. Nothing was used. The stale scripts are quarantined, and `test_only_the_loader_opens_the_works` now fails if any code in the repo opens the works except through the loader.

Lesson: two implementations that share code cannot check each other. Both attention paths split the heads with the same line, so comparing them proved nothing about that line. What caught it was a reference that shares nothing.

### Addendum to Entry 13: what reviewing blog part 4 caught (2026-09-20)

Three reviews of [blog/04-the-model.md](../blog/04-the-model.md), 91 issues. Most were missing definitions (softmax, targets, seed, learning rate, "gradients off", kernel). Six were errors, and the disputed technical ones were re-measured on the CPU, since the laptop was at 23% battery.

| The draft, and Entry 13, said | Measured |
|---|---|
| The head-reshape fault shrank "the model's context" to 43 tokens | 43 is the reach of ONE attention layer from the last position (213 to 255). Through six layers information relays further: the last position reaches back to position 128, half the window, and no further. Position 128 (the 129th) sees only itself at any depth; position 100 sees 16 positions through one layer and everything before it through three. (*measured*, autograd on the full-size config with the fault planted.) The fault also needs BOTH rearrangements wrong: either alone leaks at every cut point and is caught at once. |
| A benchmark table with "GPU, 32-bit: 158 ms" and "GPU, 16-bit: 96 ms", and "1.6 times faster" | 158 ms is the built-in attention and 96 ms the hand-written one. Like for like, hand-written: 164.8 against 95.5 ms, 1.7 times. The table now has all four GPU rows, labelled. |
| "So we predicted 4.662" | An auditor derived the 0.077 correction after the gate had run, to explain an offset it had observed. The first gate checked real text against a window. The post now says so. |
| "32 faults ... all 32 are caught by the unit tests, and the three ... by the gate" | Self-contradictory, and miscounted. 30 faults in the first harness (one of them invalid: it simplified back to the correct code), 2 more valid ones in the second, and 3 that exist only in 16-bit on the GPU: 34 valid faults, 31 caught by the unit tests and 3 by the gate. |
| "Part 1 counted ... 10,757,760" | Part 1 counted 10,758,528 (100 characters). 10,757,760 arrives in part 3. |
| "the shortest file we have written so far" | Nine files are shorter. |

Also measured for the post (CPU, random tokens): a correct untrained model scores 4.678; with targets not moved along by one, 3.52 (the residual line carries each token's own table row to the output, where it matches itself: about three times the probability of any other piece); with every matrix ten times too large at the start, 10.5. An idle norm scale under decay 0.1 falls to 0.607 at a constant rate of 1e-3 over 5,000 steps, and to 0.760 with a warm-up and a cosine decay to a tenth.

One commitment made in the post: every check in the gate ran on a model that knows nothing. **The no-peeking check will be run again on the first trained model.**

The commit message for the model (ef75834) says "All 32 faults I then planted are caught"; the count there is wrong in the same way, and a commit message cannot be edited after the fact without rewriting public history, so the correction lives here.

## Entry 14. Training: the loop, three pilots, baselines, and the rules for every comparison to come (2026-09-20)

*Status while writing: the code is written, audited, rewritten and audited again; three pilot runs are done. Results of the baseline runs are added at the end of this entry when they exist. The section "Decided now" was written, and committed, BEFORE any of those runs.*

### The benchmark, again, on mains power

`scripts/benchmark.py`, rewritten in Entry 13, run on mains (*measured*, median of three interleaved rounds, range in brackets):

| Characters, one full training step of 64 × 256 tokens | ms per step | Tokens per second | Memory in use |
|---|---|---|---|
| GPU, 32-bit, hand-written attention | 196 (177 to 212) | 83,000 | 4.79 GB |
| GPU, 32-bit, built-in attention | 194 (180 to 205) | 85,000 | 4.79 GB |
| GPU, 16-bit, hand-written attention | 103 (98 to 105) | 160,000 | 3.29 GB |
| GPU, 16-bit, built-in attention | 113 (113 to 120) | 144,000 | 3.82 GB |
| CPU, 32-bit | 1,520 | 10,800 | |

BPE with 2,048 pieces: 195, 198, 105 and 121 ms in the same order. **The mains figures are slower than the battery figures of Entry 13 (158 to 165 ms).** The likeliest reason is heat, not power: this run came straight after an hour of GPU audits, with the battery charging from 22%. The pilots below agree with the slower figure: over 40 minutes the machine sustains about 190 ms per step. A short benchmark on a cool machine flatters it. 16-bit is 1.9 times faster than 32-bit like for like.

Same start, same batches, 300 steps, with the 100-step warm-up: 16-bit ends +0.0091 nats behind 32-bit over steps 201 to 300 (two other runs of the same comparison gave +0.0019 and +0.0033). Two 32-bit runs from one seed differ by more than that at step 300 (Entry 13), so at 300 steps the two formats cannot be told apart. Entry 13's "trails by 0.02 to 0.03" was measured without a warm-up, through a loss spike.

### The training library

`src/gp_thee/train.py`, about 400 lines, half of them comments. What it holds, and the decision behind each part:

- **`RunConfig`**: every setting of a run in one frozen record, written to `config.json`, stored in every checkpoint. The recipe is nanoGPT's Shakespeare recipe unchanged: AdamW, peak rate 1e-3 falling to 1e-4 on a cosine, 100 steps of warm-up, betas (0.9, 0.99), decay 0.1 on matrices and tables only, gradients clipped at 1.0, dropout 0.2, 64 windows of 256 tokens per step. We did not tune it. (My first comment on `beta_2` said nanoGPT lowers it "for small data". An auditor read nanoGPT's config: its default is 0.95 and the Shakespeare config RAISES it to 0.99 because a step holds few tokens. The comment now only says what the number does.)
- **Length in passes.** One pass = as many tokens as the training stream holds (294 steps for characters). Not "every token once": windows are cut at random, so in one pass about 37% of the text is never read (e^-1) and about 26% is read more than once.
- **`evaluate`**: one number per token, the surprise in nats, from fixed overlapping windows (stride 128), each token scored exactly once, in 32-bit with dropout off whatever the caller is doing. Sums, never averages of batch averages.
- **Bits per character** = total nats / ln 2 / characters. START's surprise is counted; START adds no characters. All five tokenizers give 274,727 validation characters (*measured*, and re-measured by an auditor from the text itself).
- **The "seen" sample**: as many tokens of training text as validation holds, from 16 places, each read with 256 tokens of run-up that are not scored. See "What the audit found" for what this number does and does not mean.
- **Checkpoints**: weights, optimizer state, step, best-so-far, minutes used, the three random generators' states, the git commit (taken once, when the run starts), and SHA-256 checksums of the tokenizer file and both token streams. Saved under a temporary name and renamed, so a power cut cannot destroy the last good one. When a stop is the best so far, `best.pt` and `last.pt` are both written under temporary names and then renamed one straight after the other. The log row and the sample follow, and the checkpoint carries that row and the sample's heading, so a run killed at any point inside a stop resumes into the same log and the same samples as a run that was never stopped (tested at four kill points, on the CPU). At the end of a run `best.pt` is opened and must hold the step the result names. Loaded with `weights_only=True`: the file can hold numbers and text, nothing that runs.
- **Resume** takes its settings from the run's own folder. A flag that contradicts them is an error.
- **A small sampler**, always the same prompt (`HAMLET.` on a line of its own) and the same dice, so that samples from different moments of a run differ only because the model does.

`src/gp_thee/evaluation.py` splits a score by work and into speaker-label lines against everything else (the parts are checked to add up to the whole), and holds the baselines.

### The first audit of the training code, and what it found

Four auditors, CPU only, while the first pilot trained. The arithmetic of the published number was right: an evaluator written independently from the description agreed with `evaluate` to 0.0 nats per token on the full-size model over the whole validation stream, and reproduced on the CPU the step-0 row the pilot had computed on the GPU (6.7064). Everything around an interrupted or repeated run was wrong:

| Fault | Effect |
|---|---|
| Resume trusted the command line, not the checkpoint | The documented resume command, used on a run started with any non-default flag, trained the saved model on the wrong token stream and overwrote `last.pt` before crashing. Other mismatches were accepted silently and written into the new checkpoint as if true. |
| `if step == last_step: break` | A resume with a smaller budget than the step already reached never stopped. |
| Resuming a finished run | Crashed (`facts` unassigned). |
| The clock | Restarted at every resume; `minutes` under-reported. |
| Checkpoints written in place | A kill during a save would have left no usable checkpoint. |
| A new run in an old run's folder | Replaced `best.pt` with the untrained model within seconds and left the old `result.json` to be believed. |
| `map_location=device` on load | Read from PyTorch's source, not measured: after a resume on the GPU, AdamW's 39 step counters would live on the GPU and cost 39 blocking reads per step. Now loaded to the CPU first. |
| `evaluate` inside a 16-bit block | Ran in 16-bit despite its docstring. `train()` never called it that way, so no number was affected. |
| No tokenizer checksum in checkpoints | Entry 13 had promised one. |
| Evaluations every 250 STEPS | 21 chances to catch the best moment for characters, 8 for the 4,096-piece tokenizer, in a comparison decided at each run's best moment. Now a fixed number of evaluations per run. |

**30 of 64 planted faults survived the 45 tests.** Among them: training on the validation stream; `best.pt` overwritten at every evaluation; the learning rate set one step late; weight decay on the norm scales; no clipping; dropout never switched on; the seed ignored. The reason was the same each time: the tiny test run improved at every evaluation, so best equalled last; it used seed 0 and default settings only; and nothing looked at what the optimizer was doing. The tests were rewritten (81 now). One of the new ones writes the whole loop out by hand with every setting changed from its default and demands bit-identical weights; another scripts a run that over-fits, stops it, resumes it, and checks that the best moment survives.

**Two comments of mine claimed more than the code measures:**

- I wrote that the gap between the seen sample and validation "shows how much of what the model knows is just the training text by heart". A character 5-gram cannot hold more than five characters, and it shows a gap of 0.35 bits per character between works it has counted (1.912) and the validation works (2.258) (*measured*). The gap is mostly names, vocabulary and phrasing that belong to each work. For the BPE tokenizers part of it is the tokenizer itself, which was fitted on the training works. So the number is useful within a run (seen falling while validation rises is over-fitting) and must never be compared between tokenizers. Verbatim memorisation gets its own measurement in step 10.
- "Without a warm-up the loss spikes at step 5" is one observation on one seed, not a law. An auditor's CPU runs show jumps at steps 2 and 5 for one seed and only a bump at step 2 for another.

**Entry 13 said "every token is scored once with at least 128 tokens of context".** True except for the first 127 targets of a stream, which have what there is.

### Baselines: how well can Shakespeare be predicted without a neural network?

`scripts/baselines.py`, fitted on the 39 training works, scored on the 2 validation works, bits per character (*measured*; an auditor's independent implementation agrees to every printed digit):

| Predictor | All | Speaker-label lines | Everything else |
|---|---|---|---|
| Blind guess among 97 characters | 6.600 | | |
| Character counts | 4.783 | 7.227 | 4.629 |
| 3-gram | 2.902 | 3.618 | 2.856 |
| 5-gram | 2.258 | 3.540 | 2.177 |
| 6-gram | 2.231 | 3.772 | 2.134 |
| 8-gram | 2.408 | 4.227 | 2.293 |
| bzip2 -9 | 2.483, or 2.299 after reading the training works | | |
| xz -9e | 2.802, or 2.390 after reading the training works | | |

The n-grams use interpolated Witten-Bell smoothing, chosen because it has no constant to tune. It is a weak smoother (modified Kneser-Ney would score lower), so "beats the n-gram" is a low bar and the margin is what matters. We report every order rather than choosing one on validation. Speaker-label lines are 5.95% of the validation characters. (I had written that priming "cannot help bzip2" because it works in 900 kB blocks. It helps: the validation text shares its block with the tail of the training text. Measured before it reached a reader, for once.)

### Three pilots, and what they taught about the length of a run

All with seed 0, characters, 32-bit. Timings are not clean: audits were using the CPU.

| Run | Steps | Best validation | At | Seen, at that moment |
|---|---|---|---|---|
| `pilot-char-17` | 4,992 | 1.7706 | the last step | 1.5197 |
| `pilot-char-34` | 9,985 | 1.7404 | step 9,750 (33.2 passes) | 1.3654 |
| `pilot-char-68` | 19,969 | 1.7577 | step 10,750 (36.6 passes) | 1.3715 |

**The 68-pass pilot over-fitted, plainly.** After its best moment at 36.6 passes, validation rose to 1.7937 by the end while the seen sample kept falling, to 1.1787. It never reached the 34-pass run's 1.7404, most likely because its learning rate was still high when the over-fitting set in. By the rule below the character model trains for **34 passes**: doubling from 17 gained 0.030, doubling again lost 0.017.

The 17-pass pilot, taken apart by `scripts/evaluate.py`: All's Well 1.715, Romeo and Juliet 1.823; speaker-label lines 2.179, everything else 1.745. Every baseline is at least 0.46 behind.

**The length of the run is a setting, and choosing the best checkpoint does not make it harmless.** The two pilots share a seed and therefore their batches, and are identical to 0.002 up to step 1,000. From step 2,000 the 34-pass run is BEHIND the 17-pass run at every equal step, by 0.005 to 0.022, because its learning rate is still high (5.9e-4 against 1.05e-4 at step 4,750). A checkpoint from the middle of a long cosine is not the end of a short one. Yet by step 5,250 the long run had passed the short run's best, with 4,700 steps of cooling still to come. So "best at the last step, still falling" is what an under-trained run looks like, and the best-checkpoint rule only protects against a run that is too LONG. An auditor's CPU miniature put numbers on the asymmetry: too long cost about 0.01, too short 0.2 to 0.6.

**The no-peeking check, on trained weights, as promised in Entry 13:** `scripts/check_model.py --trained runs/pilot-char-17/best.pt`. All 255 cut points × both attentions × 32- and 16-bit × gradients off and on: largest change in the past 0.0. The same loss from both attentions on GPU and CPU to 2.4e-7 in 32-bit, 3.6e-5 with 16-bit included.

### Decided now, before any run that will be compared with another

1. **Seeds.** Seed 0 is used up by the pilots. Every comparison uses seeds 1, 2 and 3.
2. **Each tokenizer gets its own run length**, found with seed 0 by one rule: start near 5,000 steps; while doubling improves the best validation score by more than 0.01 bits per character, double again, up to 20,000 steps; if a run's best moment falls before two thirds of its length, also try half; keep the SHORTEST length within 0.01 of the lowest score. Equal passes would favour whichever tokenizer converges in fewer steps per pass, and equal steps would mean 17 passes for one arm and 54 for another. (The 0.01 was fixed after I had seen 17 against 34 passes for characters, a difference of 0.030, and before the 68-pass result.) To save pilots: the search is run for characters, bpe-1024 and bpe-4096; bpe-1536 takes bpe-1024's length in steps and bpe-2048 takes bpe-4096's, unless those two lengths differ by more than a factor of two, in which case all four are searched.
3. **40 evaluations per run**, evenly spaced, plus one before the first step.
4. **Frozen for every arm:** every other setting (the peak learning rate is NOT tuned per tokenizer: a limitation, stated here), the evaluation stride, the measure, the speaker-label rule (it marks 30,538 lines on the training and validation works; a second auditor's looser search found about 30 mistakes, 0.1%, all in training works and none in validation, after a first had reported 2; it cannot be debugged on the test works, so it is frozen as it is, with the known mistakes listed in its docstring and pinned by a test). The context is 256 TOKENS for every arm, which is 256 characters for one and 640 to 820 for the others: that is part of what is being compared.
5. **The only ground for discarding a run** is a loss that is not a number. It is re-run with seed + 1000 and reported.
6. **"The spread"** in Entry 12's rule now has a definition: the standard deviation over seeds, pooled over all arms. Two tokenizers differ only if their means are further apart than t × pooled sd × √(2/3), which is 1.82 × the pooled sd for five arms of three seeds. Otherwise it is a tie and the smaller vocabulary wins. (An auditor's simulation: with "larger than the two arms' own standard deviation" five truly equal arms would be declared different 31% of the time.) The procedure, spelled out: find the arm with the lowest mean; the winner is the SMALLEST vocabulary whose mean is within that threshold of it. (An auditor's simulation of five truly equal arms: any given pair is called different 5.0% of the time, as designed; at least one of the ten pairs 24.6% of the time, which is why the table of all pairs is information and only this procedure decides; under it the character arm is wrongly passed over 7.3% of the time.) Reported beside it, not deciding: for the two finalists, the per-character difference on the same text with a block-bootstrap interval, and its sign in each play.
7. **Expectations, stated in advance.** Works differ: a 5-gram scored on each training work with that work left out has a standard deviation of 0.14 across works (*auditor-measured*). So the test score may sit 0.1 to 0.2 from the validation score for no reason but which works they are. And the winner's validation score is optimistic, because it was chosen on it.
8. **Number format.** The baseline is 32-bit. One extra run, seed 1 in 16-bit, is compared with the two 32-bit runs of seed 1. If it lies no further from their mean than the larger of (their difference, the standard deviation over seeds), the tokenizer sweep may run in 16-bit, all arms alike. The published baseline stays 32-bit either way.

### The second audit (three auditors, on the rewritten code)

**Every bug of the first audit is fixed**, confirmed by running each case on the CPU, including a real `SIGKILL` of `scripts/train.py`. A run killed between stops and resumed is bit for bit the uninterrupted run. What the second round found was smaller:

- `--name ..` passed the name check and would have written checkpoints into the repository root.
- A run killed after its checkpoint but before its log row came back with a row of `nan`s; killed during the sample, it lost that sample. My sentence "resumes without a gap" was true of the step column only. Fixed as described above.
- The three pilots' checkpoints were written by the first version of the code and the new, stricter loader refused them. The loader now allows the one harmless object they contain. They can be scored, not resumed.
- The checksums in a checkpoint were of the token FILES, read a second time, not of the tokens the run had actually loaded. Every test run showed the difference (the tests train on the first 40,000 tokens). Now the arrays in memory are hashed; it costs 7 ms.
- **21 of 28 newly planted faults survived the 81 new tests**: checkpoints written in place, the two saves in the wrong order, half of the overwrite guard, `weights_only=False`, a git stamp that was always "unknown" (the tests run outside any repository, and "unknown" has the seven characters the test asked for), `tokens_per_second` wrong three ways, and the whole command-line script, which had no test. The same hole as before, one level further out: new safety code, checked only on the path where nothing goes wrong. 102 training tests now.
- **The evaluation code computes the right numbers.** An independent n-gram agrees on every one of the 274,727 validation characters for orders 1 to 8, to 0.0 bits; the compressor figures match the command-line tools digit for digit. 5 of 39 planted faults survived its 27 tests (both of `breakdown`'s self-checks, bits per BYTE instead of per character, wrong compressor settings); 33 tests now. `breakdown` said it checked its assumptions and did not check where the START tokens were; now it does.
- **bzip2's "after reading the training works" figure (2.299) is partly luck.** It depends on where a 900 kB block edge falls: shortening the training text by 0 to 800 kB moves it between 2.27 and 2.47 (*measured, by an auditor and again by me*). xz's moves from 2.390 to 2.396.
- A docstring of mine said ROMEO has to be spelled out "thirty times a scene". Measured: 162 times in the play, 6.8 per scene, never more than 26. The same sentence was in the blog draft.

Still owed when this was written: a test for `scripts/summarise_runs.py`, which applies the rule of item 6 and has none; tighter limits in `check_model.py --trained`; the third auditor's report (a second mutation round).

### Addendum to Entry 14: what reviewing blog part 5 caught (2026-09-21)

Three reviews of the draft of [blog/05-training.md](../blog/05-training.md) (fact-check, ML expert, newcomer), 93 issues. Every table matched its log or JSON file to the digit. The errors were all in sentences that said more than the data does, and every new number the reviewers produced was measured again by me before it went in.

| The draft said | Measured |
|---|---|
| The sample shown "after 1,000 steps" of the 17-pass run | It was the 34-pass run's sample at that step. Same seed, but a different run length, so a different learning rate at every step after the warm-up. |
| The opening sample shows what the model writes "by the end" | Step 3,000 of 9,985 of the 34-pass pilot, picked from 143 samples because Polonius walks in. The post now says so. |
| "At every matching step the long run is BEHIND" | Level to 0.002 up to step 1,000; the two trade places between 1,000 and 2,000 (the long run is 0.012 *ahead* at step 1,250); behind at all twelve stops from 2,000 to 4,750, by 0.005 to 0.022. This entry had it right ("from step 2,000"); the post did not. |
| After its best moment the 68-pass run's validation "climbed steadily" | It hovered between 1.761 and 1.777 for 4,000 steps and then drifted up: it rose at 21 of the 37 later stops and fell at 15. Its seen score did fall at every one of the 80 stops. |
| "With a look-back of seven, almost everything in a new play is unseen" | 19.2% of validation characters follow a run of seven that never occurs in the training works (4.9% for five, 2.1% for four). The 8-gram loses because thin counts are over-trusted by Witten-Bell, a weak smoother. |
| 6.60 bits "is the ceiling" | The untrained model scores 6.71, two sections later in the same post. |
| "Bit for bit" resume | On the CPU. The post now says what holds on the GPU. |
| The two attentions agree "to seven decimal places" | The difference is 2.4e-7: six places. |
| Clipping stops "one enormous step", and a fresh model's gradients "point in wild directions" | Both are the plain-gradient-descent story. Under AdamW the size of a step does not follow the size of the gradient; the benchmark's jump to 7.2 happened with the clip on; and in the first baseline run the clip fired on 9.6% of the first 250 steps and on none after. At the first step AdamW's update is exactly plus or minus the learning rate for every parameter, which is why a full rate from step one overshoots. |

**New measurements made for the post** (*measured*, CPU, validation and training works only; script in the session scratchpad, `blog5/measure.py`):

- Of the 1,773 speaker labels in the validation plays, 915 carry a name that also heads a speech somewhere in the training works and 858 do not. The first kind costs 0.92 bits per character, the second 3.45 (17-pass pilot).
- From 17 to 34 passes, label lines went from 2.179 to 2.310 while everything else improved from 1.745 to 1.704. All of the damage is in the letters after the first letter of an unknown name: 4.04 to 4.62 bits each (known names: 0.66 to 0.76). First letters (3.90 to 3.68) and full stops (2.57 to 1.16) improved. One pair of runs, one seed.
- In *Hamlet*, a training work, the 34-pass pilot scores 0.34 on label lines and 1.44 on everything else.
- 45% of validation characters cost under half a bit and 15% over 4 bits; the first letter of a word costs 3.76 bits, later letters 1.43.
- Scored in windows that do not overlap, as nanoGPT does, the 34-pass pilot gets 1.243 nats per character against 1.206 with 128 tokens of run-up: 0.037 of any comparison with the tutorial's 1.47 is the ruler.
- Block bootstrap over 55 blocks of 5,000 characters: one score is uncertain by 0.013 bits from the choice of text alone (more with longer blocks, per the first audit); the difference between the 17- and 34-pass pilots is 0.030 ± 0.003.
- bzip2's primed figure: 2.299, 2.380, 2.473 and 2.266 with the first 0, 200, 400 and 700 kB of training text dropped; xz 2.390 and 2.396.

The post is the longest of the series by half (about 8,000 words). The newcomer's review said so. It stays whole for now; splitting it is the owner's call.

### The baseline (2026-09-21, 00:38 to 03:20): characters, 34 passes, 32-bit, seeds 1 to 3

All five runs from commit `5c6bd68` with a clean tree, 9,985 steps, 40 stops, run one after another on mains power. *Measured*; gathered by `scripts/summarise_runs.py char-34-` into [docs/results-char-34.json](results-char-34.json).

| Run | Best validation bpc | At step | Final | Seen at best | Minutes |
|---|---|---|---|---|---|
| `char-34-seed-1` | 1.7490 | 7,239 | 1.7551 | 1.4276 | 37.7 (audits were using the CPU) |
| `char-34-seed-1-again` | 1.7517 | 8,986 | 1.7614 | 1.3794 | 35.8 |
| `char-34-seed-2` | 1.7597 | 8,986 | 1.7663 | 1.3744 | 35.8 |
| `char-34-seed-3` | 1.7504 | 9,735 | 1.7584 | 1.3633 | 36.0 |
| `char-34-seed-1-16bit` | 1.7592 | 8,238 | 1.7685 | 1.4032 | 21.7 |

- **Baseline: 1.7535 bits per character, standard deviation over seeds 0.0054** (n = 3; the twins of seed 1 averaged first).
- **Same seed twice: 0.0027 apart**, and the two had their best moments at different stops.
- **Where the spread is.** By `scripts/evaluate.py`, per seed: everything else 1.7147 / 1.7135 / 1.7121 (sd 0.0013); speaker-label lines 2.315 / 2.490 / 2.356 (sd 0.09). The twins differ by 0.005 on ordinary text and by 0.125 on labels. Nearly all the noise in the deciding number comes from 6% of the characters. The pre-registered rule decides on the whole text and that does not change; the two-way split is reported beside it, as Entry 12 said it would be. *All's Well* 1.7035 (sd 0.0014), *Romeo and Juliet* 1.8007 (sd 0.0116).
- **16-bit: not adopted for the sweep.** Item 8's limit was the larger of the twins' difference (0.0027) and the seed spread (0.0054). The 16-bit run is 0.0088 from the twins' mean. It is inside the range of the three 32-bit seeds (seed 2: 1.7597), so this is not evidence that 16-bit learns worse, only a failure to show that it learns the same. The rule was fixed in advance, so the sweep runs in 32-bit.
- **The rule's decisions hold up, their sizes do not:** see "A scare" below.
- Every run ended 0.006 to 0.010 above its best, and the best moments fell at 72% to 97% of the run: over-fitting has just begun at 34 passes.
- **Speed**, with the machine otherwise idle: 85,400 tokens per second in 32-bit (192 ms per step), 153,800 in 16-bit. The 41 stops cost about four minutes per run.
- **Clipping:** on about 10% of the first 250 steps in every run (gradient length 0.81 to 0.82 on average there), and on no step after that (0.26 on average late in the run).
- **`check_model.py --trained runs/char-34-seed-3/best.pt`**, now comparing the loss at every token: no peeking 0.0 in all eight combinations; worst single token 1.2e-5 apart in 32-bit across attentions and devices; with 16-bit, 0.099 at the worst token and 5.3e-4 in the mean. The limits in the script are set from these.

### A scare: does the rewrite train worse? No. An edit re-rolls the dice (2026-09-21, 03:26 to 06:30)

The 34-pass pilot (seed 0, first version of the code) scored 1.7404, below all four new runs. Lucky seed, or had the rewrite changed the training? Four more runs, all 34 passes, *measured*. The first-version runs were made from an auditor's scratch copy of the old file (it was never committed) and live in the session scratchpad, not in `runs/`.

| Code | Seed | Best | Final | Second-half validation | Second-half training loss |
|---|---|---|---|---|---|
| first version (`pilot-char-34`) | 0 | 1.7404 | 1.7423 | 1.7554 | 1.0915 |
| first version | 0, again | 1.7423 | 1.7457 | 1.7558 | 1.0913 |
| rewritten (`char-34-seed-0`) | 0 | 1.7525 | 1.7553 | 1.7632 | 1.0916 |
| rewritten (`char-34-seed-0-again`) | 0, again | 1.7514 | 1.7538 | 1.7612 | 1.0914 |
| first version | 1 | 1.7571 | 1.7631 | 1.7724 | 1.0925 |
| rewritten (`char-34-seed-1`) | 1 | 1.7490 | 1.7551 | 1.7605 | 1.0927 |
| rewritten (`char-34-seed-1-again`) | 1, again | 1.7517 | 1.7614 | 1.7657 | 1.0924 |

How I got there, including the wrong turn. After the first three rows I called it luck ("two coin tosses") and wrote that the seed adds nothing beyond the GPU's last-digit noise. A fact-checking agent pointed out that both old-code runs lay below all five new-code runs, a 1-in-21 event if they were draws from one bag, and that at the runs' best moments the old code's lead was mostly on ordinary text, not labels. So I ran the two direct tests: the rewritten code with seed 0 again (1.7514: it repeats itself), and the first version with seed 1 (1.7571: WORSE than the rewritten code's 1.7490 and 1.7517). Neither version is better.

- **On the CPU the two versions give bit-identical weights, optimizer averages and validation score** from the same seed (78 steps of a small model with dropout on). They do the same arithmetic.
- **On the GPU the training loss follows the seed and ignores the version**: 1.0913 to 1.0916 for all four seed-0 runs, 1.0924 to 1.0927 for all three seed-1 runs.
- **The same program with the same seed repeats to within 0.003** in its best score: three pairs, 0.0019, 0.0011 and 0.0027 apart (pooled sd 0.0014). At single stops a pair can be 0.019 apart, and the seed-1 twins stay 0.005 apart on average over the second half.
- **A change of program that changes no arithmetic moves the score as far as a change of seed**: +0.010 for seed 0, -0.007 for seed 1. My earlier sentence, that the seed adds nothing we can see, was wrong: within one program the seed is most of the spread (seeds 0 to 3 of the rewritten code: sd 0.0045). What I had mistaken for same-seed noise was the difference between two programs.
- Mechanism, *a guess, not tested*: the one non-deterministic operation (the token-table gradient, Entry 13) adds up in an order that depends on how the program's work is scheduled on the GPU, which is steady for one program and different for another. The rewrite keeps one more tensor per step and evaluates at other steps; that is all it would take.

**Consequences.** All nine 32-bit 34-pass runs: mean 1.7505, sd 0.0062, 1.7404 to 1.7597. The 17-pass pilot (1.7706) is behind all nine, by 0.011 to 0.030, so "longer than 17" stands against the rule's 0.01, with less room than the pilots suggested. The 68-pass pilot (1.7577) beats one of the nine by 0.002, so "no gain from 68" stands and more than half of "68 lost 0.017" was one run's luck. The rule picks 34 either way. The pre-registered baseline is unchanged: seeds 1 to 3 with the committed code, 1.7535, sd 0.0054. With seed 0 as a fourth seed (two runs, averaged), which is what `docs/results-char-34.json` now holds: 1.7531, sd 0.0045.

**For every comparison from here on:** all arms are run by the SAME commit of `src/`. An edit re-rolls every score by about 0.006. That is noise and not bias (it went one way for seed 0 and the other for seed 1), so mixing commits would not favour an arm, but it would add spread for nothing. The character arm will therefore be run again with the sweep's commit. Pilots made by an older version are not comparable with runs of a newer one at the third decimal.

**16-bit, the quieter evidence.** The training loss follows the seed and ignores the program: the three 32-bit runs of seed 1 give 1.0924 to 1.0927. The 16-bit run of seed 1: 1.0952, 0.0027 above them. One run; a hint. The validation score could not show it.

**For part 6.** Three seeds per tokenizer give a mean with a standard error of about 0.003. The pre-registered test will therefore call anything under about 0.01 a tie, which is as it should be. The per-line-type split is where a real difference would show first, since ordinary text is four times quieter than the whole (sd 0.0013 against 0.0054 at the best moments).

## Entry 15. The tokenizer comparison: the plan, as a program, before the runs (2026-09-21)

Entry 12 fixed the deciding number and Entry 14 the seeds, the lengths, the stops and the test. This entry turns those words into two programs and settles what the words left open. It is committed before the first run it governs.

**The length rule, as code: `scripts/find_length.py`** (pure functions `next_to_try` and `choose`, pinned by `tests/test_find_length.py`). Three readings of Entry 14's sentence had to be chosen:

- *"While doubling improves the best score by more than 0.01, double again"*: the run of twice the starting length is ALWAYS made, even if the 5,000-step run over-fitted early. We cannot know that doubling does not help without trying it. "More than 0.01" is strict: 0.0099 stops the doubling.
- *"If a run's best moment falls before two thirds of its length, also try half"* applies to every run tried, the halves included, down to a floor of 625 steps. (Below that, a 100-step warm-up is a sixth of the run and the question stops being about this recipe.)
- *"Keep the shortest length within 0.01 of the lowest score"*: within means at most 0.01 above.

The search runs with seed 0 for bpe-1024 and bpe-4096. bpe-1536 borrows bpe-1024's length in steps and bpe-2048 borrows bpe-4096's, unless the two lengths found differ by more than a factor of two, in which case all four are searched (`scripts/sweep.py` refuses to start otherwise). The character length stays 9,985 steps (34 passes), as found in Entry 14.

**Known weakness, stated now:** Entry 14 showed that a single run is a draw with a standard deviation of about 0.006. The rule compares single pilot runs against a threshold of 0.01, so its choices are partly luck. It is the same rule and the same luck for every tokenizer, "the shortest within 0.01" forgives a near miss, and changing the rule after seeing what it costs would be worse than living with it.

**The comparison, as code: `scripts/sweep.py`.** Five arms (char, bpe-1024, bpe-1536, bpe-2048, bpe-4096) × seeds 1, 2, 3 = 15 runs named `sweep-<tokenizer>-seed-<n>`, 32-bit, 40 stops, every other setting at its default. Seed by seed, not arm by arm, so that a machine that warms up or a sweep cut short treats every arm alike. **All 15 runs come from one clean commit**: the script checks before each run and stops if the code has moved (Entry 14: an edit re-rolls a score as surely as a new seed). So the character arm is run again; the three baseline runs of Entry 14 (commit `5c6bd68`) stay in the record as the baseline they were, and take no part in the comparison.

**What decides, and what is only reported.** Deciding: mean over three seeds of each run's best validation bits per character over all the text; the winner is the smallest vocabulary whose mean is within t × pooled sd × √(2/3) of the lowest mean (`scripts/summarise_runs.py sweep-`). Reported beside it, not deciding: the table of all ten pairs; each arm's steps, passes and parameters; the score by play and for speaker-label lines against everything else (Entry 14 found most of the run-to-run noise in the labels, so this split is where a real difference should show first); for the two finalists, the per-character difference on the same text with a block-bootstrap interval, and its sign in each play.

**A run is discarded only if its loss stops being a number**, and then it is re-run with seed + 1000 and reported. Nothing else is grounds: not a bad score, not an odd curve.

**Expected cost** (*estimate*): length search about one hour per tokenizer (5,000 + 10,000 steps, plus halves), so two to four hours; the sweep about five hours (three character runs at 36 minutes, twelve word-fragment runs at 10 to 35). On mains power.

### Addendum to Entry 15: what a pre-flight review found, and more words fixed before the sweep (2026-09-21)

While the first pilot trained, three auditors attacked the plan on the CPU. No blocker. The word-fragment arms train and are scored correctly: a full-size pilot's recorded score was reproduced on the CPU to 1.4e-7; one set of per-character probabilities gives the identical bits per character through all five segmentations (to 3e-14); the step-to-passes conversion is exact for every arm and every length from 1 to 20,000 steps; the saved token streams equal a fresh encoding. What they did find was in the two runner scripts, which had only ever been dry-run:

- **A run whose loss stops being a number would have been RESUMED** when the command was given again, and the sweep could never have got past it. The rule "discard it and re-run with seed + 1000" existed only in words. Now: the run is kept, marked with a `DIVERGED` file, never resumed, and replaced by `sweep-<arm>-seed-<seed + 1000>` from the same commit at the same length (seed + 2000 if that one diverges too). A diverged pilot is replaced by seed 1000. `summarise_runs.py` lists every discarded run.
- `sweep.py` never checked that a run made earlier had the length the plan asks for now, so one arm could have ended up with mixed lengths. It checks length, tokenizer and seed now.
- Its one-commit check only looked at runs earlier in today's order: an auditor got "all done" with 14 runs from one commit and 1 from another. It now gathers every `sweep-*` run first, refuses when git says "unknown", and its message says how to carry on. The check stays on the whole commit, so **nothing is committed while the sweep runs**.
- Three doors the programs left open that the words do not: a hand-made pilot of any length entered the choice (an off-grid 7,500-step pilot became "the rule's pick" in the auditor's test); a searched length for a borrower silently overrode the borrowing; and `summarise_runs.py` named a winner for a half-finished sweep (simulated: the choice after two seeds differs from the choice after three 9% of the time with equal arms, 46% with one arm truly better by 0.015). Now: only lengths the rule itself asks for are read, and only pilots made with seed 0 and default settings; borrowers borrow unless the factor-of-two escape fires; a verdict is printed once, when every arm has the same number of seeds.

**More words, fixed now because results could steer them later:**

1. **A chosen length stands** even if its pilot's best moment, or all three of its sweep runs' best moments, fall at the last stop (the cap of 20,000 included). The write-up will call that arm possibly under-trained. No length is searched again once `docs/run_lengths.json` is committed.
2. **Borrowing is not even-handed, and we say so now.** If the two searched lengths differ by exactly a factor of two, bpe-1536 inherits the longer one (it can only be trained too long, which the best checkpoint partly repairs) and bpe-2048 inherits the shorter one with 11% fewer passes on top (it can only be trained too briefly, the expensive direction). If all three of bpe-2048's runs have their best moment at the last stop, its score is a lower bound on what that vocabulary can do. The decision is unchanged.
3. **What the comparison can and cannot show** (an auditor's simulation of the exact procedure, per-run sd 0.0045 to 0.0062): a vocabulary truly better than characters by 0.03 bits per character takes the title every time; by 0.02, 94 to 99.8% of the time; by 0.01, about half the time; by 0.005, one time in five. The threshold itself will land somewhere between 0.007 and 0.015. So "characters win" will mean "nothing beat characters by more than about 0.011 under this recipe and this length rule". It will not mean characters are better. The burden of proof is on the larger vocabulary, by design, and when the true gap is near 0.01 the arm crowned is as likely to be a smaller neighbour as the truly best one. If the word-fragment arms turn out noisier than characters, the pooled threshold grows and all of this gets weaker.
4. **The length rule is itself noisy.** It compares single pilots (sd about 0.006) against 0.01. Simulated on the real code: where the curve is steep it picks the right length 92 to 100% of the time; where the last doubling truly gains only 0.012 to 0.020 it stops one step short 12 to 41% of the time, which can cost an arm about 0.01, the size of the smallest difference the test can see. Only the character length has been confirmed by more than one run. A tie or a narrow loss for a word-fragment arm therefore means "no better under this recipe and this length rule".
5. **Bits per character is exact for characters and an upper bound for word fragments.** The score charges for the one spelling our tokenizer produces; a word-fragment model also gives some probability to other ways of cutting the same text, and that is lost. An auditor measured it on the first full-size pilot by summing over every way of cutting 2,997 chunks: about 0.008 bits per character (0.004 to 0.013), four fifths of it in ten unseen names such as BENVOLIO. The pre-registered measure stays, because it is the cost of the model as it will be used. The slack will be measured and reported for any word-fragment arm that finishes within 0.02 of the winner. It does not enter the decision.
6. **A named secondary, deciding nothing:** the identical test (same seeds, same t × pooled sd × √(1/n + 1/n), the pooled sd computed on that statistic) applied to each run's best checkpoint scored on everything except speaker-label lines. The checkpoint is still the one chosen on the whole text, so nothing new is selected. Entry 14 measured an sd of 0.0013 there for characters, against 0.0054 on the whole text. How the four outcomes will be worded: both agree, say so; primary tie and secondary different, "the vocabulary predicts ordinary text better by X; on the whole text that is inside the noise of unseen names; the rule still gives characters the title"; primary different and secondary tie, "the difference is in the names"; both different in opposite directions, report both and let the rule stand. Unseen names hit larger vocabularies hardest, so this split flatters them by construction, which is one more reason it cannot decide.
7. **The finalists, on the same text** (`scripts/compare_finalists.py`, written and smoke-tested on pilots before the sweep): characters against the word-fragment arm with the lowest three-seed mean; each run at its `best.pt`; a token's bits shared equally among its characters, START's bits on the first character of the work it opens; per arm the mean over three seeds; blocks of 5,000 characters within each play; 10,000 resamples of blocks, play by play, `default_rng(0)`, 95% by percentiles; the sign per play. Printed beside it: the interval holds the six models fixed and measures the choice of text, not the luck of training, and cannot overturn the rule.

**And one more, found by the tests written for the fixes.** A helper agent wrote tests for all four scripts (69 new tests; 134 of 140 planted faults killed, the other six shown to be equivalent) and found a bug in MY rewrite of `sweep.py`: it asked git where the code was once, at the start, so a commit landing mid-sweep went unnoticed (the helper's copy finished "all done" with 2 runs from one commit and 13 from another). It now asks before every run. Also tightened before any sweep run: no verdict while any run of the prefix is unfinished or any arm has fewer than three seeds (the sweep goes seed by seed, so "every arm has the same number of seeds" was true after two seeds, exactly the early look the rule was meant to prevent); a pilot or a sweep run made earlier counts only if every setting but its name, tokenizer, seed, length and device is the default.

### The length searches (2026-09-21, 07:31 to 09:45; seed 0; commit `f5713e4`; *measured*)

`scripts/find_length.py`, exactly as the rule asks: 5,000 steps, then the compulsory 10,000, then halves while a best moment falls before two thirds of its run.

| Tokenizer | Steps | Passes | Best validation bpc | At | Final | Seen at the end |
|---|---|---|---|---|---|---|
| bpe-1024 | 2,500 | 21.2 | 1.8085 | 98% | 1.8104 | |
| bpe-1024 | 5,000 | 42.3 | 1.8001 | 57% | 1.8108 | 1.20 |
| bpe-1024 | 10,000 | 84.6 | 1.8045 | 30% | 1.9203 | |
| bpe-4096 | 2,500 | 27.2 | 1.8515 | 90% | 1.8538 | |
| bpe-4096 | 5,000 | 54.4 | 1.8733 | 30% | 1.9792 | 0.93 |
| bpe-4096 | 10,000 | 108.7 | 1.8759 | 15% | 2.2218 | 0.57 |

**The rule picks 2,500 steps for both** (bpe-1024: the lowest is 5,000's 1.8001, and 2,500's 1.8085 is within 0.01 and shorter; bpe-4096: 2,500 is simply the best). The two lengths are equal, so bpe-1536 and bpe-2048 borrow 2,500 steps: 22.9 and 24.2 passes. Characters keep 9,985 steps (34 passes).

What the pilots already show, as single draws and nothing more: the word-fragment models over-fit far sooner than the character model (at 10,000 steps bpe-4096 scores 0.57 on text it has trained on and 2.22 on validation), their best moments come after 16 to 25 passes where the character model's came after 25 to 33, and a SHORTER run beat a longer one for bpe-4096, because the learning rate has cooled before the over-fitting sets in. For bpe-1024 the rule went one step short in the way the pre-flight review predicted: its chosen run had its best moment at 98% and gives away 0.008 against the 5,000-step pilot. As written down in advance, the choice stands and that arm will be called possibly under-trained. All four word-fragment pilots' best scores (1.80 to 1.88) are behind the character runs (1.75), but pilots decide nothing.

## Entry 16. The tokenizer comparison: characters win (2026-09-21)

Fifteen runs, `scripts/sweep.py`, all from commit `0f72600` with a clean tree, 09:46 to 13:29 on mains power, seed by seed. No run diverged, none was discarded, none was resumed. *Measured*; the record is [docs/results-sweep.json](results-sweep.json).

| Arm | Parameters | Steps (passes) | Seed 1 | Seed 2 | Seed 3 | **Mean** | sd |
|---|---|---|---|---|---|---|---|
| characters | 10,757,760 | 9,985 (34.0) | 1.7492 | 1.7571 | 1.7575 | **1.7546** | 0.0047 |
| bpe-1024 | 11,113,344 | 2,500 (21.2) | 1.7895 | 1.8006 | 1.8136 | **1.8012** | 0.0121 |
| bpe-1536 | 11,309,952 | 2,500 (22.9) | 1.8313 | 1.8219 | 1.8186 | **1.8240** | 0.0066 |
| bpe-2048 | 11,506,560 | 2,500 (24.2) | 1.8227 | 1.8291 | 1.8198 | **1.8239** | 0.0048 |
| bpe-4096 | 12,292,992 | 2,500 (27.2) | 1.8460 | 1.8539 | 1.8533 | **1.8511** | 0.0044 |

**The rule's verdict: characters.** Pooled standard deviation 0.0071 on 10 degrees of freedom, so two arms differ beyond 0.0130. Characters have the lowest mean and no other arm is within reach: the nearest, bpe-1024, is 0.0466 behind, 3.6 times the threshold. Of the ten pairs, nine are different and one is a tie (bpe-1536 and bpe-2048, 0.0001 apart). The order is the order of vocabulary size.

What had been written down in advance, against what happened:

- *The test can see 0.02 and cannot see 0.01.* The smallest gap to characters is 0.047. This is not a close call that the rule's weakness could explain.
- *bpe-1024 may be under-trained, by about 0.008.* Two of its three runs had their best moment at the very last step, as its pilot did. Even the 5,000-step pilot (1.8001) is 0.046 behind characters.
- *The word-fragment score is an upper bound, slack about 0.008.* Promised to be measured for any arm within 0.02 of the winner. None is.
- *The pooled spread assumes equal noise in every arm.* It is not quite equal: bpe-1024's seeds spread 0.012, the others 0.004 to 0.007. That widened the threshold from the expected 0.011 to 0.013 and changes nothing.
- *The character arm, run again from the sweep's commit:* 1.7546 (sd 0.0047), against the baseline's 1.7535 (sd 0.0054) from commit `5c6bd68` in Entry 14.

**The named secondary (deciding nothing): everything except speaker-label lines.** Characters 1.7117; bpe-1024 1.7745, bpe-1536 1.7764, bpe-2048 1.7772, bpe-4096 1.7799; pooled sd 0.0036, so differences beyond 0.0066 count. Characters differ from all four (0.063 to 0.068 ahead). **The four word-fragment vocabularies tie with each other on ordinary text.** So the whole ordering AMONG them comes from speaker labels: 2.225 bits per character for bpe-1024, 2.575, 2.561 and 2.976 for the larger three (characters: 2.434). In the words fixed in advance: both tables agree that characters win; among the word-fragment arms, "the difference is in the names".

**The finalists on the same text** (`scripts/compare_finalists.py`, [docs/finalists.json](finalists.json)): bpe-1024 minus characters = +0.0466 bits per character, 95% interval over the choice of text +0.0361 to +0.0595 (56 blocks of 5,000 characters); +0.0207 in *All's Well*, +0.0711 in *Romeo and Juliet*; characters are the better of the two in 80% of the blocks. On speaker-label lines bpe-1024 is the BETTER one, by 0.209; on everything else characters are, by 0.063. (The interval holds the six models fixed and measures the choice of text, not the luck of training.)

**Where they differ, an exploration made after the verdict** (`scripts/where_they_differ.py`, [docs/where_they_differ.json](where_they_differ.json)). Every tokenizer cuts the text into the same chunks first and no piece crosses a chunk's edge, so each model's bits can be added up exactly per chunk; the groups' totals add up to each arm's score. (My first attempt shared a token's bits equally among its characters and "found" that word fragments are twice as bad at spaces and better at every kind of word. That was the leading space of each word-fragment piece, moved onto the space by the sharing. Chunks fix it.) In bits per character of the whole text that each group adds:

| Kind of chunk | Share of text | characters | bpe-1024 | bpe-4096 |
|---|---|---|---|---|
| word seen 1,000 times or more in training | 36.4% | 0.5026 | 0.5249 | 0.5233 |
| seen 100 to 999 times | 23.3% | 0.3671 | 0.3737 | 0.3691 |
| seen 10 to 99 times | 16.3% | 0.3013 | 0.3132 | 0.3173 |
| seen 1 to 9 times | 7.3% | 0.1641 | 0.1828 | 0.1848 |
| never seen | 3.3% | 0.1320 | 0.1296 | 0.1374 |
| punctuation | 4.4% | 0.1228 | 0.1200 | 0.1168 |
| line breaks | 3.0% | 0.0194 | 0.0240 | 0.0246 |
| speaker-label lines | 5.9% | 0.1448 | 0.1324 | 0.1771 |

Characters are ahead on words of every frequency, by a little each: 5.12 bits against 5.35 for a very common word, 18.3 against 20.4 for a word seen under ten times. The rarer the word, the bigger the gap per word, until the word is unseen, where bpe-1024 is level. Word fragments are slightly better at punctuation. And the largest vocabulary loses 0.032 of its 0.097 on speaker labels alone.

**Why? Guesses, none of them tested.** (1) Data. 4.8 million characters is tiny. A character model meets each of its 97 symbols tens of thousands of times in every sort of company; a word-fragment model has to learn thousands of pieces from a third as many examples, and the pilots show it memorising the training plays instead (0.57 on seen text, 2.22 on unseen). (2) Names. At 1,536 pieces and above the training plays' names are single pieces (`HAMLET`), so after a blank line the model bets on whole names it knows; `ROMEO` must then be spelled `RO|M|E|O` against that bet. At 1,024 fewer names are whole pieces, and that arm is the best of all five on labels. (3) The recipe. Learning rate, dropout and batch were tuned, by someone else, for a character model, and we changed nothing. (4) The run lengths and the measure each lean against word fragments by up to about 0.01. Together (3) and (4) could plausibly explain 0.02. They do not explain 0.047 to 0.097.

**Decision.** GP-Thee uses the character tokenizer: 98 symbols, 10,757,760 parameters, 34 passes. Speed, for the record: 80,000 to 87,000 tokens per second for every arm, so a word-fragment model reads characters about 2.5 to 3 times faster; that did not buy a better model here.

### Addendum to Entry 16: what reviewing blog part 6 caught (2026-09-21)

Three reviews of the draft of [blog/06-which-tokenizer.md](../blog/06-which-tokenizer.md) (fact-check, ML expert, newcomer), 85 issues. Every table cell matched its source to the digit, and the fact-checker confirmed from git that Entry 15 was committed 9 seconds before the first pilot started and its addendum 7 seconds before the first sweep run. The errors were again in my prose, and this time in my explanations. Every number below was measured by a reviewer and again by me (`blog5/measure6.py` in the session scratchpad).

**Both guesses in Entry 16 under "Why?" are undercut by the project's own data.**

- *"Scarce pieces" (guess 1).* At 4,096 pieces the middling piece occurs 101 times in the training stream and 49% of pieces under 100 times; at 1,024 the middling piece occurs 769 times and 4% are that rare. If scarcity were the cause, bpe-4096 should be far worse than bpe-1024 on ordinary text. It is 0.0054 worse (a tie). Words seen 1 to 9 times cost every vocabulary the same 20.3 to 20.6 bits. And the largest slice of bpe-1024's 0.047 deficit is the COMMONEST words (0.022), single pieces met thousands of times; the rare-word row adds 0.019. The per-word gap is not monotone either: 4%, 2%, 4%, 11%, then bpe-1024 level or ahead on unseen words. So whatever costs 0.06 on ordinary text is shared by all four vocabularies and is not piece scarcity.
- *"Betting on whole names" (guess 2).* Names that are single pieces: 4 at 1,024 (FALSTAFF, KING, DUKE, QUEEN), 12 at 1,536, 33 at 2,048, 151 at 4,096, heading 4.6%, 12.3%, 23.7% and 59.2% of training speeches. Label scores: 2.225, 2.575, 2.561, 2.976. Doubling the whole names from 1,536 to 2,048 moved nothing; the jump is from 1,024 to 1,536 where whole names barely change. Split as blog 5 did, known names (915 labels) against unknown (858): characters 0.917 / 3.963; bpe-1024 0.694 / 3.767; bpe-1536 0.791 / 4.375; bpe-2048 0.848 / 4.288; bpe-4096 1.009 / 4.959. The larger vocabularies are worse on KNOWN names too, and 1,830 of the 3,418 bits of bpe-1024's lead over characters on labels are on known names. The label column is the noisiest there is (pooled sd 0.083, so only differences beyond 0.15 count): what stands is that bpe-4096 is worst and bpe-1024 best; and the arms trained for different lengths, which blog 5 showed matters for labels.

What is left as candidates, all untested: fewer updates per pass (a word-fragment step holds 2.5 to 3 times as many characters), a recipe tuned for characters, step-counted settings falling unevenly (warm-up 4% against 1% of a run; weight decay acting over four times as many steps on the character model), and the measure's slack. The blog now says "I do not know".

**Other corrections.**

| The draft, or an earlier entry, said | Measured or checked |
|---|---|
| The largest vocabulary "has all but learned the 39 plays by heart", and 0.57 (its seen score) set beside the character model's 1.18 | "By heart" is the claim blog 5 retracted, and Entry 14 forbids comparing seen scores between tokenizers. On validation alone: after about 68 passes the character pilot stood +0.036 above its own best, bpe-1024 +0.090, bpe-4096 +0.238. Also: the training works are 34 plays and 5 books of poems, not 39 plays. |
| Power table headed "it takes the title" (the same slip is in Entry 15, item 3) | The figures are the chance that characters LOSE the title. In the auditor's followed-through case the truly better arm is crowned 90% of the time at a true 0.02 and 24% at 0.01, a no-better smaller neighbour 22%. |
| bpe-1024 "carries a handicap of about 0.008" | The difference between two single runs, inside their wobble. Its three sweep runs at 2,500 steps average 1.8012, level with the 5,000-step pilot's 1.8001. The evidence of under-training is the signature (best moment at the last stop), not a size. An auditor also noticed that whether 2,500 steps was tried at all hung on the fifth decimal: the 5,000-step pilot's stops at 2,875 and 3,375 tie at 1.8001, on either side of the two-thirds line. |
| "A full-size pilot's score recomputed by an independent route" | Two checks merged. The full-size pilot was re-scored on the CPU with our own code (1.4e-7). The scorer written from scratch was run on four small models. |
| The faulty commit check "looked only at runs made today" | It looked only at runs EARLIER IN THE RUNNING ORDER than the one about to be made. |
| "The bigger the table, the worse the score" | bpe-2048 (1.82386) is a hair ahead of bpe-1536 (1.82396): the one tie among the ten pairs. |
| "On ordinary text the four vocabularies are the same" | A tie under a test that can reliably see about 0.01. The means do rise with vocabulary size, 0.0054 from first to last. |
| "A character model meets each of its 97 symbols tens of thousands of times" | 45 of the 97 occur 10,000 times or more (98.1% of the text); 25 occur under 100 times. |
| Invented spellings of Benvolio | Real: `B|EN|V|OL|IO` at 1,024 and 1,536, `B|EN|VOL|IO` at 2,048, `B|EN|VOLIO` at 4,096. |
| Pooled test with unequal spreads "changes nothing" | True, and now shown: judged on the two arms' own spreads (Welch, 2.6 degrees of freedom), characters against bpe-1024 needs about 0.026 and the gap is 0.047. What that harsher test removes is bpe-1024's lead over the two middle vocabularies (0.023 apart, about 0.025 needed). |
| "A shorter run beat a longer one" stated as mechanism | One pair of runs, but supported: the 10,000-step pilot also reached only 1.8759 and the three 2,500-step sweep runs scored 1.846 to 1.854. At step 1,500 the long run's rate was 0.00083 and the short run's 0.00043; from there to step 2,500 the long run's seen score fell 1.43 to 1.22 while its validation rose 1.873 to 1.901. Still untested: no run varied the schedule alone. |

**Cost, which the draft never mentioned.** Every arm ran at 80,000 to 87,000 tokens per second, so a character run took 35.2 minutes against 9.6 to 10.0, and the character model will need 2.5 times as many steps as bpe-1024 to write the same text. That, and the window, is the main reason large models use word fragments; "they have more data" is at best part of it.

A last fact-check of the rewritten post found one wrong cell (bpe-1536 on unknown names is 4.37, not 4.38: 4.37475 rounded twice) and eight more sentences that said a little more than the data. The instructive ones: "more than half of bpe-1024's lead on labels is on known names" is 53.5% on the means but runs from a third to all of it seed by seed; "best moments came after about 2,300 steps against about 9,300" is true and proves nothing, because both are 93% of runs whose lengths we fixed (the pilots are the evidence: step 1,500 and about 3,000 against about 10,000); and "these words were not written blind" was true of the auditor, who had seen one pilot, but the build log's addendum went in after all six (0.06 to 0.14 behind). Two time ranges in this log were wrong and are corrected in place: the pilots ran 07:31 to 09:45 and the sweep 09:46 to 13:29.

## Entry 17. Part 7, written before the measurement: does it recite, how does it speak, and which run do we release? (2026-09-21)

Three designers worked on the CPU while nothing trained, and this entry fixes what they settled. As in entries 14 and 15, it is committed before the measurement it governs. **Unlike those, this pre-registration is weaker, and the weakness is disclosed here rather than discovered later:** the memorisation designer prototyped at about a tenth scale and therefore saw model numbers before the rule below was written, and so have I. The rule is still written down, still committed first, and the full-scale measurement is still fixed in advance. But nobody should read it as blind.

### A first correction: my innocent baseline was circular

I measured, first-hand, that no 50-character window of the two validation plays occurs anywhere in the 39 training works (the longest shared run in either direction is 47 characters, and it is a scene heading plus "Enter"). I was about to use that as the innocent rate: what a writer of Shakespeare who has memorised nothing would score.

It is worthless for that. `scripts/make_split.py` **chose** the held-out works by exactly this test: a 50-character window after the same normalisation, asserted to be empty but for one accepted scene heading (entry 10). The validation plays score zero because a zero was the condition of their being validation plays. A reader could not have seen that, and neither did I until a designer said so.

**The honest baseline is leave-one-out over the 39 training works**, which were selected for nothing: for each work, the share of its 50-character windows that occur in the other 38. *Measured*, by the designer and again by me, to the same digits: **0.000190** over all 39 works, and **0.000046** without the five works that genuinely reprint one another (The Passionate Pilgrim, The Sonnets, Love's Labour's Lost, Venus and Adonis, The Rape of Lucrece). What is left is dominated by the editor, not the poet: dedications to Henry Wriothesley, "Dramatis Personæ" headers, "ACT I / SCENE I. London. An ante-chamber". Poet-only it is 0.0000068 (*measured by me*: of the 202 shared windows, 30 are the poet's), about seven windows in a million.

### Memorisation: the measurement, fixed now

**A copy** is a run of characters occurring verbatim in the 39 training works. **Raw characters**, not normalised, because "word for word" is a claim about what the model emitted; the normalised figure is printed on the next line as a sensitivity check. **The unit** is PLAN's 50-character window, stride 1, but the reported object is the whole run-length curve (20, 25, 30, 40, 50, 60, 80, 100 characters and the maximum), because 50 alone is a near-certain zero for model and innocent alike and so carries no information.

**Three statistics, always together:** the share of 50-character windows that are copies (PLAN's headline), the share of characters inside a copy, and the longest common substring.

**The poet and the editor, both co-primary.** Every figure is reported twice: over everything the model wrote, and over Shakespeare's own words with speaker labels, stage directions, scene headings, cast lists and front matter removed by a published rule (the frozen label rule of `gp_thee.evaluation`, plus bracketed directions, ACT/SCENE/PROLOGUE/EPILOGUE/INDUCTION/Dramatis lines, Enter/Exeunt/Exit/Re-enter/Alarum/Flourish/Sennet/Manet lines, and front matter). A window counts only if it holds at least 25 letters, which drops runs of indentation. Neither figure is a footnote: the prototype found that the long passages are editorial furniture, so the answer to "does it recite" turns entirely on this split, and both numbers are published.

**The instrument is an exhaustive scan, not only sampling.** Sampling asks what the model happens to say; the scan asks what it can be made to say. Stage one, the screen: read the true training text in the same overlapping windows `train.evaluate` uses and mark every position where the model's most likely next character is the true one; a run of L consecutive marks is a candidate. Stage two, the confirmation: for every candidate of 40 characters or more, generate greedily from the real 256-token context and count how many characters the model actually writes before diverging. **Only confirmed lengths are reported.** The designer's first version skipped the confirmation and was wrong: 14 candidates of 50 characters or more became 9 confirmed, and the longest fell from 99 to 66. The same scan runs over the validation plays, which is the model's own innocent ceiling.

**Sampling grid**, for the released model: four prompt kinds (unprompted, 256 characters from the training works, 256 characters from the validation plays, the ten-prompt suite), temperatures 0, 0.5, 0.8 and 1.0, 200,000 characters per cell, prompt positions from an arithmetic rule and not chosen, fixed seeds, only windows wholly inside what the model wrote. Reported as counts over a stated number of characters with a Poisson interval, never as a bare share.

**Per checkpoint:** scan all eleven 34-pass runs and the three pilots at `best.pt` and `last.pt`; the full sampled grid for the released model only. The training-length axis (pilot-char-17/34/68) is comparable among itself and not with the sweep runs at the third decimal.

**The verdict, fixed now.** We will say GP-Thee-11M **recites** its training text if the longest confirmed passage of Shakespeare's own words (no speaker label, stage direction, scene heading or cast list, and at least 25 letters) that greedy decoding reproduces from the training works reaches 50 characters **and** is at least twice the longest such passage the same model reproduces from the two validation plays it never read. We will say it **does not recite** if that figure is under 50 characters and exceeds the validation figure by no more than 10. Anything between is reported as "it reproduces N passages of between X and Y characters", with every passage printed in full and classified by hand.

**Controls:** a character 5-gram sampler fitted on the same works (a writer that provably cannot hold more than four characters), and the validation plays shuffled by character, by word and by line. **One device** for the whole measurement, named in the report, with a fixed slice re-run on the other to check the longest confirmed run is unchanged: argmax is discontinuous and entry 16 recorded cross-device agreement only to 1.2e-5 at the worst token.

### The sampler (PLAN step 11)

New: `src/gp_thee/sampling.py`, `scripts/sample.py`, `tests/test_sampling.py`. **`src/gp_thee/train.py` is not edited**, so the eleven existing runs' `samples.txt` stay comparable with each other and with any later run.

**The prompt normaliser**, in one sentence: a character already in the corpus's 97-character alphabet is never touched; one that is not is either mapped to the thing this corpus uses in its place, or the prompt is refused and the character named. NFC first, CRLF to a line break. Straight apostrophe (and prime, acute, grave) to `’`; straight double quote to `“` after a space, line break, `(`, `[`, `{`, `-` or `—` and at the start, `”` elsewhere; tab to a space, taking a space on either side with it, as the cleaning script did; CR, vertical tab, form feed, NEL, LS, PS to a line break; Unicode spaces to a space; en dash, figure dash, horizontal bar to em dash; minus sign to hyphen; zero-width characters and the byte-order mark removed; **everything else refused by name, with its codepoint and the nearest plain letter when NFD offers one**. Trailing spaces are stripped with a note (a space belongs to the next token); an empty prompt is refused. START is never prepended: the only way to feed it is `--new-work`. The notes are printed with the sample and written into the suite file.

**The sampler:** temperature, optional top-k and top-p, a CPU generator from a seed, stop at START or at an optional stop string, a prompt longer than the window fed as its last 256 tokens with the dropped characters reported. **Defaults for everything published: temperature 0.8, no top-k, no top-p, stop at START.** Any other setting is recorded in the header.

**The "speak as a character" wrapper:** the transcript is a page of a play and nothing else; each turn appends a blank line, the speaker's name in capitals, a full stop, a line break, the line, then the answerer's label. The transcript never ends with a spare line break (three in a row is a sonnet boundary in this corpus). With no `--as`, the model casts the answerer itself and that is reported. The reply is the text up to the first blank line, capped at 600 characters, cut back to the last line break if capped; what follows the blank line is discarded and shown only with `--show-cut`. Each turn reports how many characters fell out of the window.

**The ten-prompt suite**, fixed from now on and published per checkpoint as `runs/<name>/suite.txt` and `suite.json`: a speaker label, an unseen name, a stage direction, a sonnet opening, prose typed with a straight apostrophe, a famous line, a mid-word cut, a modern sentence, an instruction ("Write me a poem about a cat."), and a bare blank line; plus an eleventh block, not one of the ten, showing the wrapper on a fixed two-turn scene. The header records the checkpoint's SHA-256, step, validation score, tokenizer checksum, commit, sampler settings, seed rule and device, so two checkpoints can be set side by side (`--compare A B`).

### Which run becomes GP-Thee-11M

**Eligible** only if made by commit `0f72600` (the commit that decided the tokenizer) with a clean tree, seed 1, 2 or 3, 32-bit, and every other setting at its default. Three qualify: `sweep-char-seed-1`, `-2`, `-3`. The eight excluded, each with its reason, are tabled in the designer's report and summarised: four earlier-commit baseline runs from entry 14, one 16-bit run the pre-registered rule declined to adopt, two seed-0 runs, and `pilot-char-34`, which was made by the first version of the code with a different stop grid and which entry 14 showed was a lucky draw (1.7404, the best number in the project, and the one number I must not release).

*A correction to entries 14 and 16: that is four commit stamps among the eleven runs, not three. `8b3a57f` differs from `5c6bd68` only by an edit to `train.py` outside the training loop, and is byte-identical to `0f72600` on every path `git_commit()` counts as mattering.*

**The rule:** the released run is the eligible run with the lowest best-validation bits per character as recorded in its own `result.json`, never a re-score. Ties break by smallest seed, then run name. **Applied: `sweep-char-seed-1`, `best.pt`, step 9,236, validation 1.749198301521595.**

**What that costs, stated before any test score exists.** Two layers of selection sit on the released model's validation number and neither sits on its test number. Within the run, best of 41 stops: the best stop beats the mean of the last ten by 0.0055, 0.0076 and 0.0079 for the three seeds, while the late stop-to-stop wobble is 0.0035 to 0.0048, so the whole edge is the size of twice the noise. Between runs, best of three: the released run is 0.0054 below the three-run mean, and *measured by simulation with this project's own spread, picking the best of three flatters a score by about 0.005 and the best of eleven by about 0.010*. The three eligible runs are 0.0083 apart, which the sweep's own pooled test (0.0130) calls a tie, so **the release buys nothing this project can demonstrate**, and the model card must say so. Altogether the validation figure 1.7492 is about 0.011 optimistic as a description of what this recipe produces. The test figure will not be, because the test works chose nothing.

**Part 8 reports two headlines, not one:** the released model's own test score (the artifact), and the mean and spread of all three eligible runs on test (the recipe). Both in the same table, with the sentence that the three are a tie.

### The retrain on all 44 works: revoked

Entry 11 promised that the released model would be retrained on all 44 works. **That promise is withdrawn here, with the reason.** The whole series argues that an unmeasured claim is not a claim; making the headline artifact the one model that can never be measured would invert that on the last page. The gain is 10.6% more text (*measured*: 507,888 characters on top of 4,811,336 — the 44-work model absorbs the 2 validation plays as well as the 3 test works, and an earlier draft of this entry said 4.8%, having counted only the test works), what that gain buys cannot be measured once the works that would measure it are eaten, and the cost is permanent.

**GP-Thee-11M is the 39-work model with a real test score, and it is the only model in the Hugging Face repository.** If a 44-work model is wanted later it is a separate artifact, trained only after part 8 is published, named `GP-Thee-11M-all44` and never `GP-Thee-11M`, with everything it inherits fixed now: the same tokenizer file unchanged (*measured*: `char.json`'s 97 characters are exactly the whole-corpus alphabet recorded before the split existed, so a refit on 44 works changes nothing), seed 1, the recipe unchanged, 34 passes (about 11,038 steps on 5,319,268 tokens), and no best checkpoint, because there is no validation text: the last checkpoint is released and the card says that no moment of the run was chosen. Its first paragraph must say that it has no held-out score and never can have one.

### Part 8's protocol, fixed now

One script, `scripts/final_evaluation.py`, committed before it is ever run, taking no arguments. It refuses to start unless `docs/release.json` exists and is committed, the tree is clean for `src/`, `scripts/` and `data/`, and `docs/final-evaluation.json` does not already exist. It writes that file once. Every invocation gets a line in this log with its timestamp and outcome. The unlock string `this is the final evaluation` appears in that script and nowhere else.

## Entry 18. Part 7 measured: it does not recite Shakespeare, it recites his editors (2026-09-21)

The measurement of entry 17, run at full scale on the released model, `sweep-char-seed-1/best.pt` (step 9,236), on the GPU. *All measured.* The record is [docs/memorisation.json](memorisation.json).

### The scan, which is the instrument the verdict rests on

Walk the model along the true text and ask at every position whether the character that really comes next is the one it would have written.

| | Training works (4.8M characters) | Validation plays (274,727) |
|---|---|---|
| Its first guess is right | 69.72% of characters | 63.63% |
| Candidate runs of 40 characters or more | 52 | 1 |
| Longest candidate | 99 characters | 40 |
| **Longest confirmed** | **66** | **40** |
| Confirmed runs of 50 or more | 9 | 0 |

The confirmation stage earns its place. *Measured*, every candidate of 50 characters or more, screened → confirmed: 99→27, 68→34, 66→66, 65→15, 60→15, 59→59, 58→58, 57→57, 56→56, 54→54, 52→5, 51→51, 51→51, 50→50. So 14 candidates of 50 or more become 9 confirmed, and the longest confirmed passage comes from a candidate of exactly 66, not from the 99-character candidate, which collapses to 27 once the model must write it from a full 256-character run-up. (An earlier draft of this entry said the 99 "becomes 66", conflating two different candidates. Corrected against a fresh measurement, [docs/candidates.json](candidates.json).) A candidate is not an extraction.

**Every one of the nine confirmed passages of 50 characters or more is the editor's furniture.** In full, longest first:

```
 66  'IRD PART OF KING HENRY THE SIXTH\n\n\n\n\nDramatis Personæ\n\nKING HENRY '
 59  'exandria. A Room in the Palace.\n\nEnter Antony and Cleopatra'
 58  '_Exeunt Antipholus of Syracuse and Dromio of Syracuse._]\n\n'
 57  'andarus’ house.\n\nEnter Pandarus and Cressida.\n\nPANDARUS.\n'
 56  '\nEnter Antipholus of Syracuse.\n\nANTIPHOLUS OF SYRACUSE.\n'
 54  '_]\n\nSCENE III. The same. A Room in the Palace.\n\nEnter '
 51  'ntipholus of Syracuse.\n\nANTIPHOLUS OF SYRACUSE.\nThe'
 51  '\n\nSECOND FRIEND.\nA thousand pieces!\n\nFIRST FRIEND.\n'
 50  'COND\n\n\n\n\nDramatis Personæ\n\nKING RICHARD THE SECOND'
```

Title pages, cast lists, scene headings, entrances and speaker labels. *Measured* character by character against the editor mask, the poet's share of each, longest first: 0, 1, 2, 2, 2, 3, 4, 22, 0 characters. Seven of the nine carry nothing but the blank lines between speeches; the seventh carries `\nThe`; only the eighth carries a line, `A thousand pieces!`, eighteen characters.

**The longest confirmed passage that is mostly the poet's is 44 characters**, and it spans two speeches:

```
'or he hath done me wrong.\n\nKING HENRY.\nWhat '
```

Thirty-two of those characters are Shakespeare's (the end of one line and the first word of the next); the twelve in the middle are `KING HENRY.` and its line break. (An earlier draft of this entry said thirty and thirteen, by eye rather than by the mask.) The longest unbroken stretch of his verse the model reproduces anywhere is **"or he hath done me wrong.", twenty-five characters.**

On the two plays it never read, the model's single candidate of 40 characters or more confirms at 40, and it is a scene heading. Its longest poet-only confirmed passage there is zero.

### The verdict: between the two, as the rule allowed

Entry 17 fixed it: recites if the longest confirmed poet-only passage from the training works reaches 50 characters and is at least twice the validation figure; does not recite if it is under 50 and exceeds the validation figure by no more than 10.

**It is 44 against 0.** Under 50, so not "recites". But 44 exceeds 0 by more than 10, so not "does not recite" either. The rule lands in the middle and says: report it as it stands, with every passage printed, which is what this entry does.

The middle case is not an accident of the model, it is a weakness of the rule that is worth recording: the validation ceiling came out at 0, and the second clause asks the training figure to be within 10 of it, which almost nothing could satisfy. A rule written after seeing that 0 would have said something else. This one was written before, so it stands as written.

### A bug in my own rule, in the direction that flattered the model

The first run of this measurement reported five confirmed passages as "mostly the poet's". Looking at them one by one, as entry 17 requires, four were nothing of the kind:

```
'um and colours. Enter King Henry, Gloucester'      ("Drum and colours. Enter King Henry...")
'LOUCESTER, brother to the King.\nDUKE OF '          (a cast list)
'UCIUS, servant of Timon’s creditors\n'              (a cast list)
'BER, Conspirator against Caesar.\nC'                (a cast list)
```

Two faults, both mine. The front-matter cut-off was written as "everything before the first heading", and `Dramatis Person` was itself in the list of headings, so the cut-off landed on the cast list's own title and left the whole cast list unmarked. And the list of words that open a stage direction had no `Drum`. Both made the model look as though it were reproducing Shakespeare when it was reproducing his editors: **the direction that flatters it.**

Fixed: the front matter now runs to the first `ACT`, `SCENE`, `PROLOGUE` or `INDUCTION` line, and the direction list is longer. The editor's share of the text moves from 9.6% to 10.3% of the training works and 9.3% to 10.1% of the validation plays, which stays symmetric. A second fault of the same kind: the poet-only counting in the sampled grid tested only that a window held 25 letters, so a scene heading with enough letters in it counted as the poet's. It now also asks whether the editor wrote that passage where it occurs in the corpus. Both are pinned by tests that name the four passages above, so they cannot quietly come back. The measurement was re-run from scratch with the corrected rule.

Two of entry 17's figures were also re-measured here rather than taken on trust. The cost of selection: *measured* by simulation over 400,000 draws against both spreads this project has published, the best of 3 sits 0.0046 (sd 0.0054) to 0.0060 (sd 0.0071) below the mean, and the best of 11 sits 0.0086 to 0.0112 below it, so entry 17's "about 0.005 and about 0.010" stands. The best of 41 stops: `log.csv` carries exactly 41 scored stops, as claimed.

The lesson is the one from parts 3 to 6, in a new place: a rule I wrote to protect a measurement had a hole in it, and the hole leaned the way I would have liked the answer to lean.

### What it copies when simply asked to write

20,007 characters per cell, four kinds of prompt, four temperatures, seeds fixed. (Entry 17 said 200,000 characters per cell. **That is a deviation, and the reason is speed:** the sampler re-reads its whole window for every character, so the grid at full size would take nine hours. The scan above, which is the primary instrument, is at full scale over every one of the 4.8 million characters. The sampled grid is secondary and is reported at a tenth of the promised size.)

Copies of 50 characters or more, out of 20,007 characters written, by the corrected rule, with the poet-only count beside it:

| Prompt | t=0 | t=0.5 | t=0.8 | t=1.0 |
|---|---|---|---|---|
| unprompted | 0 | 0 | 0 | 5 (0 of the poet's) |
| from the training works | 0 | 1 (0) | 1 (0) | 0 |
| from the validation plays | 7 (0) | 0 | 0 | 0 |
| the ten-prompt suite | 0 | 0 | 0 | 0 |

Every copy of 50 characters or more that sampling found is a scene heading, such as `'t._]\n\nSCENE III. The same. A Room in the Palace.\n\nEnter '` (56 characters, written greedily from a validation prompt). **Not one is a line of verse.** Sampling also demonstrates why it is the weaker instrument: prompted with its own training text at temperature 0.8, the model produced one copy in 20,007 characters, while the scan proves there are 52 places in the corpus where it could have produced 40 or more.

### How this answers part 5

Part 5 found a gap between the score on text the model had trained on (1.37 bits per character) and on the validation plays (1.75), retracted the claim that the gap shows memorisation, and promised a direct measurement in a later part. This is it. The scan opens the gap up and shows what is inside: 69.72% next-character agreement on the training works against 63.63% on the plays it never read, six points, and the passages behind those six points are cast lists, scene headings and entrances. The model has learned the *shape* of an edition of Shakespeare very well, and its verse hardly at all, word for word.
