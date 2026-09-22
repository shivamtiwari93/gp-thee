# Building GP-Thee, part 1: prerequisites and setup — simplified

*We define a tiny language-model experiment, choose a reproducible stack, check the Mac GPU, and estimate the model's size before training anything.*

> This is the simplified edition. For the full technical reasoning, measurements, and parameter derivation, read [the original Part 1](01-prerequisites-and-setup.md).

## What you will learn

- How a strict product constraint can make a machine-learning experiment easier to understand.
- Why environment pinning and hardware checks are part of correctness, not housekeeping.
- How a model can have a known parameter count before it learns anything.
- Why the first design is a testable hypothesis rather than a claim that the architecture is optimal.

## The experiment in one sentence

Suppose the complete works of Shakespeare were the only text in existence. Could we build a language model inside that closed universe?

That is GP-Thee: a GPT project built from a corpus containing 38 plays, 154 sonnets, and 5 poems. It gets no pretrained weights, no text from the internet, and no borrowed tokenizer whose vocabulary was learned elsewhere.

A few terms are enough to follow the project:

- A **language model** predicts what text comes next.
- A **GPT** is a widely used architecture for doing that prediction.
- A **tokenizer** turns text into numbered pieces called tokens.
- **Parameters**, or weights, are the numbers adjusted during training.

The closed-universe rule is not a claim that this is the best way to build a useful model. It is an experimental control. If GP-Thee produces a Shakespeare-like speech, its behaviour must come from this corpus and the learning process, not hidden knowledge imported through a model or tokenizer.

The expected product is therefore modest: an in-character autocomplete with speaker turns and stage directions, not an assistant that answers questions. Setting that expectation early matters. A system should be judged against the job it was designed to do.

## Start with constraints, not tools

The corpus is only 5.4 MB, about 963,000 words. That is tiny by modern language-model standards. The computer used for the project is not tiny: an Apple M5 Max MacBook Pro with an 18-core CPU, a 40-core integrated GPU, and 128 GB of shared memory.

That imbalance tells us where the real risk is. Hardware capacity is not the limiting factor. Data quantity and experimental discipline are.

At this stage, we estimated that one training run would take 10 to 30 minutes, based on published results from older Macs. That was an estimate, not a measurement; later parts measure the real runtime. Keeping estimates visibly separate from measured facts is a recurring rule in this series.

The same project should run on smaller Apple Silicon Macs, although a machine with 8 GB of memory will need a smaller batch—that is, less text processed in each training step. Linux should need only a device-name change from Apple's `mps` backend to NVIDIA's `cuda`. Windows GPU setup is less direct because the lockfile selects a CPU-only PyTorch build there. These are reasonable expectations, but only the M5 Max setup was tested at this point.

## The stack: boring on purpose

The project uses four main pieces:

| Need | Choice | Why it matters |
|---|---|---|
| Reproducible environment | `uv` | Manages Python, the virtual environment, packages, and the lockfile in one place. |
| Python | `uv`-managed CPython 3.14 | Keeps the project independent of Homebrew's upgrade schedule. |
| Deep-learning framework | PyTorch 2.14.0 | Has clear reference implementations and runs on the Mac GPU. |
| Saved token arrays | NumPy | Stores the corpus as compact integer arrays. |

The key product decision is reproducibility. Homebrew's Python can move underneath a project. GP-Thee instead asks `uv` for its own interpreter and commits `uv.lock`, which records the exact dependency set. `.python-version` pins Python 3.14, though not its patch release. A clone gets a controlled environment rather than “something recent enough.”

PyTorch won over Apple's MLX even though MLX is designed for Apple chips. Speed did not decide it; neither had been benchmarked on this machine yet. PyTorch had three practical advantages:

1. **A close reference.** Andrej Karpathy's nanoGPT implements almost this exact experiment in PyTorch.
2. **A larger debugging trail.** Mature error messages and years of answered questions reduce learning risk.
3. **Portability.** The same model code can move to NVIDIA hardware by changing the device.

MLX remained a useful later comparison. The lesson is broader than these libraries: when performance is not yet the bottleneck, choose the tool that makes mistakes easiest to find.

We also deliberately avoided `transformers`, `tiktoken`, and pretrained tokenizers. They are excellent tools, but their learned vocabularies contain information from outside Shakespeare. Installing them would weaken the experiment's central rule.

## Pin risky infrastructure and test it

PyTorch was pinned to exactly 2.14.0, not merely “2.14 or newer.” Research found three relevant Apple-GPU reports: incorrect 32-bit multiplication for a transposed matrix in earlier releases, causal attention leaking future positions in 16-bit arithmetic, and repeated 16-bit calls producing different answers on M5 chips.

These were **silent** failures: a program could finish normally and return the wrong result. That is more dangerous than a crash because a plausible training curve can hide bad arithmetic.

The response had two parts:

- Freeze the known software versions and avoid upgrading the operating system or PyTorch mid-project.
- Run a smoke test that compares the GPU's 32-bit results with the CPU and with a higher-precision reference.

The first smoke test reported an exact GPU-versus-CPU difference of zero for a large matrix multiplication. That sounds ideal, but it raised a new question: was the comparison accidentally testing a value against itself?

So the test was tested. It now confirms that results really live on different devices, checks both against 64-bit arithmetic, and includes a control with deliberately different inputs that must fail the comparison. The important results were:

- The ordinary and transposed matrix multiplications were bit-identical across this CPU/GPU pairing.
- Layer normalization, linear layers, activation, and causal attention differed only by tiny amounts expected from 32-bit arithmetic.
- Changing the last token changed no earlier attention output, exactly as required.
- The deliberately different control produced a clear difference.

The exact matrix match turned out to be real: the CPU and GPU happened to add products in the same order with the same fused instruction on this hardware and software combination. That is an observation, not a guarantee to carry elsewhere.

One boundary is important. [The smoke-test script](../scripts/smoke_test_gpu.py) uses 32-bit arithmetic throughout. It exercises the path planned for the first runs and the reported transposed-matrix case; it does **not** reproduce the two cited 16-bit failures. Those reports justify caution and version pinning, while later full-model checks cover more of the actual system.

## The shortest useful setup

The full original shows how the repository was initialized. A reader reproducing it today needs only:

```bash
brew install uv
git clone https://github.com/shivamtiwari93/gp-thee && cd gp-thee
uv sync
```

The `brew` line assumes macOS; on Linux or Windows, install `uv` using the method for that operating system and continue from `git clone`. The environment definition lives in [`pyproject.toml`](../pyproject.toml), and exact resolved versions live in [`uv.lock`](../uv.lock).

`uv sync` downloads the pinned Python environment and writes `.venv/`; its first run is the slow one and depends on network speed. The checks below run from the repository root after that command finishes.

Before running the GPU test, confirm which environment `uv` selected:

```bash
uv run python -c 'import sys, torch; print("Python:", sys.version.split()[0]); print("PyTorch:", torch.__version__); print("MPS available:", torch.backends.mps.is_available())'
```

On the machine used for this project, the three important lines are:

```text
Python: 3.14.7
PyTorch: 2.14.0
MPS available: True
```

A different Python patch number is possible because `.python-version` pins Python 3.14, not 3.14.7 exactly. On a non-Mac, `MPS available` will be `False`; do not run the MPS-only smoke test there without adapting its device.

On Apple Silicon, run it now:

```bash
uv run python scripts/smoke_test_gpu.py
```

The smoke test writes no files and took about three seconds on the project machine. A successful run should show `causal leak (must be 0)` as zero and `control (must NOT be 0)` as non-zero. The precise rounding differences and speed depend on the machine.

## One early decision we reversed

The source is Project Gutenberg's eBook #100, one plain-text file of 5,422,721 bytes. The first plan kept that raw file outside the repository because Project Gutenberg's name is a trademark.

That caution created a bigger reproducibility problem. Project Gutenberg can revise a file, so a future download may no longer match the version used for every cleaning rule and measurement. An educational project that hides its dataset is also hard to audit.

After reading the licence terms, the project stored the unmodified file in the repository with the required notice and checksum. This was not legal advice; it was a documented project decision. The exact source and fingerprint are recorded in [`data/raw/README.md`](../data/raw/README.md).

The mistake is useful: risk controls should be evaluated as a system. Reducing one perceived risk can create a larger integrity risk somewhere else.

## Knowing the model size before training

Training changes parameter **values**, not the number of parameters. Once the architecture is chosen, its size is arithmetic.

The first design used:

| Choice | Value |
|---|---:|
| Transformer blocks | 6 |
| Width of each token representation | 384 |
| Attention heads | 6 |
| Context window | 256 tokens |
| Raw-file character vocabulary | 100 |

Most parameters sit in the attention and feed-forward matrices inside the six blocks. Token and position embeddings add smaller tables. The output reuses the token-embedding matrix, a technique called **weight tying**, so it does not add a second full output matrix. Biases and layer-normalization shifts are disabled; enabling them in this historical 100-character design would add 25,828 parameters.

The resulting formula gives **10,758,528 parameters**, about 10.8 million. [A small PyTorch script](../scripts/count_params.py) builds the same shapes and independently reaches the same number. That check caught a naming issue before training: the working name GP-Thee-10M rounded the size the wrong way, so it became **GP-Thee-11M**.

This table uses the 100 characters known from the raw file at this point in the story. Cleaning in Part 2 changes the alphabet. Preserving that historical value matters because the table documents the decision as it was made, while the script labels both the historical and later configurations.

Try the independent count yourself:

```bash
uv run python scripts/count_params.py
```

This check writes no files and normally takes under a second. Look for these two rows. `match=True` means the formula and the model PyTorch constructed agree exactly:

```text
current char model: 6L/384d, V=98            pytorch= 10,757,760  hand= 10,757,760  match=True
Part 1 base: 6L/384d, V=100                  pytorch= 10,758,528  hand= 10,758,528  match=True
```

The first row is the eventual 98-piece character tokenizer, including the START marker introduced in Part 3. The second reproduces the historical 100-character estimate in this part.

## Parameters are not the whole memory bill

At four bytes per 32-bit parameter, the model itself is about **43 MB**. Training also stores a gradient and two optimizer averages for every parameter. Together with the weights, those four copies measured about **174 MB**.

An early draft stopped there and claimed the whole job would fit under 1 GB. That was wrong by about five times.

During training, PyTorch must also retain intermediate results from the forward pass so it can compute gradients backwards through the model. These **activations** grow with batch size. A rough probe used about:

- **1.3 GB** for 16 sequences of 256 characters;
- **4.8 GB** for 64 sequences, the planned batch.

The correction changes the practical advice. On the 128 GB development machine, 5 GB is easy. On an 8 GB Mac, the right response is to reduce the batch, not assume the model itself is too large. It also demonstrates why an architecture's parameter count is not a capacity plan for training.

## Why start near 11 million parameters?

There is no theorem saying 10.8 million is the right size for 5.4 MB of Shakespeare. It is a hypothesis chosen to test.

A well-known scaling result suggests roughly 20 training tokens per parameter when fresh data is abundant. Reading each character as a token would point toward only about 270,000 parameters—roughly forty times smaller than GP-Thee.

But this project has the opposite constraint: it will reuse the same small corpus for many passes. Research on data-constrained models found that the first few repeated passes can still be useful, while the value fades and is nearly exhausted by around 40 passes. That work also found that the best model for a fixed dataset can be larger than the simple 20-to-1 rule suggests. Its experiments were much larger than this one, so applying the result here is an extrapolation, not proof.

nanoGPT provides a closer practical reference. It uses this same basic shape on a 1 MB Shakespeare sample and learns successfully, but eventually memorizes. GP-Thee has about five times more text. A 5,000-step run would revisit nanoGPT's sample around 80 times but this corpus around 17 times.

The plan was therefore comparative, not dogmatic: begin at 10.8 million parameters, then train smaller **4.8 million** and larger **37.9 million** versions using the same data, tokenizer, and context. The expectation was that the largest would memorize more. Only the later measurements could decide.

## Takeaways

- A narrow premise makes provenance and success criteria easier to inspect.
- Reproducibility requires pinning the interpreter, packages, data bytes, and hardware assumptions.
- A successful test needs a control showing that it can fail.
- The smoke test established confidence in the planned 32-bit path, not every precision or future model operation.
- Parameter count is known from architecture; training changes the numbers, not how many exist.
- Model weights were only 43 MB, while training activations pushed the real memory estimate to about 4.8 GB.
- The 11-million-parameter design was a starting hypothesis with explicit smaller and larger comparisons planned.

At the end of Part 1 there is still no model. There is a reproducible environment, a pinned corpus, a checked 32-bit GPU path, and a model-size calculation that PyTorch agrees with.

[Next: Part 2, the data — simplified →](02-the-data-simplified.md)
