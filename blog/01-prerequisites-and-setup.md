# Building GP-Thee, part 1: prerequisites and setup

*What we started with, what we installed and why, and how we know the model will have 10.8 million parameters before we have trained anything.*

## The premise

Suppose the only text that ever existed was the complete works of Shakespeare. No internet, no Wikipedia, no other books. Could you build a language model in that universe?

That is GP-Thee: a GPT trained from scratch on 38 plays, 154 sonnets and 5 poems, and nothing else. The rule is strict. No pretrained weights. No borrowed tokenizer, because a tokenizer learned from the web is knowledge smuggled in from outside the universe. No extra text.

Four words we will lean on throughout:

- A **language model** is a program that is given some text and predicts what comes next.
- A **GPT** is the most common design for one. It is the design behind ChatGPT.
- The model never sees letters directly. A **tokenizer** first chops the text into small pieces called **tokens** (single characters, or fragments of words) and gives each piece a number.
- **Weights**, also called **parameters**, are the numbers a model learns during training. **Pretrained** weights are ones somebody else has already trained, usually on text from the internet.

The rule is also what makes this a good project to learn from. The whole corpus is 5.4 MB, and the full pipeline is small enough to read end to end: data, tokenizer, model, training, evaluation, and sampling (generating new text from the trained model). We expect each training run to take 10 to 30 minutes on the laptop described below. That is an estimate from other people's published runs on older Macs, and we will measure it in a later part.

This series documents every step, including the mistakes. There are several in this part alone. It was built with Claude Code as a pair programmer. Everything is in the repo: [github.com/shivamtiwari93/gp-thee](https://github.com/shivamtiwari93/gp-thee).

One expectation to set now. The result will be an in-character autocomplete, not an assistant. It should write convincing pseudo-Shakespeare with proper speaker turns and stage directions. It will not answer questions, and it will never have heard of anything that is not in the plays.

## Where we started

### The machine

| | |
|---|---|
| Computer | MacBook Pro, Apple M5 Max |
| CPU | 18 cores |
| GPU | 40 cores, built into the same chip |
| Memory | 128 GB, shared between CPU and GPU |
| Free disk | about 1.6 TB |
| OS | macOS 26.5.2 |

This is far more computer than the project needs. As you will see below, the model is 43 MB, and a training step needs roughly 1 to 5 GB depending on how much text we process at once. The thing that limits this project is not the hardware. It is the size of the corpus: about 963,000 words is a tiny amount of text by language-model standards, and most of the design decisions in this series are about coping with that.

**Can you do this on your computer?** Probably, but we have only tested ours.

- Any Apple Silicon Mac on macOS 14 or later should work with the same commands. A base-model chip will be slower than the timings we report, and with 8 GB of memory you will want to process less text per step (a smaller batch; see below).
- On Linux, install `uv` with the installer from its documentation rather than Homebrew. The same lockfile will also download NVIDIA's GPU libraries there, so expect several GB instead of our 650 MB, and the code will need one word changed (the device name, from `mps` to `cuda`).
- On Windows the lockfile installs a CPU-only PyTorch, so using a GPU needs PyTorch's own package index.

### What was already installed

| Tool | Version | What it is for |
|---|---|---|
| Homebrew | 7.0.4 | installs command-line tools on a Mac |
| Python | 3.14.6, from Homebrew | not used directly; see below |
| Xcode | 26.6 | Apple's developer tools, which include the compilers and `git` |
| git | 2.50.1 | version control |
| GitHub CLI (`gh`) | 2.96.0 | creates the repo and pushes from the terminal |

### What was missing

Everything specific to machine learning. No PyTorch, no NumPy, no tokenizer libraries, and no environment manager beyond the `venv` module that ships with Python. That is a clean start, which is what you want for a project that claims to be reproducible.

## The stack we chose

| Layer | Choice | Why |
|---|---|---|
| Environment manager | `uv` 0.12.17 | one tool for the Python version, the virtual environment and a lockfile |
| Python | CPython 3.14.7, installed and managed by `uv` | isolated from the system, so a Homebrew change cannot break the project |
| Deep learning framework | PyTorch 2.14.0 | the clearest reference material for this exact task, and it runs on the Mac's GPU |
| GPU backend | MPS (Metal Performance Shaders) | Apple's library of GPU routines. PyTorch's `mps` device is built on it; you select it with `device="mps"` |
| Arrays | NumPy 2.5.3 | stores the tokenized corpus on disk as a flat array of integers |

### Why `uv`, and why not the Python that was already there

Homebrew's Python refuses `pip install` by design. It is marked "externally managed", so that you do not break tools that depend on it. A virtual environment (a private folder of packages for this one project, here `.venv/`) is mandatory, not optional.

We went one step further and told `uv` to use its own copy of Python rather than Homebrew's:

```toml
[tool.uv]
python-preference = "only-managed"
```

The reason is control. Homebrew's Python exists for Homebrew's own packages. It gets upgraded, and eventually removed, on Homebrew's schedule, and a virtual environment that points at it can break when that happens.

With a managed interpreter and a lockfile (`uv.lock`, which records the exact version of every package, like `package-lock.json` or `Cargo.lock`), anyone who clones the repo and runs `uv sync` gets a `uv`-managed CPython 3.14 and exactly the package versions we used. The file `.python-version` pins the minor version, 3.14, not the patch release. We ran 3.14.7.

### Why PyTorch, and why not Apple's MLX

Apple has its own framework, MLX, built for these chips. It was the runner-up.

At this scale, speed should not decide it. Our estimate for PyTorch is 10 to 30 minutes per run, and we expect MLX to be in the same range, though we have measured neither yet. What decided it:

- **Reference material.** Andrej Karpathy's nanoGPT is a line-by-line PyTorch implementation of almost exactly this model, trained on a Shakespeare sample. When our numbers look wrong, there is something to compare against.
- **Debugging help.** Both frameworks let you stop anywhere and print an intermediate value. PyTorch's advantage is everything around that: clearer error messages and a decade of answered questions. That matters when you are learning.
- **Transfer.** The same code runs on NVIDIA GPUs by changing the device name.

Porting the model to MLX and timing both is a planned later experiment. We could not find a published training benchmark for the M5 Max in either framework, so that comparison would be new information.

### Why the version is pinned to exactly 2.14.0

```toml
dependencies = [
    "numpy>=2.5.3",
    "torch==2.14.0",
]
```

PyTorch's support for Apple GPUs is younger than its support for NVIDIA's, and during research we found bug reports that matter for exactly what we are about to do. All three are *silent*: no error, no warning, just wrong numbers.

- **Wrong matrix multiplies** in ordinary 32-bit arithmetic, when the left-hand matrix is a transposed view. Results were off by 10 to 30% depending on what memory had been used earlier in the program. Reported on versions 2.7 through 2.13.0, and at the time of writing the report is still open: 2.14 avoids the problem rather than fixing its root cause ([#193487](https://github.com/pytorch/pytorch/issues/193487)). A GPT multiplies by transposed matrices constantly.
- **Attention that can see the future.** In 16-bit arithmetic, the operation that is supposed to stop each token from looking at later tokens let three out of every four positions peek ahead. Affected 2.12.1 and earlier, fixed in 2.13 ([#195910](https://github.com/pytorch/pytorch/issues/195910)). For a model whose whole job is predicting the next token, seeing the next token is the worst possible bug: training would look spectacular and the model would be useless.
- **Different answers on repeated calls**, specific to M5 chips, in 16-bit arithmetic. The report shows a difference of 123.5 between two identical calls. PyTorch 2.13 added a workaround ([#180776](https://github.com/pytorch/pytorch/issues/180776)).

So we want 2.14.0 and nothing else. The lockfile already freezes every package at an exact version, NumPy included. The `==` in `pyproject.toml` is a second guard: it stops a later `uv add` or `uv lock --upgrade` from quietly moving PyTorch. We will not update macOS or PyTorch until the project is finished, and we check the GPU's answers against the CPU's in the smoke test below.

### What we deliberately did not install

| Not installed | Why |
|---|---|
| `tiktoken`, `transformers`, pretrained tokenizers | They carry vocabularies learned from internet text. That breaks the rule. We will write our own tokenizer and train it on Shakespeare only. |
| MLX | Later, as a comparison. |
| `matplotlib` | Not needed until there is something to plot. It gets added when we train. |
| Jupyter | Every step is a plain script, so every result can be regenerated with one command. |

### What came along for the ride

We asked for two packages and got eleven (`uv pip list` shows a twelfth, the project itself). On a Mac, PyTorch brings nine dependencies with it:

| Package | Why PyTorch wants it |
|---|---|
| `sympy`, and its dependency `mpmath` | symbolic maths, used to reason about the shapes of tensors (PyTorch's word for arrays of numbers) |
| `networkx` | graph algorithms, used by PyTorch's compiler to decide which intermediate results to keep and which to recompute |
| `jinja2`, and its dependency `markupsafe` | text templates, used by the compiler to generate GPU code |
| `filelock` | stops two processes writing to the same cache file |
| `fsspec` | reading and writing checkpoints (saved copies of a model) on different kinds of storage |
| `setuptools` | builds C++ extensions on the fly |
| `typing-extensions` | newer type-hint features |

Most of these serve `torch.compile`, an optional step that compiles a model into faster GPU code. We will not use it: on Apple GPUs it is still labelled an early prototype. We checked which of the nine actually get loaded. A plain `import torch` loads only `typing-extensions`, and one training step adds `sympy` and `mpmath`. The other six are installed but idle.

The whole environment is about 650 MB on disk, of which PyTorch itself is about 570 MB. Compare that with the corpus at 5.4 MB and the model we are about to build at 43 MB. The tools are more than ten times the size of the corpus and the model put together.

## The setup, command by command

```bash
brew install uv

cd ~/Documents/GitHub
uv init --package --name gp-thee --python 3.14 --vcs git gp-thee
cd gp-thee

# add the [tool.uv] block shown above to pyproject.toml, then:
uv add "torch==2.14.0" numpy
```

`uv init --package` creates a small installable Python package (`src/gp_thee/`), a `pyproject.toml`, a pinned Python version and a git repository. `uv add` installs the packages, records them in `pyproject.toml`, and writes exact versions to `uv.lock`.

To reproduce our environment from the repo, you need only this:

```bash
brew install uv
git clone https://github.com/shivamtiwari93/gp-thee && cd gp-thee
uv sync
```

## Does the GPU work? A smoke test, and a check on the check

Given the bug reports above, we checked that the operations a GPT is made of give the same answers on the GPU as on the CPU before building anything on top of them. The script is [scripts/smoke_test_gpu.py](../scripts/smoke_test_gpu.py):

```bash
uv run python scripts/smoke_test_gpu.py
```

The first version of this test multiplied two 2048×2048 matrices on each device, compared the results, and printed a difference of exactly `0.0`.

That looks like good news, and it should make you suspicious. Two devices often disagree in the last digit or two. A single add or multiply is rounded the same way everywhere, but a matrix multiply adds up thousands of products, and the rounded total depends on the order in which they are added. A GPU usually adds in a different order from a CPU. So an exact zero across four million outputs is also what you would see if the test were broken, for example if it accidentally compared a result with itself.

So we tested the test, three ways:

1. **Are the results where we think they are?** The script asserts that one result lives on the CPU and the other on the GPU.
2. **Are they right, and not merely equal?** Both are compared with a 64-bit reference. Two tables full of zeros would match each other too.
3. **Can the comparison fail at all?** A control runs the same comparison on inputs that are deliberately different. If the control ever reports zero, the script stops with an error.

| Check | GPU vs CPU | Each vs 64-bit reference |
|---|---|---|
| Matrix multiply, 2048×2048 | bit-identical | 0.00054, on values averaging 36: normal for 32-bit |
| Matrix multiply with a transposed left side (the case in bug #193487) | bit-identical | 0.00047 |
| Three other operations the model uses (LayerNorm, Linear, GELU) | 0.0000017 | 0.000003 |
| Causal attention, in the shape our model will use | 0.0000007 | 0.000001 |
| Does attention leak the future? Change the last token, check no earlier position changes | exactly 0, as it must be | |
| Control: inputs that really are different | 0.16, so the comparison can fail | |
| 50 matrix multiplies | 68 ms on the GPU, 479 ms on the CPU | |

The multiply really is bit-identical on this machine, and we can say why. [scripts/explain_bit_identical.py](../scripts/explain_bit_identical.py) recomputes 60 of the four million outputs one product at a time, adding strictly in index order with a fused multiply-add (one rounding per step instead of two). All 60 match both devices exactly. Adding in reverse order matches 1 of 60. So Apple's CPU maths library and its GPU routine happen to add in the same order, with the same instruction, and identical steps give identical bits. That is a property of this pairing of hardware and software, not something to rely on. The other rows show the tiny differences you would normally expect.

The test was fine. It took a few minutes to find that out, and it was worth it: a test that cannot fail tells you nothing.

This is a smoke test, not proof. The full check, one real training batch through the whole model on both devices, comes once there is a model.

## The dataset, briefly

The corpus is Project Gutenberg's eBook #100, *The Complete Works of William Shakespeare*: one plain-text file of 5,422,721 bytes. We downloaded it once, verified its SHA-256 checksum, and stored it read-only. Part 2 covers why we chose it over the alternatives we checked (Karpathy's tiny-shakespeare sample, several Hugging Face datasets, the Folger texts and a Kaggle spreadsheet) and how we clean it.

One decision is worth telling here because we got it wrong first. The initial plan kept the raw file out of the repository, because it carries Project Gutenberg's name and that name is a trademark. That was too cautious. As we read it (we are not lawyers), Project Gutenberg's licence allows free, no-charge redistribution of the file provided a specific notice, with a link to the full licence, is displayed with it. More importantly, an educational project that hides its dataset is missing the point, and Project Gutenberg revises its files from time to time, so a download script with a pinned checksum will eventually stop working. The file is now [in the repo](../data/raw/), unmodified, with the notice beside it.

## How we know the parameter count before training

This question came up when naming the project. The working name was GP-Thee-10M. How can anyone know the "10M" before anything has been trained?

Because the number of parameters is a property of the architecture, not of the training. Training changes the *values* of the parameters. It never changes how many there are. You choose the shape of the model, and the count follows by arithmetic.

### What a parameter is

A parameter (also called a weight) is one adjustable number inside the model. They are stored in matrices. A layer that turns a list of 384 numbers into another list of 384 numbers is a 384×384 matrix, because each of the 384 outputs is a weighted sum of all 384 inputs. That is 147,456 parameters. Counting a model's parameters means listing its matrices and adding up their sizes.

### What the model does, in four sentences

Our first model works on single characters: one token is one character, so the vocabulary is the 100 distinct characters in the file. The model reads up to `T` tokens and predicts the next one. Inside, every token is represented by a list of `d` numbers, and that list is refined by passing it through `L` identical stages called transformer blocks. What is inside a block is in the diagram below.

### The shape we chose

| Symbol | Meaning | Our value |
|---|---|---|
| `d` | width: how many numbers represent each token | 384 |
| `L` | layers: how many transformer blocks are stacked | 6 |
| `V` | vocabulary: how many distinct tokens exist | 100 (the distinct characters in the raw file; cleaning in part 2 may change this by a few, at 384 parameters per character) |
| `T` | context: how many tokens the model can look back over | 256 |

### Where the parameters live

```
token ids
   │
   ├─ token embedding      V × d     one row of d numbers per character
   ├─ position embedding   T × d     one row per position in the context
   │
   ▼
┌─ block (× L) ───────────────────────────────────────────────┐
│  layer norm            d                                    │
│  attention             4 × d²   queries, keys, values, out  │
│  layer norm            d                                    │
│  feed-forward          8 × d²   d → 4d, then 4d → d         │
└─────────────────────────────────────────────────────────────┘
   │
   ├─ final layer norm     d
   └─ output layer         0        reuses the token embedding
```

This is a map of where the parameters live, not a full wiring diagram. Steps that have nothing to learn are left out. Two are worth knowing about now: the token row and the position row are added together before the first block, and inside a block the output of each step is added back onto its input rather than replacing it (a residual connection, which we will meet when we write the model).

**Embeddings.** Token ids are just the characters' numbers, 0 to 99. An embedding is a lookup table: row *i* holds the `d` numbers that stand for token *i*. The position embedding does the same for "first character, second character, and so on", because the blocks have no other way to know the order of the text.

**Attention, `4d²` per block.** Attention is the step where each token looks back at the earlier tokens and pulls in information from the ones that matter. To do that, each token is turned into a query (what am I looking for?), a key (what do I offer?) and a value (the information itself), which is three `d×d` matrices. A fourth `d×d` matrix mixes the result. Our model runs 6 of these lookups side by side (6 attention heads), but heads do not change the count: each head works on 384 / 6 = 64 of the numbers, so six heads are the same four matrices cut into 6 parts.

**Feed-forward, `8d²` per block.** One matrix expands each token from `d` to `4d` numbers, and another brings it back. That is `d×4d + 4d×d`.

**Layer norm, `d` each.** A layer norm rescales each token's `d` numbers so they stay in a sensible range, and it learns one scale factor per number.

So each block has `12d²` parameters in its matrices, plus `2d` for its two layer norms.

**The output layer costs nothing.** To predict the next character, the model has to turn its final `d` numbers into one score for each of the `V` possible characters. That takes a `d×V` matrix, the same shape as the token embedding turned on its side. A standard trick called weight tying uses the same matrix for both jobs. Here it saves only 38,400 parameters (0.4% of the model), but it will save 786,432 once the vocabulary grows to 2048. In published experiments on word-level models it also improved quality (Press and Wolf, 2017). Nobody has shown that for a 100-character vocabulary, so we adopt it because GPT-2 and nanoGPT do, and list it as something to test.

**One detail makes the layer-norm rows `d` and not `2d`.** A standard layer norm learns two lists of `d` numbers, a scale and a shift, and a standard matrix layer has an extra learned list called a bias. We leave out every shift and bias. That is a common modern simplification (nanoGPT's training script turns them off by default) and it keeps the formula short. With them the count would still be plain arithmetic, just longer: 25,728 more parameters, a 0.24% difference. In PyTorch this means writing `bias=False` on every `nn.Linear` and `nn.LayerNorm`, because the defaults include them.

```
parameters = 12·d²·L  +  (2L + 1)·d  +  V·d  +  T·d
```

### The arithmetic

```
12 × 384² × 6    = 10,616,832    attention and feed-forward, 6 blocks
13 × 384         =      4,992    layer norms (2 per block, 1 final)
100 × 384        =     38,400    token embedding (shared with the output)
256 × 384        =     98,304    position embedding
                   ----------
                   10,758,528    about 10.8 million
```

Notice where the bulk is. 98.7% of the parameters are in the blocks. The vocabulary barely registers. That will change when we move from single characters to a tokenizer whose 2048 tokens are fragments of words: the token embedding grows from 38,400 to `2048 × 384 = 786,432` parameters, which takes the total to 11,506,560, about 11.5 million.

Because the count moves with choices like this, the repository is just `gp-thee`, and each released model carries its size in its name. The arithmetic also corrected that name. 10,758,528 rounds to 11 million, not 10, so the working name GP-Thee-10M became `GP-Thee-11M`. The exact figure goes on the model card.

### Checking the arithmetic against PyTorch

A formula is a claim, so we tested it. [scripts/count_params.py](../scripts/count_params.py) builds the model's skeleton in PyTorch, with the real matrix shapes but no training, and asks PyTorch to count:

```bash
uv run python scripts/count_params.py
```

| Configuration | PyTorch counts | Formula gives |
|---|---|---|
| GP-Thee: 6 layers, width 384, 100 characters | 10,758,528 | 10,758,528 |
| the same with a 2048-entry vocabulary | 11,506,560 | 11,506,560 |
| smaller: 6 layers, width 256 | 4,813,056 | 4,813,056 |
| larger: 12 layers, width 512 | 37,943,808 | 37,943,808 |

They agree to the last parameter. If you know nanoGPT, you may remember 10.65 million for this shape. That is the same model with a 65-character vocabulary, and nanoGPT leaves the position embedding out of its headline number: `10,616,832 + 4,992 + 65 × 384 = 10,646,784`.

### What 10.8 million parameters costs

Each parameter is stored as a 32-bit float, PyTorch's default, which is 4 bytes.

- **The model:** `10,758,528 × 4 bytes ≈ 43 MB`.
- **Training it, the fixed part.** Training nudges every parameter a little at each step. For each parameter that takes three more numbers: its gradient (which way to nudge it, and how hard) and two running averages of past gradients, which the optimizer (the routine that does the nudging; ours is called AdamW) uses to keep the nudges steady. With the weights themselves that is four copies, about 172 MB. We measured 174 MB.
- **Training it, the part that grows.** The model trains on a batch: a chunk of text processed in one step. To work out the gradients, PyTorch keeps every intermediate result of the forward pass until the backward pass has used it. This depends on the batch size, not the parameter count, and it is much larger. A rough probe of this shape held about 1.3 GB with 16 sequences of 256 characters per batch, and about 4.8 GB with 64, which is the batch size nanoGPT uses.

An earlier draft of this post said the whole job fits in under 1 GB. That was wrong by a factor of five: it counted the four copies and forgot the intermediate results, which dominate. A reviewer caught it, and we then measured it. On a 128 GB machine 5 GB is nothing. On an 8 GB Mac it means choosing a smaller batch, not giving up.

### Why about 10 million, and not 100 thousand or 100 million

This is the real design question, and we should be honest that our answer is a hypothesis to test, not a fact.

A well-known result from DeepMind (the "Chinchilla" paper) says that if you have a fixed budget of computation, you get the best model by spending it on about 20 tokens of text per parameter. Our model reads one character at a time, so our corpus is about 5.4 million tokens. Read naively, 5.4 million ÷ 20 suggests a model of roughly 270,000 parameters, forty times smaller than ours. (Chinchilla's tokens were word fragments of three or four characters. Counted that way, the gap is wider still.)

But that rule assumes you never show the model the same text twice, because its authors had more text than they could use. We have the opposite problem. We will pass over the same 5 MB many times.

Later research on exactly this situation (Muennighoff et al., 2023, "Scaling Data-Constrained Language Models") found that up to about 4 passes over the same data are almost as good as fresh data, that the benefit fades after that, and that by about 40 passes further repetition adds almost nothing. It also found that with a fixed dataset, the lowest loss came from models much larger than the 20-to-1 size. Two cautions. The same paper found that extra parameters lose their value faster than extra passes do, which argues against going very large. And even their smallest experiments used far more text than we have, so applying them to a 5 MB corpus is an extrapolation.

The practical evidence is closer to home. nanoGPT's Shakespeare demo uses exactly this shape on a 1 MB sample, and it works, with a catch: the model starts memorising. Its score on unseen text gets better for a while and then gets worse as it learns the training text by heart. That happens even though the demo already uses the usual defences:

- **dropout**: randomly switching off parts of the model during training, so it cannot lean on any one of them;
- **weight decay**: a steady pull of every parameter towards zero;
- **keeping the best checkpoint**: saving the model only when its score on held-out text (whole works we set aside and never train on) improves.

We will use the same three and expect to tune them. What we have that the demo does not is five times as much text. The same 5,000-step run that makes about 80 passes over nanoGPT's 1 MB sample makes about 17 passes over ours.

So the plan is to start at 10.8 million, then run the experiment properly: train a 4.8 million and a 37.9 million parameter model with the same tokenizer, the same context and the same data, and compare them on held-out text. We expect the big one to mostly memorise. We will find out.

## Where we are

In the repo at the end of this part:

```
gp-thee/
├── README.md, LICENSE               what this is, progress checklist; MIT
├── pyproject.toml, uv.lock          the environment, exactly
├── .python-version                  3.14
├── blog/                            this series
├── data/raw/100-0.txt               the entire universe, 5.4 MB
├── data/raw/README.md               checksum and the Project Gutenberg notice
├── docs/PLAN.md                     the roadmap
├── docs/BUILD_LOG.md                every step, command and number
├── scripts/download_data.py         verifies the corpus checksum
├── scripts/smoke_test_gpu.py        GPU vs CPU, with a control
├── scripts/explain_bit_identical.py why the matrix multiply matches exactly
├── scripts/count_params.py          the parameter arithmetic, checked
└── src/gp_thee/                     the package, still empty
```

No model yet. Next, in part 2: opening that 5.4 MB file and finding out how much of it is not Shakespeare.
