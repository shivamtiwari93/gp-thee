# Building GP-Thee, part 1: prerequisites and setup

*What we started with, what we installed and why, and how we know the model will have 10 million parameters before we have trained anything.*

## The premise

Suppose the only text that ever existed was the complete works of Shakespeare. No internet, no Wikipedia, no other books. Could you build a language model in that universe?

That is GP-Thee: a GPT trained from scratch on 38 plays, 154 sonnets and 5 poems, and nothing else. The rule is strict. No pretrained weights. No borrowed tokenizer, because a tokenizer learned from the web is knowledge smuggled in from outside the universe. No extra text.

The rule is also what makes this a good project to learn from. The whole corpus is 5.4 MB, so every experiment runs in minutes on a laptop, and the full pipeline is small enough to read end to end: data, tokenizer, model, training, evaluation, sampling.

This series documents every step, including the mistakes. It was built with Claude Code as a pair programmer. Everything is in the repo: [github.com/shivamtiwari93/gp-thee](https://github.com/shivamtiwari93/gp-thee).

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

This is far more computer than the project needs. As you will see below, the model and everything needed to train it fit in under 1 GB of memory. The thing that limits this project is not the hardware. It is the size of the corpus: about 963,000 words is a tiny amount of text by language-model standards, and most of the design decisions in this series are about coping with that.

You do not need this machine to follow along. Any Apple Silicon Mac should work with the same commands. On a PC with an NVIDIA GPU, or with no GPU at all, the code will need one word changed (the device name), and training will take longer on a CPU. We have not tested those setups.

### What was already installed

| Tool | Version | What it is for |
|---|---|---|
| Homebrew | 7.0.4 | installs command-line tools on a Mac |
| Python | 3.14.6, from Homebrew | not used directly; see below |
| Xcode | 26.6 | Apple's developer tools, which include the compilers and `git` |
| git | 2.50.1 | version control |
| GitHub CLI (`gh`) | 2.96.0 | creates the repo and pushes from the terminal |

### What was missing

Everything specific to machine learning. No PyTorch, no NumPy, no tokenizer libraries, and no tool for managing Python environments. That is a clean start, which is what you want for a project that claims to be reproducible.

## The stack we chose

| Layer | Choice | Why |
|---|---|---|
| Environment manager | `uv` 0.12.17 | one tool for the Python version, the virtual environment and a lockfile |
| Python | CPython 3.14.7, installed and managed by `uv` | isolated from the system, so a Homebrew upgrade cannot break the project |
| Deep learning framework | PyTorch 2.14.0 | the clearest reference material for this exact task, easy to debug, and it runs on the Mac's GPU |
| GPU backend | MPS (Metal Performance Shaders) | PyTorch's way of talking to an Apple GPU; you select it with `device="mps"` |
| Arrays | NumPy 2.5.3 | stores the tokenized corpus on disk as a flat array of integers |

### Why `uv`, and why not the Python that was already there

Homebrew's Python refuses `pip install` by design. It is marked "externally managed", so that you do not break tools that depend on it. A virtual environment is mandatory, not optional.

We went one step further and told `uv` to use its own copy of Python rather than Homebrew's:

```toml
[tool.uv]
python-preference = "only-managed"
```

The reason is timing. Homebrew will move to Python 3.15 at some point, and a virtual environment built on top of Homebrew's interpreter breaks when that interpreter is replaced. With a managed interpreter and a lockfile (`uv.lock`), anyone who clones the repo and runs `uv sync` gets the same Python and the same package versions we used.

### Why PyTorch, and why not Apple's MLX

Apple has its own framework, MLX, built for these chips. It was the runner-up.

At this scale, speed does not decide it. Both frameworks train a 10-million-parameter model in minutes. What decided it:

- **Reference material.** Andrej Karpathy's nanoGPT is a line-by-line PyTorch implementation of almost exactly this model, trained on a Shakespeare sample. When our numbers look wrong, there is something to compare against.
- **Debugging.** PyTorch runs each operation immediately, so you can print any intermediate value. That matters when you are learning.
- **Transfer.** The same code runs on NVIDIA GPUs by changing the device name.

Porting the model to MLX and timing both is a planned later experiment. We could not find a published training benchmark for the M5 Max in either framework, so that comparison would be new information.

### Why the version is pinned to exactly 2.14.0

```toml
dependencies = [
    "numpy>=2.5.3",
    "torch==2.14.0",
]
```

During research we found two bugs in PyTorch's GPU backend that affect M5 chips specifically: half-precision matrix multiplication that gave slightly different answers from run to run (worked around in 2.13), and corrupted attention output on macOS 26 (fixed in 2.14). An accidental upgrade or downgrade halfway through the project could change our results without any change to our code. So the version is pinned, and we will not update macOS or PyTorch until the project is finished.

### What we deliberately did not install

| Not installed | Why |
|---|---|
| `tiktoken`, `transformers`, pretrained tokenizers | They carry vocabularies learned from internet text. That breaks the rule. We will write our own tokenizer and train it on Shakespeare only. |
| MLX | Later, as a comparison. |
| `matplotlib` | Not needed until there is something to plot. It gets added when we train. |
| Jupyter | Every step is a plain script, so every result can be regenerated with one command. |

### What came along for the ride

We asked for two packages and got eleven. PyTorch brings nine dependencies with it:

| Package | Why PyTorch wants it |
|---|---|
| `sympy`, and its dependency `mpmath` | symbolic maths, used by PyTorch's compiler to reason about tensor shapes |
| `networkx` | graph algorithms, used when the compiler rearranges a model's operations |
| `jinja2`, and its dependency `markupsafe` | text templates, used by the compiler to generate GPU code |
| `filelock` | stops two processes writing to the same cache file |
| `fsspec` | reading and writing checkpoints on different kinds of storage |
| `setuptools`, `typing-extensions` | Python packaging and type-hint support |

Most of these serve `torch.compile`, which we will not use: on Apple GPUs it is still labelled an early prototype. They are installed but idle.

The whole environment is 617 MB on disk, of which PyTorch itself is 544 MB. Compare that with the corpus at 5.4 MB and the model we are about to build at 43 MB. The tools are more than ten times the size of the corpus and the model put together.

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

Before building anything on top of the GPU, we checked that it computes correctly. The test multiplies two 2048×2048 matrices on the CPU and on the GPU and compares the answers.

The first version of the test printed a difference of exactly `0.0`.

That looks like good news, and it should make you suspicious. Floating-point arithmetic on two different devices almost never agrees to the last bit, because the additions happen in a different order. An exact zero is what you see when a test is broken, for example when it accidentally compares a tensor with itself.

So we tested the test. We compared both results against a higher-precision reference, and ran a second group of operations that uses different GPU code.

| Check | Result |
|---|---|
| GPU available to PyTorch | yes |
| Matrix multiply, GPU vs CPU | bit-identical |
| The same multiply vs a 64-bit reference | largest error 0.00054 on values averaging 36, which is normal for 32-bit floats |
| LayerNorm, then Linear, then GELU: GPU vs CPU | largest difference 0.0000019, which is normal |
| 50 of those matrix multiplies on the GPU | 69 ms |

The multiply really is bit-identical on this machine, and the second path shows the tiny differences you would expect. The test was fine. It took two minutes to find that out, and it was worth it: a test that cannot fail tells you nothing.

## The dataset, briefly

The corpus is Project Gutenberg's eBook #100, *The Complete Works of William Shakespeare*: one plain-text file of 5,422,721 bytes. We downloaded it once, verified its SHA-256 checksum, and stored it read-only. Part 2 covers why we chose it over six alternatives and how we clean it.

One decision is worth telling here because we got it wrong first. The initial plan kept the raw file out of the repository, because it carries Project Gutenberg's name and that name is a trademark. That was too cautious. Project Gutenberg's licence allows free redistribution as long as a specific notice travels with the file. More importantly, an educational project that hides its dataset is missing the point, and Project Gutenberg revises its files from time to time, so a download script with a pinned checksum will eventually stop working. The file is now [in the repo](../data/raw/), unmodified, with the required notice beside it.

## How we know it is 10 million parameters before training

This question came up when naming the project. If the model is going to be called GP-Thee-10M, how can we know the "10M" before we have trained anything?

Because the number of parameters is a property of the architecture, not of the training. Training changes the *values* of the parameters. It never changes how many there are. You choose the shape of the model, and the count follows by arithmetic.

### What a parameter is

A parameter is one adjustable number inside the model. They are stored in matrices. A layer that turns a list of 384 numbers into another list of 384 numbers is a 384×384 matrix, which is 147,456 parameters. Counting a model's parameters means listing its matrices and adding up their sizes.

### The shape we chose

Four numbers define a GPT of this kind:

| Symbol | Meaning | Our value |
|---|---|---|
| `d` | width: how many numbers represent each token | 384 |
| `L` | layers: how many transformer blocks are stacked | 6 |
| `V` | vocabulary: how many distinct tokens exist | 100 (the distinct characters in the corpus) |
| `T` | context: how many tokens the model can look back over | 256 |

The model also has 6 attention heads, but heads do not change the count. They split the same matrices into 6 parts.

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

**Attention, `4d²` per block.** Each token is turned into a query, a key and a value, which is three `d×d` matrices. A fourth `d×d` matrix mixes the result.

**Feed-forward, `8d²` per block.** One matrix expands each token from `d` to `4d` numbers, and another brings it back. That is `d×4d + 4d×d`.

So each block has `12d²` parameters in its matrices, plus `2d` for its two layer norms.

**The output layer costs nothing.** To predict the next character, the model needs a `d×V` matrix, the same shape as the token embedding turned on its side. A standard trick called weight tying uses the same matrix for both jobs. It saves parameters and usually helps small models.

We use no bias terms, which is a common modern simplification and keeps the formula exact:

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

Notice where the weight is. 98.7% of the parameters are in the blocks. The vocabulary barely registers. That will change when we try a larger tokenizer: a 2048-entry vocabulary adds `2048 × 384 = 786,432` parameters, which takes the total to about 11.5 million.

That is why the repository is called `gp-thee` and each released model will be named by its real count, for example `GP-Thee-10M`.

### Checking the arithmetic against PyTorch

A formula is a claim, so we tested it. [scripts/count_params.py](../scripts/count_params.py) builds the model's skeleton in PyTorch, with the real matrix shapes but no training, and asks PyTorch to count:

```bash
uv run python scripts/count_params.py
```

| Configuration | PyTorch counts | Formula gives |
|---|---|---|
| 6 layers, width 384, 100 characters | 10,758,528 | 10,758,528 |
| the same with a 2048-entry vocabulary | 11,506,560 | 11,506,560 |
| smaller: 6 layers, width 256 | 4,813,056 | 4,813,056 |
| larger: 12 layers, width 512, vocabulary 2048, context 512 | 39,072,256 | 39,072,256 |

They agree to the last parameter. If you know nanoGPT, you may remember 10.65 million for this shape. That is the same model with a 65-character vocabulary, and nanoGPT leaves the position embedding out of its headline number: `10,616,832 + 4,992 + 65 × 384 = 10,646,784`.

### What 10 million parameters costs

Each parameter is a 32-bit float, which is 4 bytes.

- **The model:** `10,758,528 × 4 bytes ≈ 43 MB`.
- **Training it:** the optimizer we will use (AdamW) keeps three more numbers for every parameter: its gradient and two running averages. That is four copies, about 172 MB, plus working memory for the batch being processed.

On a machine with 128 GB, memory will never be the problem.

### Why 10 million, and not 100 thousand or 100 million

This is the real design question, and we should be honest that our answer is a hypothesis to test, not a fact.

A well-known result from DeepMind (the "Chinchilla" paper) says a model trained efficiently should see about 20 tokens of text per parameter. Our corpus is about 5.4 million characters. Read naively, that suggests a model of roughly 270,000 parameters, forty times smaller than ours.

But that rule assumes you never show the model the same text twice, because its authors had more text than they could use. We have the opposite problem. We will pass over the same 5 MB many times. Later research on training with repeated data (Muennighoff et al., 2023) found that repeating data a few times is almost as good as new data, with returns shrinking after that, and that with repetition it pays to use a larger model than the 20-to-1 rule suggests. Those experiments were run at a much larger scale than ours, so applying them to a 5 MB corpus is an extrapolation.

The practical evidence is closer to home. nanoGPT's Shakespeare demo uses exactly this shape on a 1 MB sample, and it works, with a catch: the model starts memorising. Its score on unseen text gets better for a while and then gets worse as it learns the training text by heart. We have five times as much text, which should help, and we will add the usual defences: dropout, weight decay, and stopping training when the score on held-out text stops improving.

So the plan is to start at 10.8 million, then run the experiment properly: train a 4.8 million and a 39 million parameter model on the same data and compare. We expect the big one to mostly memorise. We will find out.

## Where we are

In the repo at the end of this part:

```
gp-thee/
├── README.md                 what this is, progress checklist
├── pyproject.toml, uv.lock   the environment, exactly
├── data/raw/100-0.txt        the entire universe, 5.4 MB
├── docs/PLAN.md              the roadmap
├── docs/BUILD_LOG.md         every step, command and number
├── scripts/download_data.py  verifies the corpus checksum
└── scripts/count_params.py   the parameter arithmetic, checked
```

No model yet. Next, in part 2: opening that 5.4 MB file and finding out how much of it is not Shakespeare.
