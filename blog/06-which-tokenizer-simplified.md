# Building GP-Thee, part 6: choosing the tokenizer — simplified

*We trained five versions of the same model and decided how it should read Shakespeare. The important part was not the race. It was writing the rules before seeing the winner.*

> This is the simplified edition. The [full technical version](06-which-tokenizer.md) contains the complete pilot runs, statistical tests, audits, and exploratory analysis.

## What you will learn

- Why comparing tokenizers is harder than changing one configuration value.
- How we designed a fair-enough experiment before running it.
- Why the character tokenizer won this project, even though modern large models usually use word fragments.
- How to report a result without pretending it applies everywhere.

## The decision

A **tokenizer** turns text into the units a model reads. Part 3 built five choices:

- one token per character;
- word fragments learned with byte-pair encoding, or **BPE**, using vocabularies of 1,024, 1,536, 2,048, or 4,096 pieces.

The neural network stayed almost the same. What changed was the way text reached it.

That difference matters. The character model needs one token for every character. A BPE model can represent a common fragment such as ` the` with one token. As a result, a 256-token window covers 256 characters for the character model but roughly 640 to 790 characters for the BPE models.

| Tokenizer | Training tokens | Characters per token | Text inside 256 tokens |
|---|---:|---:|---:|
| characters | 4,811,375 | 1.00 | 256 characters |
| BPE 1,024 | 1,936,643 | 2.49 | about 640 characters |
| BPE 1,536 | 1,787,863 | 2.66 | about 680 characters |
| BPE 2,048 | 1,694,913 | 2.78 | about 710 characters |
| BPE 4,096 | 1,507,176 | 3.07 | about 790 characters |

BPE therefore offers a real product benefit: more text fits in the same context window, and the model needs fewer tokens to read or write the same passage.

But it also makes the experiment difficult to compare fairly.

## Why the same number of steps or passes is not a fair rule

Suppose every model trains for the same number of update steps. The BPE models move through the corpus much more often because their version of the corpus contains fewer tokens.

Now suppose every model trains for the same number of passes through the text. The character model receives roughly three times as many update steps.

Neither comparison is neutral.

Training length also changes the learning-rate schedule. A longer run keeps a high learning rate for longer. Part 5 showed that an overly long run can continue improving on training text while getting worse on unseen text. Keeping the “best checkpoint” helps, but it does not completely repair a badly chosen schedule.

So each tokenizer **arm**—one tokenizer configuration and its runs—was assigned a run length by a rule written in advance:

1. Start near 5,000 steps.
2. Try twice as many steps.
3. If the best moment came early, also try half as many.
4. Keep the shortest run whose validation score is within 0.01 bits per character of the best pilot.

This selected 9,985 steps for characters and 2,500 for every BPE arm. Characters, BPE 1,024, and BPE 4,096 had direct length searches. Under the prewritten rule, the two middle vocabularies borrowed the shared 2,500-step result from the outer BPE pilots rather than receiving independent searches. All BPE lengths ultimately rested on single pilot runs, so they could be a little lucky or unlucky. We disclosed that before the main comparison.

## The rule for choosing a winner

The final race used fifteen runs: five tokenizer arms, each trained with seeds 1, 2, and 3. A **seed** fixes the repeatable starting choices, such as initial weights and batch order; a **checkpoint** is a saved model state from partway through a run.

The decision rule was fixed before those runs:

- Compare the best validation checkpoint from each run.
- Use **bits per character**, where lower is better, so every tokenizer is measured with the same unit.
- Average the three seeds for each tokenizer.
- Estimate normal run-to-run variation across the experiment.
- Treat two tokenizers as meaningfully different only when their means are more than 1.82 times the pooled run-to-run standard deviation apart.
- If several tokenizers are inside that margin, choose the smaller vocabulary.

That last rule is an explicit product preference: complexity has to earn its place.

The multiplier was fixed in advance. After the runs, the pooled standard deviation was 0.0071 bits per character, so the rule's observed margin became about 0.0130.

Before running the race, simulations showed what the experiment could detect. A real improvement of 0.02 bits per character would almost always be noticed. An improvement of 0.01 would often be missed. Therefore “characters win” could only mean “no BPE model beat characters by an amount this experiment can reliably see.”

## Try it: replay the decision without retraining

You do not need five hours of GPU time to inspect how the decision was made. The repository includes the run metadata and measured results, but not the large training checkpoints.

First, replay the prewritten length rule in read-only mode:

```bash
uv run python scripts/find_length.py --tokenizer bpe-1024 --dry
```

It reads the completed pilot metadata, runs no training, writes nothing, and should finish with:

```text
the rule picks 2,500 steps = 21.2 passes
```

Next, list the fifteen runs demanded by the comparison plan:

```bash
uv run python scripts/sweep.py --dry
```

The output should show three seeds for each of `char`, `bpe-1024`, `bpe-1536`, `bpe-2048`, and `bpe-4096`. In this checkout, every line ends in `done`. The `--dry` flag is important: without it, the script starts or resumes missing runs and can occupy an Apple GPU for hours.

Finally, read the committed result artifact with a short Python program:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("docs/results-sweep.json").read_text())
for group in report["groups"]:
    name = group["settings"]["tokenizer"]
    mean = group["mean_best_validation_bpc"]
    print(f"{name:>8}: {mean:.4f} bits/character ({group['seeds']} seeds)")
print("choice:", report["choice"]["settings"]["tokenizer"])
PY
```

This prints the five means in the table below and ends with `choice: char`. It is an audit of the published evidence, not an independent reproduction of training. To reproduce the expensive part on Apple Silicon, remove `--dry` from `scripts/sweep.py`; expect roughly five hours on the reference machine, keep the laptop plugged in, and do not change the code while it is running. On other platforms, the official sweep script still defaults to `mps`; running the fifteen experiments on CPU would require changing that workflow and would no longer be an exact replay.

## The result

| Tokenizer | Steps | Seed 1 | Seed 2 | Seed 3 | **Mean** |
|---|---:|---:|---:|---:|---:|
| characters | 9,985 | 1.7492 | 1.7571 | 1.7575 | **1.7546** |
| BPE 1,024 | 2,500 | 1.7895 | 1.8006 | 1.8136 | **1.8012** |
| BPE 1,536 | 2,500 | 1.8313 | 1.8219 | 1.8186 | **1.8240** |
| BPE 2,048 | 2,500 | 1.8227 | 1.8291 | 1.8198 | **1.8239** |
| BPE 4,096 | 2,500 | 1.8460 | 1.8539 | 1.8533 | **1.8511** |

**The character tokenizer won.**

The nearest BPE model was 0.0466 bits per character behind. With the observed variation, the experiment needed a gap of about 0.013 to call two arms different, so this was not a close decision.

We also gave BPE the benefit of the known caveats:

- The 1,024-piece arm may have been trained a little too briefly. Single pilots suggested a possible penalty near 0.008, but the true figure could plausibly be anywhere from zero to about 0.016.
- BPE text can sometimes be spelled with more than one token sequence. Our scoring method counts only the spelling produced by the tokenizer. One pilot estimated this overcharge near 0.008, with a range of roughly 0.004 to 0.013; it was not remeasured on the final runs.

Even returning both amounts leaves the character model about 0.03 bits per character ahead.

That is strong evidence for this project. It is not a universal law.

## Where the difference appeared

The result becomes more useful when split by type of text.

| Validation slice | characters | BPE 1,024 | BPE 4,096 |
|---|---:|---:|---:|
| all text | **1.7546** | 1.8012 | 1.8511 |
| ordinary text | **1.7117** | 1.7745 | 1.7799 |
| speaker-label lines | 2.434 | **2.225** | 2.976 |

Characters were clearly better on ordinary text. The smallest BPE vocabulary was better on speaker labels, while the largest vocabulary was much worse there.

Names are difficult because many names in the validation plays never appear in training. Larger BPE vocabularies contain more whole names from the training works, but that did not translate into better handling of new names. This was a predeclared secondary slice, but its label scores were the noisiest and the arms trained for different lengths. The apparent BPE 1,024 advantage should therefore be read descriptively, not as a causal finding.

An exploratory word-level comparison found that characters were ahead on common words, medium-frequency words, and rare words seen during training. BPE 1,024 was roughly level on words never seen before. The largest share of the overall gap came from common words simply because common words occupy most of the text.

## Why did characters win?

The honest answer is: **we do not know**.

Several explanations are plausible:

- BPE models received fewer update steps before they began overfitting.
- The learning rate, dropout, batch size, and other settings came from a recipe designed for a character model.
- Settings measured in steps affected the arms differently. A 100-step warm-up is 4% of a 2,500-step BPE run but only 1% of the character run.
- The scoring method slightly disadvantages BPE.

One tempting explanation does not fit the results well: “large vocabularies have too many rare pieces.” If that were the main problem, BPE 4,096 should have been dramatically worse than BPE 1,024 on ordinary text. It was only about 0.005 worse there—too small for this experiment to separate confidently.

The result may also change with more data. Large production models use word fragments because they reduce the number of tokens needed for the same text, saving compute and expanding effective context. Those advantages matter enormously at scale. Our project had only about five megabytes of training text, and its character model took about 35 minutes per run versus roughly 10 minutes for a BPE run.

## What the experiment does—and does not—prove

It supports this statement:

> For this roughly 11-to-12-million-parameter model family trained on this Shakespeare corpus with this unchanged recipe, none of the four tested BPE vocabularies matched the character tokenizer.

It does not prove that:

- characters are the best tokenizer in general;
- the BPE models were trained as well as they could be;
- a tuned BPE recipe would lose;
- the result applies to larger models or larger datasets;
- the same winner will appear on the unopened test works.

That scope is not defensive language. It is the boundary of the evidence.

## Lessons for product and engineering teams

**Define the decision before the dashboard appears.** If the rule is written after results arrive, normal noise can be narrated as success.

**Compare systems in the unit users care about.** Tokens are implementation details here; characters are the shared output. That is why bits per character is the fair unit.

**Price complexity explicitly.** “Choose the smallest option inside the uncertainty margin” is a practical way to stop marginal gains from buying permanent operational cost.

**Separate a result from its explanation.** The winner is measured. The reason it won is still a hypothesis.

## Takeaways

- Tokenization changes context length, compute cost, training dynamics, and model size at the same time.
- Fair experiments often need arm-specific budgets rather than one identical budget.
- Three runs per arm were enough to make this decision because the observed gap was much larger than the noise.
- Characters won this project by a clear margin, under an untuned character-model recipe.
- The correct public claim is narrow, reproducible, and useful—not universal.

[← Part 5: training — simplified](05-training-simplified.md) · [Next: Part 7, does it recite? — simplified →](07-does-it-recite-simplified.md)
