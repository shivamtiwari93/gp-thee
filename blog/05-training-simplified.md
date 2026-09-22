# Building GP-Thee, part 5: training, and what a number goes through before we believe it — simplified

*The learning loop is only a few lines. The hard work is deciding what to measure, how long to run, how to survive failures, and when a good-looking number deserves trust.*

> This is the simplified edition. For every run, failure mode, and audit result, read [the original Part 5](05-training.md).

## What you will learn

- What happens during one training step and why the learning rate changes over time.
- How bits per character makes different tokenizers comparable.
- Why a validation set, simple baselines, and best-checkpoint rules matter.
- How the team chose 34 passes rather than 17 or 68.
- Which operational bugs survived apparently successful training tests.
- What multiple seeds, repeated runs, and GPU nondeterminism say about the final score.

## The six lines that learn

Part 4 ended with an untrained model that had passed every check we knew to make. The core training loop is familiar PyTorch:

```python
tokens, targets = random_batch(training, 64, 256, batches, device)
loss = model(tokens, targets)[1]
optimizer.zero_grad()
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

One trip through these lines is a **step**.

First, the code samples 64 windows of 256 tokens from random locations. The targets are the same windows moved one token forward, creating 16,384 next-token predictions.

The model reports the average **loss**, or surprise, across those predictions. If the correct next token received probability `p`, its surprise is `-log(p)`. Lower is better. An untrained character model starts near 4.6 nats.

`loss.backward()` calculates one **gradient** for each of the model's 10,757,760 parameters: an estimate of how the loss would change if that parameter moved slightly. `zero_grad()` is essential because PyTorch otherwise adds new gradients to the old ones. Forgetting it does not necessarily crash; the model may continue learning badly, which makes the bug dangerous.

Gradient clipping treats all gradients as one long arrow. If its length exceeds 1.0, every component is scaled equally until the length is one. In each final baseline run, clipping activated on roughly one step in ten during the first 250 steps and on none of the next 9,700. A typical later gradient length was 0.26.

Finally, **AdamW** updates the parameters. It averages recent gradients and their squares, roughly remembering 10 and 100 steps. Dividing by typical gradient size gives parameters comparable update scales. The learning rate is the usual scale, not a hard maximum; consistent gradients move near the full rate, while gradients that change sign move less. AdamW also applies weight decay, gently shrinking matrices unless the loss justifies their size.

## The learning rate is a schedule

The learning rate is not fixed. It rises for the first 100 steps, then follows half a cosine curve from a peak of 0.001 to one tenth of that value by the end.

The opening rise is the **warm-up**. On AdamW's first step, the normalized update is close to plus or minus one for an ordinary nonzero gradient. Without warm-up, millions of parameters starting around 0.02 can each move by about 0.001 at once. Zero or tiny gradients move less, and 0.001 is not a mathematical cap. In the first benchmark without warm-up, loss fell to 3.4 and then jumped to 7.2 at step 5 even with clipping enabled.

Late in training, large steps cause the model to bounce around a good region instead of settling. The cosine decline reduces that movement gradually. Because the curve stretches to fill the planned run, changing the run length also changes the optimization path. A checkpoint halfway through a long run is not equivalent to the end of a shorter run: the long run still has a higher learning rate.

The recipe—batch size, 256-token windows, 20% dropout, peak and floor rates, 100-step warm-up, clipping, AdamW settings, and 0.1 weight decay—comes unchanged from nanoGPT's character-level Shakespeare configuration. It was tuned by someone else on a file about one fifth the size of ours. We did not claim it was optimal for GP-Thee.

## Try it: inspect the schedule and run a smoke test

The examples below assume you are in the repository root and have run `uv sync` plus the data and tokenizer commands from Parts 1–3.

You can inspect the real scheduling code without training anything:

```bash
uv run python - <<'PY'
from gp_thee.train import RunConfig, learning_rate

run = RunConfig(name="demo", passes=17)
steps = run.steps(4_811_375)
print("steps:", steps)
for step in (0, 49, 99, 100, steps):
    rate = learning_rate(step, run.warm_up, steps,
                         run.peak_rate, run.floor_rate)
    print(f"step {step:>4}: {rate:.6f}")
PY
```

Expected output:

```text
steps: 4992
step    0: 0.000010
step   49: 0.000500
step   99: 0.001000
step  100: 0.001000
step 4992: 0.000100
```

This shows both conversions discussed above: 17 passes become 4,992 updates, and the learning rate warms up before falling to its floor. A fast, cross-platform test of this logic and the overlapping evaluation windows is:

```bash
uv run pytest -q tests/test_train.py \
  -k 'learning_rate_climbs or budget_in_passes or every_token_is_scored_exactly_once'
```

The current suite reports `38 passed` for that selection.

To exercise the complete training/checkpoint pipeline without waiting for the 11-million-parameter model, train a deliberately tiny model on the CPU:

```bash
uv run python scripts/train.py \
  --name tutorial-smoke \
  --device cpu \
  --layers 2 --heads 2 --width 32 --context 32 \
  --batch 8 --passes 0.001 --warm-up 2 --evaluations 2
```

This builds a 28,896-parameter model, performs 19 updates, and normally finishes in seconds. You should see validation loss fall modestly; with seed 0 in our check, it moved from about 6.62 to 6.04 bits per character. The files under `runs/tutorial-smoke/` demonstrate the real output contract: configuration, log, samples, best and latest checkpoints, and the final result. This smoke test verifies the plumbing, not model quality. Pick a new `--name` if you run it again, or use `--resume` after an interruption.

When you are ready for the actual experiment, the character run is:

```bash
# Apple Silicon GPU: about 36 minutes on the machine used for this project
uv run python scripts/train.py --name my-char-run --passes 34 --seed 1

# Recompute its validation breakdown from the best checkpoint
uv run python scripts/evaluate.py --run my-char-run
```

The trainer defaults to Apple's `mps` device. On Linux, Windows, or an Intel Mac, add `--device cpu` to both commands; the full run will be much slower. A fresh clone contains the committed run logs and result summaries but not the large `.pt` checkpoints, so evaluation requires the `best.pt` produced by your own training run.

## Passes, not epochs

An **epoch** normally means reading every training example once in order. GP-Thee samples random windows, so it counts **passes** instead: one pass processes as many tokens as the training stream contains. With the character tokenizer, one pass is 294 steps.

A pass does not cover every token once. For a typical token away from the stream's ends, random sampling leaves about 37% unseen after one pass and visits about 26% more than once. The 37% is the familiar `1/e` result for random draws. After 17 passes, the chance that such an interior token was never read is about one in 24 million. Tokens near the two ends participate in fewer possible windows, a small edge effect.

This distinction matters when comparing tokenizers. A larger vocabulary creates a shorter token stream. A fixed step count would give it many more passes through the underlying text, while a fixed pass count gives it fewer parameter updates. Part 6 therefore lets each tokenizer earn its own training length under a rule written in advance.

## A fair score

Training loss is not the deciding score. It is measured on text the model is learning and with dropout deliberately enabled. Evaluation turns dropout off and uses two whole held-out plays: *All's Well That Ends Well* and *Romeo and Juliet*. The three test works remain locked.

Every validation token is scored once. Windows overlap so almost every prediction has at least 128 earlier tokens of context. Surprises are summed over all 274,728 tokens and divided once; the extra token beyond 274,727 characters is START between the plays.

The comparison unit is **bits per character**. Dividing natural-log loss by `ln(2)` converts nats to bits. One bit is the surprise of a fair coin flip. Lower is better.

Characters, words, and word fragments produce different numbers of predictions, so loss per token cannot compare their tokenizers. Total surprise for the same text, divided by the same number of characters, can. One caveat remains: a BPE model may assign probability to alternative fragmentations of the same text, while evaluation scores only the tokenizer's chosen fragmentation. That makes BPE look slightly worse by an amount we have not measured.

## Establishing a useful baseline

A score such as 1.75 means little without a reference. Before training, predictors with no neural network were fitted on the same 39 training works and scored on the same validation plays:

| Predictor | Bits per character |
|---|---:|
| Blind guess among 97 characters | 6.600 |
| Character frequencies | 4.783 |
| 3-gram: looks back 2 characters | 2.902 |
| 5-gram: looks back 4 characters | 2.258 |
| 6-gram: looks back 5 characters | **2.231** |
| 8-gram: looks back 7 characters | 2.408 |
| bzip2 after reading training works | 2.299 |
| xz after reading training works | 2.390 |

An **n-gram** counts what followed short sequences in the training text. Looking back five characters works best here. Looking back seven hurts because 19% of validation characters follow a seven-character sequence never seen in training, compared with 5% for a five-character sequence. The counts become too sparse.

Compression is prediction in another form: expected text takes fewer bits. The comparison is imperfect because bzip2 and xz keep learning on validation text, while the other models stay frozen; bzip2's block boundaries also move its score from 2.27 to 2.47. The practical bar is 2.23 bits per character.

## The first real run

The first character-token run trained for 17 passes: 4,992 steps. It took 20 minutes on the laptop GPU, including 21 pauses for evaluation and sampling.

Before training, validation scored 6.706 bits per character and generated noise. After 250 steps it scored 2.948 and had learned spaces, line lengths, and speaker-label shapes. At 1,000 steps it scored 2.120, passing the 6-gram baseline. At the end it reached 1.771:

| Step | Passes | Minutes | Seen training text | Validation |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 6.703 | 6.706 |
| 250 | 0.9 | 1 | 2.950 | 2.948 |
| 1,000 | 3.4 | 4 | 1.979 | 2.120 |
| 2,000 | 6.8 | 8 | 1.739 | 1.915 |
| 3,000 | 10.2 | 12 | 1.630 | 1.842 |
| 4,000 | 13.6 | 16 | 1.555 | 1.795 |
| 4,992 | 17.0 | 20 | 1.520 | **1.771** |

At 1.77 bits, the total surprise resembles repeatedly choosing among 3.4 equally likely characters. The average hides variation: 45% of characters cost under half a bit, while a word's first letter costs 3.8 bits on average.

The score is also uneven by content:

| Validation slice | Characters | Bits per character |
|---|---:|---:|
| Both plays | 274,727 | 1.771 |
| *All's Well That Ends Well* | 133,386 | 1.715 |
| *Romeo and Juliet* | 141,341 | 1.823 |
| Speaker-label lines | 16,346 | 2.179 |
| Everything else | 258,381 | 1.745 |

Unknown names drive much of the difficulty. Familiar-name labels cost 0.92 bits per character; 858 labels with unseen names cost 3.45. Longer training made unseen-name letters worse while ordinary text improved, because the model became more certain about known casts. Later comparisons report labels separately, though the predeclared overall score still decides.

## Choosing how long to train

The 17-pass run was still improving at its final checkpoint. We doubled its length, then doubled again:

| Run length | Steps | Best validation score | Best moment |
|---|---:|---:|---|
| 17 passes | 4,992 | 1.7706 | final step |
| 34 passes | 9,985 | 1.7404 | step 9,750 |
| 68 passes | 19,969 | 1.7577 | step 10,750 |

The 34-pass run gained 0.030 bits. The 68-pass run did not improve on it and degraded to 1.7937 by the end. Its seen score continued falling, which is a classic sign of **over-fitting**: it was learning details of the 39 training works that did not help on new plays.

Best-checkpoint selection protects against training too long because it preserves the point before validation worsens. It cannot protect against stopping too soon. A run whose best checkpoint is its final one may simply have more improvement left.

The project therefore adopted a rule proposed during the audit:

> Start near 5,000 steps. Keep doubling while the best validation score improves by more than 0.01 bits per character, up to 20,000 steps. If the best point arrives before two thirds of a run, also try half that length. Keep the shortest length within 0.01 of the best.

That selects 34 passes. The threshold was written after the first two runs but before the third, an honest limitation; the 0.030 improvement was not close to 0.01. Repeated runs narrowed the original gaps but kept the decision: all nine 34-pass runs beat the 17-pass pilot, and 68 remained no better than 34.

## What the seen-versus-unseen gap does not prove

At the 34-pass run's best point, the seen score was about 1.37 and validation about 1.74. The first code comment called that gap memorization. Two auditors showed this was unjustified.

A 5-gram cannot recite anything longer than five characters, yet it scores 1.912 on works used to build its counts and 2.258 on validation. When each training work is scored after being left out of the counts, it scores 2.301. Different plays have different names, places, jokes, and phrasing, so a model can be better on familiar works without reproducing them word for word.

The gap is still useful *within a run*: if seen performance improves while validation stalls or worsens, the model is fitting training-specific detail. Actual recitation requires a separate measurement, which comes later in the series.

## The arithmetic was right; the training system was not

An independent evaluator exactly reproduced the untrained model's full validation score. The core learning numbers were sound. Much of the surrounding system was not.

The first training implementation had serious resume and checkpoint failures. A resumed run could silently use command-line defaults rather than its saved tokenizer and settings. A smaller resumed budget could loop forever. A finished run crashed when resumed. The elapsed-time clock restarted. Checkpoints were written in place, so interruption during a save could destroy them. Starting in an existing run directory could overwrite the best model. Evaluating every 250 steps gave different tokenizers different numbers of chances to catch a lucky “best” moment.

Mutation testing then planted 64 faults. Thirty passed all 45 tests, including training on validation text, overwriting `best.pt` at every stop, applying the learning rate one step late, decaying normalization scales, disabling clipping or dropout, and ignoring the requested seed.

The tests had followed only the expected path with defaults. A tiny run improved at every stop, so “best” and “latest” were indistinguishable. No test examined what the optimizer actually did.

The strongest repair was a test that writes the learning loop independently, changes nearly every setting from its default, runs 16 steps, and requires bit-for-bit identical weights. Another creates deliberate over-fitting, interrupts and resumes the run, and verifies that the true best checkpoint survives.

After the rewrite, fresh auditors confirmed the original faults were fixed, including a real forced termination. Then 21 of 28 faults planted in the *new safety code* survived. In a wider 167-fault campaign, remaining weaknesses clustered in saving, resuming, and run provenance rather than score arithmetic. Tests now interrupt at five points inside a stop and demand the same outputs as an uninterrupted run. The training suite grew to 122 tests and the whole project to 262.

Two small incidents summarize the operational lesson. A run name of `..` passed the original folder-name check and could have written checkpoints into the repository root. Later, a shell loop accidentally put `--seed 1` inside a folder name, so the run used the default seed. Both names are rejected now. Reliable experiments require boring defenses around filenames, state, and recovery—not only correct gradients.

## Rules fixed before comparing models

The tokenizer comparison uses seeds 1, 2, and 3; seed 0 is reserved for pilot runs. Every run gets 40 evenly spaced evaluation stops. Other settings remain frozen, including the character-oriented learning rate and the 256-*token* window. Those choices may favor characters and are reported as limitations.

Three seeds provide a noisy spread estimate, so variation is pooled across tokenizer arms. Means count as different only beyond 1.82 pooled standard deviations, and the smallest vocabulary within that distance of the lowest mean wins. Simulations say this procedure overlooks the character arm about 7% of the time when all five arms are truly equal; small differences will usually become ties.

One seed is run twice to measure hardware drift, and once in 16-bit. The comparison may use 16-bit only if that run lands within a threshold set from same-seed drift and cross-seed spread. This rule was written before the result.

## The final character baseline

The real baseline used 34 passes, 40 stops, 32-bit arithmetic, and the rewritten training system:

| Run | Best validation | Best step | End score | Minutes |
|---|---:|---:|---:|---:|
| seed 1 | 1.7490 | 7,239 | 1.7551 | 37.7 |
| seed 1 again | 1.7517 | 8,986 | 1.7614 | 35.8 |
| seed 2 | 1.7597 | 8,986 | 1.7663 | 35.8 |
| seed 3 | 1.7504 | 9,735 | 1.7584 | 36.0 |
| seed 1, 16-bit | 1.7592 | 8,238 | 1.7685 | 21.7 |

The baseline is **1.7535 bits per character**, with standard deviation **0.0054** across seeds after averaging the two seed-1 runs. The same-seed twins differ by 0.0027 and peak at different stops. Across nine investigated 32-bit runs, scores range from 1.7404 to 1.7597, with mean 1.7505 and standard deviation 0.0062.

Changing the program without changing its mathematics moved GPU scores by similar amounts. On the CPU, old and rewritten loops produce bit-identical weights. GPU scheduling may alter the order of repeated-token gradient additions, but that mechanism is an inference. The measured lesson is firm: one run, or even one implementation, is not stable evidence.

Most headline variation comes from speaker labels. Across seeds, ordinary text has standard deviation 0.0013 bits, while labels have standard deviation 0.09. Seed 2 looks worst almost entirely because it is worse at guessing names absent from training.

The 16-bit run missed the predeclared acceptance threshold: it was 0.0088 from the twins' mean, versus a 0.0054 threshold. That does not prove 16-bit learns worse—the result lies inside the 32-bit range—but the rule says Part 6 must use 32-bit. A 32-bit run sustains about 85,400 tokens per second and takes roughly 36 minutes; 16-bit sustains 153,800 and took 21.7 minutes.

Finally, the promised causal check was repeated on trained models. Across all 255 cut points, two attention implementations, two numerical precisions, and gradients both on and off, the largest change to any past prediction was exactly 0.0.

## Takeaways

- The optimizer loop is short; trustworthy evaluation, checkpointing, recovery, and experiment rules are the larger engineering job.
- Bits per character makes tokenizers comparable, and simple baselines turn an abstract score into a meaningful result.
- Best-checkpoint selection limits the cost of running too long but cannot recover learning lost by stopping too early.
- A seen-versus-validation gap is evidence of domain familiarity and possible over-fitting, not proof of verbatim memorization.
- Successful training does not validate the training system. Mutation tests exposed silent failures in seeds, settings, data selection, dropout, clipping, and checkpoints.
- The final character baseline is 1.7535 bits per character with 0.0054 cross-seed standard deviation, comfortably ahead of the 2.231 6-gram baseline.
- GPU nondeterminism and program-level changes move results enough that conclusions need repeated runs and rules written before the outcomes are visible.

[← Part 4: the model — simplified](04-the-model-simplified.md) · [Next: Part 6, choosing a tokenizer — simplified →](06-which-tokenizer-simplified.md)
