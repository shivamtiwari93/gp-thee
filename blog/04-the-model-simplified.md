# Building GP-Thee, part 4: the model, and how we tried to make it cheat — simplified

*A language model can be wrong and still train, produce plausible text, and report a falling loss. This part builds GP-Thee's 11-million-parameter model and, more importantly, builds evidence that it is the model we intended.*

> This is the simplified edition. For the implementation details, mathematical checks, and complete audit, read [the original Part 4](04-the-model.md).

## What you will learn

- How a small GPT-style model turns token numbers into next-token predictions.
- Why the causal mask is the most important rule in language-model training.
- Which checks catch obvious failures—and which reassuring checks can all miss the same bug.
- What independent implementations and deliberately planted faults revealed.
- Why we chose hand-written attention and 32-bit arithmetic for the first real runs.

## The model at a glance

Part 3 turned text into token numbers. The model reads up to 256 of those numbers and produces, at every position, one score for every vocabulary piece. Higher scores mean “more likely to come next.” A softmax operation converts the scores into probabilities that add up to one.

The model is a small version of the GPT-2 architecture:

```
token numbers
   │
   ├─ look up 384 numbers for each token
   └─ add 384 numbers representing its position
   ▼
six repeated blocks
   ├─ attention: gather information from this and earlier positions
   └─ feed-forward: process that information at each position
   ▼
final normalization
   ▼
scores for the next token
```

Each token is represented by a row of 384 learned numbers. The same table is reused at the output: the model compares its final representation with every vocabulary row to produce scores. This **weight tying** saves parameters and gives the input and output a shared language.

With the 98-piece character vocabulary, the model has exactly 10,757,760 parameters. The code calculates that number when the model is created and refuses to continue if it disagrees with the independently derived formula from Part 1.

A whole 256-token window is useful at once. The output at position 1 predicts token 2; the output at position 5 predicts token 6; and so on. One forward pass therefore creates 256 training exercises per window rather than only one.

That efficiency creates the central risk: the correct answer is present one position to the right in the same input.

## Attention, in plain language

Attention lets each position choose information from other positions.

Every position produces three small vectors:

- A **query** describes what this position is looking for.
- A **key** describes what this position offers.
- A **value** is the information it will contribute if selected.

The model compares a query with every allowed key, turns those matches into shares that total one, and mixes the values using those shares. GP-Thee uses six attention heads in parallel. Each head works with 64 of the 384 numbers, so a position can gather six different kinds of information at once.

The position row added at the input gives each place an explicit address. Without that signal, attention over already-made keys and values does not care about their order. Position rows are not quite the only source of order: the causal rule means earlier positions were built from shorter prefixes, and deeper layers can recover some order from that pattern. The position rows make order direct instead of forcing the model to reconstruct it.

During training, dropout randomly removes 20% of attention shares and scales the rest so the average stays unchanged. Similar dropout appears elsewhere in the model. It is a training handicap intended to reduce dependence on any one path; it is switched off during evaluation and generation.

## The rule that prevents cheating

When the model predicts token 6 from position 5, position 5 must not see position 6. Otherwise training becomes a copying exercise. Loss would collapse, the charts would look excellent, and generation would fail because there is no future token to copy.

The **causal mask** states which positions may influence which:

```
position 1 may see: 1
position 2 may see: 1, 2
position 3 may see: 1, 2, 3
position 4 may see: 1, 2, 3, 4
```

Forbidden attention matches are set to negative infinity before softmax. Their resulting share is exactly zero. Setting them to ordinary zero would be wrong, because exponentiating zero produces one—an entirely normal share.

This rule must hold through every route in the network, not only in the visible attention matrix. A normalization that accidentally averaged across positions or a batching error that mixed rows could also leak future information.

## What happens inside each block

After attention, a **feed-forward network** expands each 384-number row to 1,536 numbers and reduces it back to 384. A curved activation called GELU sits between the two matrix operations. Without that curve, two matrices in sequence would behave like a single matrix and add little expressive power.

Each attention and feed-forward stage uses a **residual connection**: its result is added to the input rather than replacing it. That keeps the original information available and gives gradients a short route back through the network. Layer normalization before each stage keeps the numerical scale manageable.

These ingredients are standard. The difficulty is not inventing them; it is implementing all of them without a quiet shape, indexing, dropout, or masking error.

## Try it: verify the model before training

Run these commands from the repository root after completing the setup and tokenizer steps in Parts 1–3. The first two work on macOS, Linux, and Windows and do not train a model or need a saved checkpoint.

First, ask PyTorch and the hand calculation to count the same parameters:

```bash
uv run python scripts/count_params.py
```

The first line should end like this:

```text
current char model: 6L/384d, V=98   pytorch=10,757,760  hand=10,757,760  match=True
```

The spacing may differ, but the two counts and `match=True` are the important evidence. Then run the focused CPU tests for the model and its independent reference:

```bash
uv run pytest -q tests/test_model.py tests/test_reference.py
```

On the repository version used for this article, this reports `53 passed`. These tests exercise masking, shape rules, weight tying, forward and backward calculations, and agreement with the slow reference implementation.

On an Apple Silicon Mac with the Metal GPU available, you can also run the full-size gate:

```bash
uv run python scripts/check_model.py
```

This is the heavier option: it takes a few minutes, runs the complete unit-test suite first, and then performs the GPU/CPU, two-attention, precision, causality, and one-batch-memorization checks described below. Success ends with `all checks passed`. It is Apple-only because the gate deliberately checks the `mps` device; use the two CPU commands above on other platforms.

After you have trained your own run, the same Apple-only gate can attack its learned attention patterns:

```bash
uv run python scripts/check_model.py --trained runs/my-run/best.pt
```

Replace `my-run` with your run name. A fresh clone does not contain `best.pt` files—training checkpoints are intentionally excluded from Git—so this command requires a locally trained run.

## Four gates before training

Neural-network bugs are unusually deceptive. A broken sorting function usually produces obviously wrong output. A broken model may still learn *something*, lower its loss, and write plausible text. We therefore built a full-size pre-training gate around four ideas.

### 1. Check the first loss against theory

An untrained model with 98 possible pieces should be surprised by roughly `ln(98) = 4.585` nats. Random starting scores make it slightly and incorrectly confident, adding about 0.077. The more precise prediction is 4.662; the measured first loss on random tokens is 4.674, within the allowed 0.03.

Random targets matter. Because each next token is independent of the preceding random tokens, there is no usable pattern, although individual guesses can still match by luck. The original check used real Shakespeare and required the loss to stay above `ln(98)`. That was not a theorem: lucky initial weights beat that line in 6 of 40 seeds.

This cheap check catches large initialization errors and misaligned targets. If the target is the current token rather than the next one, the untrained model already favors it because the same vocabulary table appears at input and output; the loss falls to about 3.5.

### 2. Change the future and demand an identical past

Take a real 256-token passage, choose a cut point, and scramble every token from that point onward. Predictions before the cut must remain identical, down to the last bit. Predictions after it must change, proving the test can observe a difference.

The gate does this at all 255 cut points under eight combinations: hand-written and built-in attention, 32- and 16-bit arithmetic, and training and evaluation modes. No earlier prediction changes.

### 3. Memorize one batch

Train the full model on the same four passages 200 times. Its loss falls from 4.64 to 0.005 in about two seconds. This proves that inputs, targets, loss, gradients, and optimizer are connected well enough to learn.

The first version used a learning rate of 0.001 and only one fixed batch. That rate was unstable: in independent trials the correct model failed on 4 or 5 of 10 batches. A check that fails randomly soon gets ignored. The revised check uses a stable setting and varied evidence.

### 4. Compare the GPU with the CPU

The same weights and batch run forward and backward on both devices. Predictions agree within 0.000002, and gradient differences are below one millionth of each tensor's largest entry. The comparison is repeated after memorization, when attention patterns are sharper and errors are easier to see.

These gates are useful. They were also not enough.

## The bug that passed every reassuring check

Attention must rearrange numbers when it splits 384 values into six heads and later joins them. There are multiple ways to produce the same output shape, and one of them puts the numbers in the wrong places.

An auditor deliberately used the wrong rearrangement both when splitting and when joining. The two mistakes cancelled enough to preserve causality. The broken model still had a normal first loss, memorized a batch, agreed across CPU and GPU, and produced identical results through the hand-written and built-in attention paths.

Yet each attention layer let the last position see only the nearest 43 positions instead of all 256. Across six layers, information could hop farther, but the model's effective reach stopped halfway through the window. Position 129 could influence nothing later than itself.

The model would probably have trained and generated plausible Shakespeare. We might have blamed the dataset or tokenizer for its weaker quality.

The failure exposed a shared assumption. Every existing check compared the model with itself under another input, device, or path, or merely asked whether it could learn. Both attention paths shared the same rearrangement code. Two implementations cannot validate code they share.

The fix was a deliberately slow reference that loops over batches, heads, and positions and shares no attention machinery. A second complete model written independently in NumPy also loads the same weights. In 64-bit arithmetic, its output and the PyTorch model agree within 0.000000000000004 on real text. Numerical gradient checks at 1,548 parameter locations agree as well.

## Testing the tests

The audit planted 61 faults; 17 passed the original 26 tests and full gate. Survivors included returning validation text when training text was requested, errors that appeared only on short prompts, evaluation-time dropout in one attention path, and 16-bit-only numerical faults.

The original checks had their own defects. The first-loss check made a false claim about real text. The memorization check was unstable but always used the one batch that happened to pass. The size check repeated a comparison the model already performed internally, so it could never fail.

After repair, model tests grew from 26 to 53 and the gate from 20 checks to 36. We then planted 34 faults ourselves. All 31 that can appear in a small model are caught by unit tests, and the three specific to 16-bit GPU arithmetic are caught by the full gate. This does not prove there are no other faults; it shows that the known holes are closed.

## What the machine changed about the plan

The audit also changed three operating decisions.

First, in PyTorch 2.14 on this Apple GPU, the built-in attention function uses its fused fast path only when gradients are off. During training it runs separate operations much like the hand-written version. In 32-bit the two are effectively the same speed; fresh-process measurements put both near 160 ms per step. In 16-bit, the built-in path is slower because it performs attention in 32-bit internally. We chose hand-written attention because the same understandable code runs during training, evaluation, and generation. The built-in path remains a cross-check.

Second, two runs with the same seed are not bit-identical on this GPU. Repeated tokens send gradients to the same vocabulary row, and those contributions can be added in a different order. The first difference is about two parts in ten million, but training amplifies it. By step 300, same-seed runs can differ by about 0.015 in validation loss. A seed still fixes starting weights, batches, and dropout, but not the final digits of GPU arithmetic.

Third, 16-bit arithmetic is about 1.7 times faster, but early 300-step measurements left it 0.02 to 0.03 behind in validation loss. That comparison included an early instability and was not decisive. Since a 32-bit run was expected to take minutes rather than days, 32-bit became the baseline and 16-bit became an experiment.

The historical battery benchmark measured 165 ms per 32-bit step with hand-written attention, about 99,000 tokens per second, and suggested roughly 14 minutes for 5,000 steps. The GPU was about 11 times faster than the CPU. Peak memory was later measured more accurately at 4.79 GB; an earlier 5.5 GB figure came from a driver counter that moved in 1 GB jumps and remains in the build log as history. These were single-pass battery measurements, so Part 5 re-measures on mains power during real training.

## Decisions carried into training

Before seeing any trained result, the project fixed these choices:

- Train with hand-written attention and 32-bit arithmetic.
- Apply weight decay to matrices but not normalization scales. A norm's neutral scale is 1, so pulling it toward zero would be harmful rather than neutral.
- Compare tokenizers by passes over their different-length token streams, not by one fixed number of steps.
- Score the complete validation text in overlapping windows, giving each token enough preceding context and counting each token exactly once.
- Add all token surprises first and divide once at the end, avoiding biased averages over unequal batches.
- Re-run the no-future-peeking gate after training, when learned attention is sharper.

These decisions matter because changing a rule after seeing results lets the result choose its own exam.

## Takeaways

- GP-Thee is a six-block, six-head, 384-wide GPT-style model with 10,757,760 parameters under the character vocabulary.
- The causal mask prevents the model from copying answers that are already present one position to the right.
- Falling loss and plausible output do not prove the architecture is correct. A model with half its intended context passed every original gate.
- Self-consistency checks cannot expose a shared mistake. A slow, independent definition of attention was the decisive reference.
- Tests need to be attacked too. Random failures, impossible-to-fail assertions, and default-only paths create confidence without evidence.
- Hardware and framework behavior are part of the system: attention paths, numerical precision, memory measurement, and nondeterministic reductions all changed the training plan.

[← Part 3: the tokenizer — simplified](03-the-tokenizer-simplified.md) · [Next: Part 5, training — simplified →](05-training-simplified.md)
