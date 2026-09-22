# Building GP-Thee, part 4: the model, and how we tried to make it cheat

*195 lines of model, and three times as many lines checking those 195. Still no training: this part is about earning the right to trust a training run.*

[Part 3](03-the-tokenizer.md) turned Shakespeare into numbers. This part builds the thing that reads them. It is the part of the project that looks most like "AI", and it is shorter than the tokenizer and half the length of the script that cleans the data. Most of this post is about something else: how you find out that a model you wrote yourself is the model you think it is.

That matters more than it sounds. A wrong sorting function gives wrong output, and you notice. A wrong neural network trains anyway. The loss goes down, text comes out, and nothing tells you that a third of it is broken. Every check in this post exists because there is a specific, quiet way to get this wrong.

A note on credit before we start. As in parts 2 and 3, most of what was *discovered* in this part was discovered by AI agents asked to attack the work, not by the author. The post says which is which.

## The model in one picture

Text goes in as token numbers. What comes out is, for every position at once, one score per piece of the vocabulary.

```
token ids
   │  look up a row of 384 numbers for each token, add a row for its position
   ▼
┌─ block, repeated 6 times ──────────────────────────────────────────────────┐
│  x = x + attention(norm(x))      each position gathers information from    │
│                                  itself and from earlier positions         │
│  x = x + feed_forward(norm(x))   each position thinks about what it        │
│                                  gathered, alone                           │
└────────────────────────────────────────────────────────────────────────────┘
   │  one last norm
   ▼
scores for the next token   (computed with the SAME table that looked the tokens up)
```

`x` is the whole window as a table: one row of 384 numbers for each position. `norm` is part 1's layer norm: it rescales each row so its numbers stay in a sensible range, then multiplies by a learned scale that starts at 1.

That is all of it: [src/gp_thee/model.py](../src/gp_thee/model.py). It is the arrangement GPT-2 uses, and the same shape as Andrej Karpathy's nanoGPT, with plain-English names. (The file has a small glossary for readers who know the usual ones: our `width` is their `n_embd`, our `to_scores` is their `lm_head`, and so on.)

**From scores to a loss.** A higher score means "more likely to be the next token", but scores are not probabilities yet. A function called **softmax** (e to the power of each score, divided by the total) turns the 98 scores into 98 probabilities that add up to 1. The loss from part 2 is the *surprise* at the token that really came next: minus the natural logarithm of the probability the model gave it. Certain and right costs 0. The less likely the model thought it, the more it costs.

Part 1 worked out the formula for this model's parameter count before the model existed, and by part 3 (97 characters plus START) the answer was 10,757,760. The model now counts itself when it is built, and refuses to exist if the two numbers differ.

### One window is 256 exercises

A language model is trained to predict token 6 from tokens 1 to 5. Feeding in a whole window to get one guess back would be wasteful, so the model guesses at every position at once: what comes out at position 1 is its guess for token 2, what comes out at position 5 is its guess for token 6, and so on. One 256-token window is 256 exercises marked in a single pass, and the answer sheet, called the **targets**, is simply the same window moved along by one token.

That is what makes training efficient. It also means the answers are in the input. The guess for token 6 is worked out at position 5, and token 6 itself is sitting at position 6, one place to the right, because it is also the input for the guess after that.

### Attention, in five lines

Attention is the step where a position looks at other positions. Each position makes three things out of its 384 numbers: a **query** (what am I looking for?), a **key** (what do I have?), and a **value** (what I will hand over if asked). How well one position's query matches another's key decides how much of that position's value gets mixed in.

Here is the whole calculation, as it appears in the code:

```python
match = (query @ key.transpose(-2, -1)) / math.sqrt(key.shape[-1])
match = match.masked_fill(~self.may_look[:length, :length], float("-inf"))
shares = F.softmax(match, dim=-1)
shares = F.dropout(shares, self.dropout, self.training)
mixed = shares @ value
```

**Line 1** compares every query with every key. (`@` is matrix multiplication. Each comparison multiplies two lists number by number and adds up the products, which comes out large when the lists point the same way.) The result is a 256 × 256 table of matches: one row per position asking, one column per position being asked. Dividing by the square root of 64, the size of a head, keeps those numbers from growing just because a head is big.

**Line 3** is softmax again, applied to each row: bigger matches get bigger shares, and every row adds up to 1. **Line 5** mixes the values in those proportions.

**Line 4** is dropout. During training it zeroes a random 20% of the shares and scales the rest up to keep the average the same, so the model cannot lean on any one of them. The same dropout is applied in three other places in the model. All four are switched off whenever the model is not training, and every check below runs with them off.

Notice what is *not* in those five lines: an explicit place in the order. Hand a position the same already-made keys and values in a different order, and its mix comes out the same. The position row added at the very start gives every place an address, so order is available from the first block. It is not quite the only order signal: the causal mask also means that earlier positions were built from shorter prefixes, and deeper blocks can recover some order from that. The position rows make it direct instead of making the model reconstruct it.

**Line 2** is the one this whole post is about.

### The rule that must never break

If position 5 could look at position 6, predicting token 6 would be copying. The loss would fall towards zero, the training curve would look like a triumph, and the model would have learned one skill: look one place to the right. When we ask it to write new text, there is nothing to the right.

`may_look` is a table that says, for each position, which positions it is allowed to see: itself and everything before it.

```
position   may look at
   1       1
   2       1 2
   3       1 2 3
   4       1 2 3 4
```

Line 2 sets every forbidden match to minus infinity before softmax runs, and e to the power of minus infinity is exactly 0: not a small share, no share at all. (Setting the match to 0 would not do. e to the 0 is 1, an ordinary share. And a position must be allowed to see itself, or position 1's row would be 0 divided by 0.)

This is called the **causal mask**: causal because, as with cause and effect, nothing later can influence anything earlier.

### The rest of a block

**Heads.** Six attention heads run side by side in each block. Every query, key and value is made from all 384 of a position's numbers and then cut into six slices of 64, and each head runs the five lines above on its own slice, with its own table of shares. That lets one position look in six places, for six different reasons, at once. One last matrix combines the six results.

**Feed-forward.** Two matrices with a bend between them (384 numbers out to 1,536 and back), applied to each position alone. The bend is a function called GELU, which passes positive numbers through almost unchanged and squashes negative ones towards zero. Without it, two matrices in a row are just one matrix, and stacking layers would add nothing.

**Residual connections.** The two `x = x + ...` lines add each step's result onto what was there instead of replacing it. Because every step only adds, what went in is still there at the end, and the corrections coming back from the loss reach the first block without having to pass through every later one. That is what makes a deep stack trainable.

We wrote attention out by hand, and we also kept a switch for PyTorch's own built-in attention function, so that two ways of computing the same thing could check each other. That turned out to be less useful than we expected, for a reason that is the main story of this post. And the built-in function held a surprise of its own, further down.

## Four ways to catch a model that is quietly wrong

Before any real training, a full-size model on the real GPU has to pass a gate: [scripts/check_model.py](../scripts/check_model.py). Four of its checks are worth understanding. They are described here as they stand today. The first versions were weaker, and the audit further down is how we found that out.

### 1. A model that knows nothing should be as surprised as theory says

Before training, the model has no idea what comes next, so each of the 98 pieces gets a probability of about 1/98, and the surprise is ln(98) = 4.585.

One of the auditors worked out that "about" more exactly. Each score is a sum of 384 products: one of the position's numbers (the last norm leaves those with a spread of about 1) times one of the table's starting values (random, with a spread of 0.02). So the scores scatter around zero with a variance of 0.02² × 384 = 0.15. Scattered scores mean the model is slightly, and wrongly, confident, and for a small scatter the cost is half the variance: 0.077. Theory says **4.662**. The model's first loss on random tokens is **4.674**, and the gate allows 0.03 either way.

(The check uses random tokens, not Shakespeare, because the next random token is independent of the ones before it. A guess can still line up by luck, but there is no pattern that lets the model do better systematically. The first version used real text, and that was a mistake we will come to.)

This check costs nothing and catches a surprising amount. Starting values ten times too large give a first loss of 10.5. Targets that are not moved along by one, so that the model is asked to "predict" the token it was just given, give 3.5, far *below* 4.585: the residual line carries each token's own table row straight through to the output, the output scores are made by comparing with that same table, and a row matches itself better than it matches any other. Even an untrained model gives the token it has just read about three times the probability of any other.

### 2. The future cannot change the past

Take a real passage of 256 tokens and compute the model's predictions. Now scramble every token from position k onwards and compute them again. The predictions *before* position k must not move. If any earlier position were getting information from position k or later, by any route at all (a wrong mask, a norm that averages across positions, rows of a batch bleeding into each other), its prediction would change.

Not "move very little". They must be identical to the last bit. A share of exactly zero contributes exactly zero, and because the scrambled passage is the same length as the original, every sum inside the model is added up in the same order. (Part 1 showed why the order matters. Compare a 20-token prompt with the first 20 positions of a full window and the last digits do differ, harmlessly, so *that* comparison gets a small tolerance and this one gets none.)

We do this for every one of the 255 possible cut points, and we require that the predictions from k onwards *do* change, so that we know the test can see a difference when there is one. Result: no change at all, at all 255 cut points, in each of eight combinations: both attentions, 32-bit and 16-bit arithmetic, training mode and evaluation mode.

### 3. Learn one batch by heart

Give the full model four passages and train on only those, 200 times. If the model, the loss and the optimizer are wired together correctly, it must be able to memorise them: the loss went from 4.64 to 0.005 in two seconds. If it cannot, something is disconnected, and there is no point looking at anything subtler.

### 4. The GPU and the CPU must agree

Part 1 found bug reports about PyTorch's backend for Apple GPUs returning silently wrong numbers. So the same batch goes through the same model on both devices, forwards (tokens in, predictions out) and backwards (from the loss back to a gradient for every parameter). Predictions agree to 0.000002, and gradients to less than a millionth of each tensor's largest entry. The bug in part 1 was an error of 10 to 30%, so that is a wide margin.

But a fresh model is a weak witness. Its attention is almost uniform, and some real faults move its scores only in the sixth decimal place. So the gate repeats the comparison after check 3, with the memorised weights, where attention is sharp.

## The audit: the model was right, and the tests were not good enough

As in parts 2 and 3, four AI agents were asked to attack the work, each writing its own tools. Two went after the code, and two went after the machine: speed, repeatability and 16-bit arithmetic. What those two found is in "Three things we did not know", below.

The first wrote the entire model again, from the description alone, in NumPy with 64-bit numbers and plain loops, and loaded our weights into it. With our model also switched to 64-bit for the comparison, the two agree to **0.000000000000004** on the full-size model with real text. (In the 32-bit arithmetic the model really runs in, they agree to 0.000002, which is as close as 32-bit gets.) It also checked our gradients the slow, independent way: nudge one parameter up a hair and down a hair, measure how much the loss moves, and divide by the size of the nudge. That slope is what the gradient claims to be. It did this at 1,548 places: 1,440 in a tiny model, covering every tensor, and 108 in the full-size one. They match. **The model computes what its description says.**

The second did what found the real problems in part 3: it broke the code on purpose, one fault at a time, and watched whether anything noticed. It planted 61 faults. **17 of them passed all 26 tests and the entire gate.**

### The bug that would have cost us half the context

This is the one worth telling. Attention splits the numbers among 6 heads, works, and puts them back. Doing that involves rearranging a block of numbers twice, and there is a wrong way to do each rearrangement that produces the right shape.

Take the 2 × 3 table `[[1, 2, 3], [4, 5, 6]]` and ask for it as 3 × 2. Turning it on its side gives `[[1, 4], [2, 5], [3, 6]]`. Re-cutting the same run of six numbers gives `[[1, 2], [3, 4], [5, 6]]`. Both are 3 × 2, nothing crashes, and only one is what you meant.

Get one of the two rearrangements wrong and the future leaks into the past, and check 2 catches it at once. Get *both* wrong and the two mistakes cancel as far as peeking goes. With that fault planted:

- the model was still perfectly causal, so "the future cannot change the past" passed;
- its first loss was normal;
- it memorised a batch without trouble;
- the GPU agreed with the CPU;
- and our two attentions agreed with each other.

And in every attention layer, the last position of the window could look at only the 43 positions nearest to it, instead of all 256. The wrong rearrangement deals the 256 positions out among the 6 heads in order, 42 or 43 to each, and a position can then look back only within its own group. Stacking six layers lets information hop from group to group, but only so far. We measured it: **the whole model's reach, from the last position, stopped at the halfway point of the window.** Position 129 of 256 could see nothing but itself, however many layers there were.

It would very probably have trained, written plausible Shakespeare, and been quietly worse than it should be. We would have blamed the data, or the size, or the tokenizer.

Look at what those five checks have in common. Each one compares the model with *itself* (on another device, on different input, through the other attention) or asks only whether it can learn. A model that is consistently wrong passes all of them. Not one compared the model with the *definition* of attention.

And the two attentions could not catch it because the rearranging happens in code they share. Both paths split the heads with the same line and merge them with the same line.

**Two implementations cannot check the code they share.** What catches this fault is a reference that shares nothing: plain loops over rows, heads and positions, working out one position at a time. That is now a test, and so is the auditor's idea of a complete second model in NumPy ([tests/test_reference.py](../tests/test_reference.py)).

### The rest of the survivors

- **The loader handing back the validation text when asked for the training text.** Every test passed. Nothing compared a saved file with the works it claimed to be. We would have trained on 5% of the data and "validated" on the same 5%.
- **Anything that only goes wrong on short text.** Every test used a full window. Prompts will be short.
- **The hand-written attention dropping values during evaluation as well as training.** Our one dropout test covered one of the two paths and asked only whether anything at all was random.
- **A numeric fault that exists only in 16-bit arithmetic.** (Doing the heavy arithmetic in 16-bit numbers instead of 32-bit is a common way to train faster, at the cost of some digits. More below.) The gate checked 16-bit for peeking, never for giving the same answers.

### Our own checks were faulty too

Three of the gate's checks were themselves wrong, which is a humbling thing to read.

The memorise check used a learning rate of 0.001. (The learning rate is how big a nudge the optimizer gives the parameters at each step. Too big, and a step overshoots.) At that rate the loss jumped *upwards* by 2 to 4 between one step and the next in 9 of the auditor's 10 runs, and the correct model failed the check for 4 or 5 of 10 different batches. Ours never got the chance to fail at random, because we always used batch 0: the coin had been tossed once and glued down. So a pass told us nothing, and several of the faults it "caught" were the coin too.

The first-loss check ran on real text and demanded that the loss be at least ln(98). That is a theorem for random tokens and simply false for real text, where lucky starting weights can beat a uniform guess. The correct model failed the check for 6 of 40 random seeds, 5 of them by falling *below* ln(98). (A seed is the number that fixes a program's "random" choices, so that a run can be repeated. Ours had passed only because our seed was fixed.)

And the size check repeated a comparison the model had already made when it was built, so it could never fail.

A check that fails at random teaches you to ignore it. A check that cannot fail teaches you nothing, and this is the third part in a row in which we have found one of our own. Both are worse than no check, because they look like safety.

### After the fixes

The model tests went from 26 to 53, and the whole project to 104. The gate went from 20 checks to 36. Then we planted 34 faults ourselves, including every one that had survived. The 31 that can show up on a small model are all caught by the unit tests, and the 3 that exist only in 16-bit on the GPU are caught by the gate.

One caution. This second round was ours: we planted the faults and we wrote the tests. It shows that the holes the auditor found are closed, not that there are no others. (And one of our own planted faults "survived" at first. When we looked, the edit we had made did not change what the code computes, so there was nothing to catch. A fault has to be a real fault.)

## Three things we did not know

The other two auditors found three facts about our setup that changed decisions.

**1. PyTorch's built-in attention is not the fast one during training, in this version of PyTorch on Apple GPUs.** To work out gradients, PyTorch has to keep a record of every step of a calculation so that it can walk back through it afterwards. When you are only *using* a model, not training it, you tell PyTorch not to bother: that is "gradients off". PyTorch's attention function has a fused kernel for Apple GPUs (a kernel is a routine that runs on the GPU; fused means one optimised routine instead of several separate steps). But in PyTorch 2.14 it only takes that route when gradients are off: evaluation and generation. During training it quietly runs the same separate steps we wrote by hand. A later version may change this. It is a fact about the software, not the chip.

This had two consequences. Our "the future cannot change the past" check ran with gradients off, so on the built-in path it had been testing a *different piece of code* from the one training would use. And it explained a puzzle in our benchmark, which is in the next section: we had expected the built-in function to win. In 32-bit the two take the same time and the same memory, and in 16-bit the built-in one is actually slower, because it quietly does its attention in 32-bit.

So we train with the hand-written attention. It is as fast or faster, the same code runs in training, evaluation and generation, and it is the code this post explains. The built-in function stays as a cross-check, and the gate now runs the no-peeking check with gradients on as well as off.

**2. Two runs from the same seed are not identical on this GPU.** We set every seed, and the runs still drift apart. The auditor tracked it down to a single operation: the token-table lookup. The same token turns up many times in a batch (there are 98 tokens and 16,384 places), every occurrence sends its own nudge back to the same row of the table, and those nudges are added up in an order that varies from run to run. Part 1 showed why order matters: every addition is rounded, so a different order gives a different last digit. Every other operation, dropout included, is exactly repeatable.

The difference starts as two parts in ten million of one tensor's gradient, and it grows, because each step's weights decide the next step's gradients. By step 300, two same-seed runs differ by about 0.015 in validation loss. Neither run is wrong.

There is a way to make runs bit-identical, by writing the lookup as a matrix multiplication. We decided against it. It is an odd thing to teach, and the honest position is simple enough: a seed fixes the starting weights, the batches and the dropout pattern, but not the last digits of each step's arithmetic, and over a few hundred steps those grow into the second decimal place of the loss. When we compare settings, we will report how much same-seed runs differ alongside how much different-seed runs differ.

**3. 16-bit arithmetic is faster, and not yet shown to be equal.** Doing the heavy arithmetic in 16-bit is about 1.7 times faster here. In our first 100-step comparison its loss was already a little behind 32-bit, by about 0.01 at the end, though that run went through an early loss spike that makes it a rough comparison. Over 300 steps an auditor measured it 0.02 to 0.03 behind on validation loss, about twice the same-seed drift. That may wash out over a full run, or it may not. At 14 minutes a run, 32-bit is cheap, so the first real runs will use it, and 16-bit becomes an experiment instead of an assumption.

## How fast is it?

Until now, how long a training run takes has been an estimate. Part 1 guessed 10 to 30 minutes, from other people's numbers on older Macs. Measured, one training step on 16,384 tokens (64 windows of 256, the batch size nanoGPT uses), with the character tokenizer:

| | Time per step | Tokens per second | 5,000 steps |
|---|---|---|---|
| GPU, 32-bit, hand-written attention (what we will train with) | 165 ms | 99,000 | 14 minutes |
| GPU, 32-bit, built-in attention | 158 ms | 103,000 | 13 minutes |
| GPU, 16-bit, hand-written attention | 96 ms | 171,000 | 8 minutes |
| GPU, 16-bit, built-in attention | 105 ms | 156,000 | 9 minutes |
| CPU, 32-bit | 1,795 ms | 9,100 | 2.5 hours |

The estimate was right, at its fast end. The GPU is 11 times faster than the CPU. The two 32-bit rows are the same operations, and an auditor who re-timed them in fresh processes got 160 ms for each, so the 7 ms between them is noise. With the 2,048-piece vocabulary a 32-bit step takes 190 ms.

At its peak a training step holds 4.8 GB of memory in 32-bit, nearly all of it the intermediate results part 1 described, not the model. (That figure is an auditor's. The first version of the script said 5.5 GB because it read a driver counter that moves in steps of 1 GB; the [build log](../docs/BUILD_LOG.md) preserves that first reading. The [benchmark record](../docs/benchmark.json) in the current repo is the later mains-power rerun promised below, and records 4.79 GB.)

A caution about these numbers, in the spirit of the rest of the post: they were measured on battery power, in a single pass. A single pass in a fixed order cannot tell a real 5% difference from the machine warming up. The benchmark script now times every GPU configuration three times in rotation, reports the median and the range, measures memory at its peak, and records whether the laptop was plugged in. We will re-measure on mains power before training, and part 5 will carry those numbers. (The audit itself took the battery from 100% to 28%, which is its own kind of benchmark.)

One trap for anyone timing a GPU: it runs *ahead* of your program. A call returns before the work is done. If you read the clock without first waiting for the GPU to finish, you measure how fast Python can queue work, which is a much more flattering number.

## Decided now, for training

Part 5 trains the model. These were settled before it, so that results cannot steer them.

- **Hand-written attention, 32-bit arithmetic.**
- **Weight decay pulls matrices towards zero, and leaves the norms alone.** A norm's neutral value is 1, not 0. Each step, decay multiplies a number by (1 − learning rate × decay). At a learning rate of 0.001 and a decay of 0.1 that is 0.9999, and 0.9999 multiplied by itself 5,000 times is 0.61. With the learning rate tapering off, as it will in a real run, it is 0.76. Either way, a scale the loss had no opinion about would be dragged well away from 1 for no reason.
- **Count passes over the text, not steps.** Most tutorials count "epochs", an epoch being one pass over the data in order. We cut windows from random places, so we have no epochs, only an average number of passes. Part 3 already ruled out comparing tokenizers at a fixed step count. Now there are numbers: 5,000 steps is 17 passes over the character stream but 42 to 54 over the shorter word-fragment streams, two and a half to three times as many.
- **Evaluate on the whole validation text, the same way every time**, not on a random sample of windows, which would score slightly differently on each visit. Two details. If the text is cut into separate 256-token windows, the first tokens of every window are predicted with almost nothing to go on, and the model looks worse than it is. So the windows overlap, and each token is scored once, from a place with plenty of text before it. And the surprise is added up over all the tokens and divided once at the end, because averaging the averages of unequal batches gives the wrong answer.

## Where we are

New in the repo:

```
gp-thee/
├── src/gp_thee/model.py       the model: 195 lines
├── src/gp_thee/data.py        now also loads token streams and cuts training windows
├── tests/test_model.py        51 tests: no peeking, definitions, dropout, batching
├── tests/test_reference.py    2 more: the whole model again, in NumPy, sharing no code
├── scripts/check_model.py     the gate: 36 checks at full size on the GPU
├── scripts/benchmark.py       how fast this machine trains
└── docs/benchmark.json        the benchmark record; part 5 later replaces its first measurements
```

```bash
uv run pytest                          # 104 tests, 20 seconds
uv run python scripts/check_model.py   # the gate: the unit tests, then 36 checks on the GPU; about a minute
```

Four parts in, and the model has not learned a single thing. But we know exactly how many numbers are in it. We have tried 4,080 times to make the future change the past (255 cut points, eight combinations, two tokenizers) and failed every time. The GPU and the CPU agree to the sixth decimal place. And every fault that we or the auditors have thought to plant is now noticed by something.

None of that is proof, and all of it was measured on a model that knows nothing. So the gate's no-peeking check will run again on the first trained model. Next, in part 5: we finally press go.
