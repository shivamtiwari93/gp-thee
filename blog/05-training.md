# Building GP-Thee, part 5: training, and what a number goes through before we believe it

*The loop that teaches the model is six lines. This part is about the other four hundred: how to score a model honestly, what a good score even is, how long to train, and how a training script that produced correct numbers still managed to be wrong almost everywhere else.*

[Part 4](04-the-model.md) ended with a model that had been checked every way we could think of and knew nothing. This part presses go. Part of the way through its training, GP-Thee writes things like this, from nothing but the word `HAMLET.` on a line of its own:

```
HAMLET.
The most bride of the heavens of good well-grace.

POLONIUS.
Not then? Angry me too.

HAMLET.
And these wars. Pilotus hath winsowed their hands.
```

Nobody told it that Polonius is in the same play as Hamlet. Nobody told it what a play is. (That sample is from step 3,000 of the second run described below, a third of the way through it. I picked it from 143 samples because Polonius walks in. Most are not so lucky.)

The same note on credit as before. I wrote the training code. Most of what was *wrong* with it was found by AI agents asked to attack it, each with tools of its own. They are the "auditors" and "reviewers" in what follows, and this post says which is which.

## Six lines

Here is the part of [src/gp_thee/train.py](../src/gp_thee/train.py) that does the learning. It is tidied for the page: our settings are written in as numbers, and two details are left out (the switch for 16-bit arithmetic, and keeping the gradient's length for the log).

```python
tokens, targets = random_batch(training, 64, 256, batches, device)   # take a batch
loss = model(tokens, targets)[1]                                     # run the model, measure the loss
optimizer.zero_grad()                                                # forget the last step's gradients
loss.backward()                                                      # work out the new ones
clip_grad_norm_(model.parameters(), 1.0)                             # clip them
optimizer.step()                                                     # nudge the parameters
```

Almost every PyTorch training script has these lines in it somewhere. (The clip is the optional one.) `batches` is the seeded dice that decide where the windows are cut. The model hands back two things, its scores and its loss, and `[1]` picks the loss. One more line sits in front of these six in the real loop and sets the learning rate for the step. It gets a section of its own below.

One trip through them is called a **step**. Line by line:

**Take a batch.** 64 windows of 256 tokens, cut from random places in the training text, and the same windows moved along by one token as the answer sheet (part 4). That is 16,384 predictions to mark per step.

**Run the model, measure the loss.** The loss is the average surprise over those 16,384 predictions: minus the logarithm of the probability the model gave to the token that really came next. An untrained model is about as surprised as a blind guess among 98 tokens: 4.6.

**Work out the gradients.** The model has 10,757,760 parameters. For each one, PyTorch works out one number: if this parameter grew a tiny bit, how much would the loss change? That collection of 10,757,760 numbers is the **gradient**. It is the whole trick of deep learning. In arithmetic it costs about twice what running the model cost. (That is the textbook count. We did not time it.) `zero_grad` comes first because PyTorch *adds* new gradients to old ones unless told to forget them. A loop without that line still runs and still learns a little, which is the worst kind of bug.

**Clip them.** Think of the gradient as one long arrow with 10,757,760 coordinates. Its length is what Pythagoras says: square every number, add them up, take the square root. If the arrow is longer than 1.0, every number in it is scaled down by the same factor until its length is 1.0. The direction is kept and the size is capped. With the plainest update rule, that is what stops one freak batch from throwing the model across the room. With the rule we use (next paragraph) the size of a step does not follow the size of the gradient, so the clip has a quieter job: it stops one freak batch from swamping the running averages that rule keeps. In our runs it is insurance that is almost never claimed. The log counts it. In each of the baseline runs at the end of this post the clip cut in on about one step in ten during the first 250 steps, when the arrow is at its longest, and on none of the 9,700 steps after that, where a typical arrow has length 0.26.

**Nudge the parameters.** Each parameter moves a little in the direction that lowers the loss. How little is the **learning rate**. The plainest rule would be: new value = old value − learning rate × gradient. The routine that does the nudging is called the **optimizer**, and ours uses a better-behaved rule called **AdamW**. It keeps two running averages for every parameter: of its recent gradients, and of their squares. A running average forgets slowly. The first remembers about the last 10 steps and the second about the last 100. AdamW steps along the averaged gradient, divided by the typical size of that parameter's gradients. The effect is that a parameter's pace no longer depends directly on whether its gradients are naturally large or tiny. The learning rate sets the usual scale of a move, not a hard ceiling: because the two averages forget at different speeds, a move can sometimes be larger. One whose recent gradients agree with each other moves at close to the full rate. One whose gradients keep changing sign hardly moves. And one noisy batch cannot yank anything in proportion to its raw size. The "W" is the **weight decay** from part 4: each step also shrinks every matrix by a hair, so that a number has to keep earning its size.

### The learning rate is not one number

Here is the learning rate over one whole run. Each column is a twentieth of the run.

```
rate
0.0010 │****
0.0009 │    **
0.0008 │      **
0.0007 │        *
0.0006 │         **
0.0005 │           *
0.0004 │            *
0.0003 │             **
0.0002 │               **
0.0001 │                 ****
       └┬─────────┬─────────┬── step
        0        half      last
```

The climb at the very start is too narrow to draw: 100 steps is 2% of a 5,000-step run. Two things to take from the picture. The rate comes down slowly at first, fastest in the middle, and slowly again at the end: half way through a run it is still above half its peak. And the shape is stretched to fit the run: a run twice as long keeps its rate high for twice as many steps.

**The climb at the start is the warm-up.** A fresh model is easy to improve, and every batch agrees about how. That is the danger. AdamW divides away most of the gradient's raw size, so a parameter whose recent gradients all agree often moves by about the current learning rate. At the very first step, after correcting the two averages for having just started, the division is close to plus or minus one for an ordinary nonzero gradient. Without a warm-up, every parameter with such a gradient therefore moves by about the full 0.001 at once; a zero or tiny gradient moves less, and the rate is not a mathematical cap. The starting weights are around 0.02, so a few such steps across millions of parameters is a big shove. The model overshoots, the next gradient points back the way it came, and the loss jumps. In the first version of our benchmark, which used the full rate from the first step, the loss fell to 3.4 and then jumped to 7.2 at step 5, with the clip switched on. (The build log, entry 13, has it. Where it jumps, and how far, depends on the seed.) So the rate climbs from almost nothing to its peak of 0.001 over the first 100 steps.

**The slide afterwards is for the opposite problem.** Late in a run the model is close to somewhere good, and big steps make it rattle around the spot instead of settling into it. So the rate slides down half a cosine wave to a tenth of its peak. Remember this picture. It comes back to bite in the section on how long to train.

### We did not invent the recipe

Every number here (the peak rate, the floor, the 100 steps, the clip at 1.0, the decay of 0.1, AdamW's two memory lengths, the 20% **dropout** from part 4, 64 windows of 256) is the recipe Andrej Karpathy's nanoGPT uses for its character-level Shakespeare model. We took it unchanged and did not tune it. It was tuned by someone else, for a character model, on a file a fifth the size of ours. Part of the honest answer to "is this the best GP-Thee could be?" is "nobody has checked". (One of those numbers I had explained backwards in a code comment. An auditor read nanoGPT's config and put me right. The story is in the build log.)

### Counting in passes

Most tutorials count **epochs**: one epoch is one trip through the data in order. We cut windows from random places, so we have no epochs. We count **passes**: one pass is as many tokens as the training text holds. For the character tokenizer that is 294 steps.

A pass is not "every token once". Windows land at random, so in one pass about 37% of the text is never read at all, and about 26% is read more than once. (The 37% is 1/e, which turns up whenever you throw as many darts as there are squares.) For a typical token away from the two ends of the stream, after 17 passes the chance that it was never read is about one in 24 million. The few tokens near the ends can sit in fewer windows, a negligible edge effect on those percentages.

## Scoring honestly

The training loss is no use for judging the model. It is measured on text the model is being trained on, and with dropout on. Dropout is a handicap we put on the model on purpose while it trains, so that it cannot lean on any one value (part 4). When we score the model we take the handicap off, because we want its best guess. At the end of our first run the training loss was 1.139 nats, which is 1.64 bits. The same kind of text, scored with the handicap off, came to 1.52.

Part 2 set aside two whole plays, *All's Well That Ends Well* and *Romeo and Juliet*, that the model never trains on. That is the **validation** text, and every number in this post that says how good the model is comes from there. (Three more works are the **test** text. Nothing in this post has looked at them, and since part 3 the code refuses to open them.)

Part 4 settled how the score is taken. The whole validation text is scored every time. The windows overlap, so that every token is predicted with at least 128 tokens behind it. (The first 127 have what there is.) Each token is scored exactly once. The surprise is summed over all 274,728 tokens and divided once at the end. (That is one more than the 274,727 characters you will meet below. The extra one is the START token between the two plays. It is scored, and it stands for no character.)

### Bits per character

The loss is in **nats**, because it uses the natural logarithm. Dividing by ln 2 = 0.693 turns nats into **bits**, which have a meaning you can feel: one bit is the surprise of a fair coin toss. A model that scores 2 bits on a character gave the character that really came a chance of 1 in 4, which is what someone choosing blindly among 4 equally likely options would have given it. 3 bits: 1 in 8. Lower is better, because fewer bits means less surprise.

And we divide by **characters**, not tokens. Part 3 built tokenizers whose pieces are whole words and word fragments. Such a model makes about a third as many predictions (31% to 40% for our four vocabularies), and each one is harder. Its loss per token is higher and means nothing next to a character model's. Total bits divided by the number of characters in the text is the one figure every tokenizer can be compared on. All five of ours agree that the two validation plays hold 274,727 characters.

(Small print, for part 6. A word-fragment model could spell the same text with other pieces: ` the` as one piece, or as ` t` and `he`. Whatever probability it gives those other spellings is lost, because we score only the one spelling our tokenizer produces. So the figure charges a word-fragment model a little too much, and a character model, which has one spelling for everything, exactly right. We have not measured how much.)

Blind guessing among the 97 characters of our alphabet costs log2(97) = 6.60 bits per character. (It is the 4.6 nats from the top of this post, divided by 0.693.) That is the score for knowing nothing. It is not quite a ceiling: a model that is sure and wrong scores worse, and our untrained model will, a little. What is the floor? Nobody knows. In 1951 Claude Shannon had people guess the next letter of a modern English book, and put the answer between 0.6 and 1.3 bits per letter. His alphabet had 27 symbols: the letters and a space. Ours has 97, and our English is four hundred years old. So his figure is a landmark, not a floor for our number.

### What is a good score? Ask a zip file

A score such as "1.77 bits per character" means nothing on its own. How low is good? So before training anything, [scripts/baselines.py](../scripts/baselines.py) scores some predictors that contain no neural network. All are fitted on the same 39 training works and scored on the same two validation plays.

| Predictor | Bits per character |
|---|---|
| Blind guess among 97 characters | 6.600 |
| Knows how common each character is | 4.783 |
| Looks back 2 characters (a "3-gram") | 2.902 |
| Looks back 4 characters (a 5-gram) | 2.258 |
| Looks back 5 characters (a 6-gram) | **2.231** |
| Looks back 7 characters (an 8-gram) | 2.408 |
| bzip2, after reading the training works | 2.299 |
| xz, after reading the training works | 2.390 |

An **n-gram** model is the classic pre-neural language model: count, in the training text, which character followed each run of n − 1 characters, and predict from the counts. The trouble is the unseen. In the training works `Rome` occurs 351 times, followed by a full stop, a space, a comma, a line break or some other mark, and never once by an `o`. The word Romeo does not occur in them. So each count is blended with the prediction from a shorter look-back, leaning on the shorter one when the longer has been seen rarely or has been followed by many different characters. We used a blending rule with no knob to tune (it is called Witten-Bell), because a baseline tuned on the validation plays would have had a look at them.

Looking further back helps up to five characters and then hurts. With a look-back of seven, one validation character in five follows a run of seven that never occurs in the training works (measured: 19%, against 5% for a look-back of five), and the counts that remain are thin. A cleverer blending rule would cope with that better than ours does. So this is a modest bar, and what matters is the margin by which the model clears it.

**A file compressor is a predictor in disguise.** It writes few bits for what it expected and many for what it did not, so compressed size in bits divided by characters *is* a bits-per-character score. We let each compressor read the training works first and charged it only for the extra bytes the validation plays cost. One difference from the other rows: a compressor keeps learning while it reads the validation plays, so after a few speeches it knows `ROMEO.` The n-gram table and our network are frozen before they see a word of them. (Treat the bzip2 row as rough. bzip2 works in blocks of 900,000 bytes, and its score moves between 2.27 and 2.47 depending on where a block edge happens to fall in the training text. Measured by an auditor, and again by me. xz's hardly moves.)

An auditor wrote all of these baselines again from scratch, without seeing our code, and got the same numbers to every printed digit.

So the bar is **2.23**. A model that cannot beat a table of six-letter counts has learned nothing worth having.

## Pressing go

The first real run: the character tokenizer, for 17 passes. That is nanoGPT's 5,000 steps, rounded to whole passes over our text: 4,992 steps. It took 20 minutes on the laptop's GPU. Every 250 steps the run pauses, scores the validation plays, and writes a sample. We will call each of these pauses a **stop**. At every stop the run also scores a sample of the *training* text the same way. It is the same size as the validation text and is taken from 16 places. We call that the **seen** score, and it matters later. Every sample starts from the same prompt and uses the same dice, so the samples differ only because the model does.

Part 4 promised its timings again on mains power. Measured, as the median of three rounds: 196 ms per step in 32-bit (177 to 212), where the battery run in part 4 said 165 ms. In 16-bit it is 103 ms, and on the CPU 1,520 ms. Slower on mains is not what we expected. The likeliest reason is heat: the benchmark came straight after an hour of GPU audits. The real runs agree with the slower figure, which is why 4,992 steps took 20 minutes and not the 14 in part 4's table: about 16 minutes of stepping, and the 21 stops make up the rest. A short benchmark on a cool machine flatters it. (Part 4 also said 16-bit "trails by 0.02 to 0.03". That was measured through the loss spike described above. With a warm-up, three tries put 16-bit 0.002 to 0.009 nats behind after 300 steps, which is less than two 32-bit runs from one seed differ. The question gets a full-length answer at the end of this post.)

Before the first step, 6.71 bits per character, a shade worse than blind guessing:

```
HAMLET.
htkâW0bCED…RRkBX6Cs O)A3”-.KO1n_05PækÀâ;CtWj3N(éîî,PSLSSbææ[ëN66œœu,S?GA;
```

(The first of four lines like it.) After 250 steps, under a minute, less than one pass. 2.95 bits. It has found spaces, line lengths, capital letters at the start of lines, and the shape of a speaker label, which it invents freely:

```
HAMLET.
The mothe this a mo a he sold nothe kere.

SPAS.
What has, be lard goo hous ther I deall, more to to so waing cile ant
And wick thy na it.
```

After 1,000 steps, four minutes, 2.12 bits. It has passed the six-letter table, and most of the words are words:

```
HAMLET.
The mother is the servants.
O no good well! Pray thou hast thou busined,
Thy faulty false life to thyself.
By call my virtue! so I know. I am so gone.
```

At the end, 1.77 bits:

```
HAMLET.
The most bribe of servants.
O God, there the court of her, the prisoner, great
More false like than the wars. Pisa is very well.
Then I’ll fight into him that I am sent
```

Here is the whole run at seven of its 21 stops. Lower is better, and the bar to beat was 2.23.

| Step | Passes | Minutes | Training text, scored the same way ("seen") | Validation |
|---|---|---|---|---|
| 0 | 0 | 0 | 6.703 | 6.706 |
| 250 | 0.9 | 1 | 2.950 | 2.948 |
| 1,000 | 3.4 | 4 | 1.979 | 2.120 |
| 2,000 | 6.8 | 8 | 1.739 | 1.915 |
| 3,000 | 10.2 | 12 | 1.630 | 1.842 |
| 4,000 | 13.6 | 16 | 1.555 | 1.795 |
| 4,992 | 17.0 | 20 | 1.520 | **1.771** |

Two to the power 1.77 is 3.4. So the model's total surprise is what you would run up by choosing blindly among 3.4 equally likely characters at every position, where the six-letter table was choosing among 4.7. (That number has a name: **perplexity**.) It is an average of a particular kind, and no single character looks like it. Measured on this model: 45% of the characters cost under half a bit, because once a word has begun the rest of it is nearly certain. The first letter of a word costs 3.8 bits, a choice among 14. And 15% of the characters cost more than 4 bits each. The difficulty of Shakespeare is in where the next word starts.

### The score, taken apart

[scripts/evaluate.py](../scripts/evaluate.py) splits the same total by play and by kind of line:

| | Characters | Bits per character |
|---|---|---|
| Both validation plays | 274,727 | 1.771 |
| *All's Well That Ends Well* | 133,386 | 1.715 |
| *Romeo and Juliet* | 141,341 | 1.823 |
| Speaker-label lines (`ROMEO.`) | 16,346 | 2.179 |
| Everything else | 258,381 | 1.745 |

Part 2 warned about this. `ROMEO.`, `PAROLLES.` and `BERTRAM.` appear nowhere in the 39 training works, so the model has to spell each of them from nothing, every time they speak. (`ROMEO.` alone heads 162 speeches.) Speaker labels are 6% of the characters and cost 25% more per character than everything else. (Part 2 said 3%. That counted only the labels whose names never head a speech in the training works: 858 of the 1,773. All speaker labels together are 6%.)

Those two halves could not be more different. The 915 labels with a name the training works also use (`KING.`, `NURSE.`, even `JULIET.` and `HELENA.`, who have namesakes) cost 0.92 bits per character. The 858 with a name they never use cost 3.45. And in a play the model *has* read, a label is among the cheapest things on the page. Measured on *Hamlet*, a training work, with the model from the next section: 0.34 bits per character on the labels, against 1.44 on everything else.

Training for longer makes the unknown names worse, not better. After 34 passes (next section) everything else had improved from 1.745 to 1.704, and the speaker labels had got *worse*, from 2.179 to 2.310. A reviewer of this post took the label lines apart to see where, and I measured it again. All of the damage is in the letters after the first letter of a name the model has never met: those went from 4.04 bits each to 4.62. First letters and full stops improved. The better the model knows the casts of the 39 plays it has read, the more firmly it expects one of those names once a label has begun, and the more an `OMEO` surprises it. (One pair of runs, one seed. The direction is what I would bet on, not the size.)

### An apology to the tutorials

Part 2 said: "expect our numbers to look worse than the ones in tutorials", meaning nanoGPT's well-known validation loss of 1.47. That figure is in nats per character. Ours is 1.227 (1.771 × 0.693). It came out lower, not higher, on plays the model had never seen.

That is not because we did anything cleverer, and the two numbers still cannot be compared. More text probably helps: the tutorial's file holds about 1.1 million characters and we have 4.8 million. But the texts differ, the alphabets differ (65 characters against 97), the splits differ (the last tenth of one file against two whole plays), and so does the ruler. The tutorial scores windows cut at random, and the first tokens of every window are predicted with nothing behind them. We give every token at least 128. Scoring our 34-pass model in windows that do not overlap gives 1.243 nats where our own ruler gives 1.206 (measured). So about 0.04 of the difference is the ruler alone. The warning in part 2 stands in both directions.

## How long should it train?

Look again at the table of steps above. At the last stop, step 4,992, the score was still falling. So we ran the same thing for twice as long, and then twice as long again.

| Run | Steps | Best validation score | Reached at |
|---|---|---|---|
| 17 passes | 4,992 | 1.7706 | the very last step |
| 34 passes | 9,985 | 1.7404 | step 9,750 |
| 68 passes | 19,969 | 1.7577 | step 10,750, barely half way; 1.7937 by the end |

Twice the training bought 0.030 bits. Doubling again gave 0.017 of that back: the 68-pass run's best moment, 1.758, is worse than the 34-pass run's 1.740, though still better than the 17-pass run's. (Hold those two sizes loosely. The end of this post runs the 34-pass setting nine times, and single runs turn out to wobble by about 0.006. The first step survives that: the 17-pass run is behind all nine. Of the second, what survives is that 68 passes were no better than 34. Neither size does.) The 68-pass run is a textbook picture of why more was not better. Its best score came at step 10,750. For the next 4,000 steps validation hovered between 1.761 and 1.777, and then it drifted up, to 1.794 by the end. Its seen score fell at every single stop, from 1.37 to 1.18. From the first step the model had been learning two things at once: Shakespeare, and these 39 works. In the second half of this run only the second kind of learning was left, and validation was paying for it. That is **over-fitting**, and it is the reason a validation set exists.

The more interesting thing is what happened on the way between the first two runs. An auditor saw it first, in my own two logs. These runs share a seed, so they start from the same weights and are dealt the same batches in the same order. The one setting that differs is the learning-rate picture near the top of this post: the long run's rate comes down more slowly.

| Step | 17-pass run | 34-pass run | The long run is |
|---|---|---|---|
| 1,000 | 2.120 | 2.122 | level |
| 2,000 | 1.915 | 1.927 | 0.012 behind |
| 3,000 | 1.842 | 1.851 | 0.009 behind |
| 4,000 | 1.795 | 1.811 | 0.016 behind |
| 4,750 | 1.775 | 1.786 | 0.011 behind |
| 5,250 | (finished at 1.771) | 1.768 | ahead, with 4,700 steps still to go |

Up to step 1,000 the two runs are level, to within 0.002. Between 1,000 and 2,000 they trade places. From step 2,000 on the long run is *behind* at every matching stop, twelve in a row, by 0.005 to 0.022. The likeliest reason is its learning rate, which is still high (0.00059 against 0.00011 at step 4,750), so it is still rattling around. A caution from part 4: on this GPU two runs from one seed are not identical to the last digit, and the difference grows. How far apart two such runs end is measured at the end of this post. So do not lean on any single row. Twelve rows in a row with the same sign are harder to explain away. A model taken from the middle of a long run is not the model a short run would have given you.

That is also the likeliest reason the 68-pass run lost even at its best moment. When over-fitting set in, at about 37 passes, its learning rate was still 0.0005, half the peak, and the model was still rattling.

This matters because of a rule we wrote down in part 3: each run is judged at its **best** validation moment, not its last. That rule protects against training too long. If the model starts over-fitting, validation turns upward, and we keep the checkpoint from before. It does nothing for a run that is too *short*. The 17-pass run's best moment was its last, its score was still falling, and nothing in its log could say that 0.03 bits were left on the table. "Best at the very end" is what an under-trained run looks like.

An auditor put numbers on the asymmetry with a miniature model on the CPU (a hundred times smaller, on 50,000 characters). A run up to five times too long cost about 0.01 bits after best-checkpoint selection. A run too short cost 0.2 to 0.6. Those sizes belong to the miniature, which was still learning fast at every length tried. On the real model the two costs we measured are closer together: half the right length cost 0.030, and twice the right length cost 0.017. What carries over is the shape. Keeping the best checkpoint puts a lid on the cost of training too long. Nothing puts a lid on the cost of stopping too soon.

So the length of a run is a setting like any other, and it has to be chosen by a rule fixed in advance. The auditor proposed this one, and I took it as it stood:

> Start near 5,000 steps. While doubling the run improves the best validation score by more than 0.01 bits per character, double again, up to 20,000 steps. If a run's best moment falls before two thirds of its length, also try half that length. Keep the shortest length within 0.01 of the best.

By that rule the character model trains for **34 passes**. Doubling from 17 gained far more than 0.01. Doubling again gained nothing, and its best moment came barely half way, so the rule says to try half of 68, which is the 34 we already had.

(Two honest footnotes. The 0.01 in the rule was fixed after I had seen the first two runs and before the third. The first decision, 0.030 against 0.01, is not close enough for that to matter. And all three of these runs were made by the first version of the training code, which stopped to score every 250 steps: 21, 41 and 81 chances to catch a lucky moment, the very unevenness the table further down lists as a fault. More chances flatter the longer runs, and the longest still lost, so the choice stands. Whether these gaps are large beside the wobble between one run and the next is settled at the end of this post.)

This will matter even more in part 6, which compares tokenizers. The same number of passes is about a third as many steps for a word-fragment model as for a character model. The same number of steps is about three times as many passes. Neither is a fair race. Each tokenizer will get its length by the rule above, found with a seed that takes no part in the comparison. To save pilot runs, the search is made for three of the five: characters, and the smallest and largest word-fragment vocabularies. The two in between borrow a neighbour's length in steps, unless the neighbours' lengths differ by more than a factor of two.

## "Seen" against "unseen", and what the gap is not

At the 34-pass run's best moment the two scores were 1.37 (seen) and 1.74 (validation).

I wrote a comment saying the gap shows "how much of what the model knows is just the training text by heart". Two auditors, independently, showed that it shows no such thing. One trained a model far too small to recite anything, for a single pass, and found a gap already. The other's argument takes one line: a 5-gram cannot hold more than five characters in a row, so it cannot recite a line of verse, and it shows a gap too. Measured: 1.912 on the works it counted, 2.258 on the validation plays. To be sure that is not just two hard plays, the auditor also scored every training work with that work left out of the counts: 2.301. A gap of 0.35 to 0.39, from a model that cannot know anything longer than five characters.

So a gap of this size *can* be entirely ordinary. Every play has its own names, its own places, its own running jokes and turns of phrase. A model that has read *Hamlet* is better at *Hamlet* than at a play it has not read, without having memorised a line. That does not show our network has memorised nothing. Unlike the 5-gram it can see 256 characters back, which is room for a speech. Whether it recites is a separate question with its own measurement, in a later part.

The number is still worth logging, for a narrower purpose. *Within* one run, if the seen score keeps falling while the validation score stops or rises, the model has started fitting what is particular to the training works. The 68-pass run above is the clear case. The 34-pass run shows the first sign of it: over its last 2,000 steps the seen score fell from 1.40 to 1.36 while validation barely moved, from 1.744 to 1.742.

## What was wrong with the other four hundred lines

While the first run trained on the GPU, four AI auditors attacked the training code on the CPU. The good news first. One of them wrote its own evaluator from a description of the method, without reading ours. The two agreed exactly on the full-size model, untrained, over the whole validation text: the largest difference in any token's surprise was 0.0 nats. It also reproduced on the CPU, to the fourth decimal, the score the real run had logged on the GPU before its first step: 6.7064.

The number was right. Nearly everything around it was wrong.

Three words first. A long run can die half way: a crash, a closed lid, a flat battery. So at every stop the run saves a **checkpoint**. It is a file that holds the weights, AdamW's running averages, the step number and the state of the dice. There are two of them: `last.pt`, the latest, and `best.pt`, the one with the best validation score so far. To **resume** a run is to start it again from `last.pt` instead of from nothing. Most of this table is about those two files.

| What was wrong | What would have happened |
|---|---|
| Resuming a run took its settings from the command line, not from the run | The resume command in my own docstring gave only the run's name, so every other setting fell back to its default. On a run started with another tokenizer it would load the saved model, train it on the *wrong* token stream, overwrite the last good checkpoint, and then crash. With any other changed setting it would carry on with the wrong settings and write them into the next checkpoint as if they were true |
| The loop ended on `step == last_step` | A resumed run with a smaller budget than the step it had reached would never end |
| Resuming a finished run | Crashed |
| The clock | Restarted at every resume, so reported training times would be too short |
| Checkpoints were written in place | A run killed during a save would leave no usable checkpoint at all |
| Starting a run in a folder that already held one | Replaced the old run's best model with an untrained one within seconds, and left the old results file to be believed |
| Scoring every 250 *steps* | A character run got 21 chances to catch its best moment; the largest word-fragment vocabulary got 8. In a comparison decided at each run's best moment |

None of these had touched a number yet. (The three runs above were made with this code. None of them was ever interrupted or resumed, and the independent evaluator vouches for their scores.) All of them were waiting for the first long night of unattended runs.

### 30 of 64

Then the usual exercise: plant a fault in the code, run the tests, see whether anything notices. The auditor planted 64. **30 survived all 45 tests.** Among the survivors:

- training on the *validation* text (every score would have been a memory test, and looked wonderful);
- overwriting the "best" checkpoint at every stop;
- setting the learning rate one step late, so the first step runs at full rate, which is the very thing the warm-up exists to prevent;
- weight decay on the norm scales, which part 4 spent a paragraph ruling out;
- no clipping at all;
- dropout never switched on;
- the seed ignored, so that "three seeds" would have been one run three times.

The tests were not careless. They were weak in one specific way, the same way as in part 3 and part 4: they checked that things *worked*, on the one path I had imagined. The tiny test run got better at every stop, so its best moment *was* its last, and "keeps the best" could not be told from "keeps the latest". It used seed 0 and default settings, so nothing that ignored the settings could be noticed. And nothing ever looked at what the optimizer was actually doing.

The repair that did the most is a test that writes the six lines out again by hand, inside the test, with every setting but one changed from its default (16-bit stays off, and has a test of its own), runs 16 steps, and demands that the real loop's weights match it bit for bit. If the real loop does its six lines in another order, skips one, or fails to pass one setting through, the weights differ. A second test scripts a run that over-fits (validation best at the second stop, then worse, while "seen" keeps improving), pulls the plug, resumes it, and checks that the best moment survived all of that.

### And again, one level further out

The training code was rewritten, the tests went from 45 to 81, and three fresh auditors were sent in. Their first finding was the good one: every fault in the table above was fixed, confirmed by running each case, including a real `kill -9` (the command that stops a program dead, with no chance to tidy up) of a real run. On the CPU, a run killed between two stops and resumed gives, bit for bit, the weights of a run that was never stopped. (On the GPU no two runs are bit for bit the same, as part 4 found. There a resumed run is as close to the unbroken one as two unbroken runs from one seed are to each other.)

Then one of them planted 28 faults in the *new* safety code, and **21 survived**. Checkpoints written in place after all. The guard against replacing an old run, cut in half. The checkpoint loader switched back to the unsafe kind. (A PyTorch checkpoint file can carry code that runs when the file is opened. Ours are opened in a mode that accepts numbers and text and nothing else.) My favourite: the test for "every checkpoint records the git commit of the code that made it" asked only that the record be at least seven characters long. The tests run in a temporary folder that is not a git repository, where the function answers `unknown`. Which has seven characters.

Another auditor went wider: 167 faults across the whole file. 122 were caught, and 28 of the first round's 30 survivors now die. The 45 that got through were all in the same places: saving, resuming, and the record of what made a run. None was in the arithmetic of the score. And 26 of the faults were caught by one test alone, the six lines written out by hand.

It is the same hole as before, moved outward. I had written code for what happens when things go wrong, and tested it only on the path where nothing does. There are 122 training tests now, 262 in the whole project. The ones added in this round kill a run at five different points inside a single stop and demand the same log, the same samples and the same best checkpoint as a run that was left alone.

One more from that round, because it is the kind this series keeps collecting. A run named `..` passed my check for "a plain folder name" and would have written 129 MB checkpoints into the top of the repository. And one the auditors did not find, because I found it the hard way an hour later: my own shell loop passed `--seed 1` as part of a run's *name*, and the code accepted a folder called `char-34-seed-1 --seed 1` and trained it with the default seed. Names with spaces are refused now.

## The check we promised

Part 4 ended with a debt. Every no-peeking check had run on a model that knew nothing. An untrained model attends to everything about equally, which can hide a leak. A trained one has sharp habits of attention, so if the future can leak into the past anywhere, this is where it would show.

```bash
uv run python scripts/check_model.py --trained runs/pilot-char-17/best.pt
```

255 cut points × two attentions × 32- and 16-bit arithmetic × gradients off and on, on validation text, with the trained weights. Largest change in the past: **0.0**, every time, on the first trained model and again on one of the final baseline models (`char-34-seed-3`, which the next numbers are from). The same script then compares the loss at every one of 2,048 tokens between the two attentions, on GPU and CPU. In 32-bit the worst single token differs by 0.00001. With 16-bit arithmetic included, the worst single token differs by 0.1 and the means agree to 0.0005.

## Rules, written down before the runs they govern

Part 3 fixed how tokenizers will be compared. The audit found two holes in that rule. It never said how long each tokenizer's model trains. And it said "closer than the spread between runs" without saying what the spread is.

Two words first. An **arm** of a comparison is one of the things being compared: here, one tokenizer, trained with several seeds. The **standard deviation** of an arm is the typical distance of its runs from their own average. One auditor tried the loose wording on made-up numbers: five arms that were truly identical, three seeds each. With "further apart than the two arms' own standard deviation" as the test, any two of them were declared different 31% of the time.

So, in [the build log](../docs/BUILD_LOG.md), committed before any of the runs it governs. The 40 stops and the 1.82 test are the auditors' proposals too.

- **Seeds.** Seed 0 belongs to the pilots. Comparisons use seeds 1, 2 and 3.
- **Length.** By the doubling rule above, found with seed 0.
- **40 stops per run**, evenly spread, for every run.
- **Everything else is frozen**, including the peak learning rate, which is not tuned per tokenizer. That is a limitation, and it leans one way. The recipe was tuned, by someone else, for a character model, and ties also go to characters. So "characters win" will be the weaker finding, and "a bigger vocabulary wins anyway" the stronger one. The window is 256 *tokens* for every tokenizer, which is 256 characters for one and 640 to 820 for the others. That is part of what is being compared. And 34 passes is the right length for *this* recipe: with more dropout a longer run might have kept improving. We did not try.
- **"The spread"** is the standard deviation over seeds. Three seeds give a poor estimate of it, so we **pool**: each run is measured against the mean of its own arm, and one spread is worked out from all fifteen runs. Two arms differ only if their means are more than 1.82 pooled standard deviations apart. (That is the ordinary 95% test for five arms of three seeds.) The procedure: find the arm with the lowest mean; the winner is the *smallest* vocabulary whose mean is within that distance of it. The auditor's simulation of five equal arms: any one pair is wrongly called different 5% of the time, as designed; at least one of the ten pairs, 25% of the time, which is why only the procedure decides; and under it the character arm is wrongly passed over 7% of the time. With three seeds a real difference has to be about three standard deviations before we would catch it nine times in ten. Small differences will be ties.
- **A run is thrown away only if its loss stops being a number.** It is re-run with a new seed, and reported.
- **One seed twice.** Part 4 found that two runs from the same seed drift apart on this GPU. So seed 1 is run twice, and the distance between the twins is reported beside the spread over seeds. A third run of seed 1, in 16-bit arithmetic, settles part 4's open question. If it lands no further from the twins' mean than the larger of their own difference and the standard deviation over seeds, the tokenizer comparison may run in 16-bit, every arm alike. The published baseline stays 32-bit either way.
- **Two expectations, in advance.** Plays differ: an auditor scored a 5-gram on each training work with that work left out of its counts, and the works spread with a standard deviation of 0.14 bits. So the final test score may land 0.1 to 0.2 away from the validation score for no better reason than which works they are. And the winner's validation score will flatter it a little, because it was chosen for that score. The same goes for every "best moment" in this post.

One more thing the reader is owed about every number here: they rest on two plays, which the model itself scores about 0.1 apart. Resampling the validation text in blocks of 5,000 characters, a single score such as 1.74 is uncertain by about 0.013 bits from the choice of text alone, and more with longer blocks. The *difference* between two models scored on the same text is far steadier: 0.030 give or take 0.003 for the 17- and 34-pass runs (measured). That is why this post argues from differences and not from levels.

## The baseline: three seeds, and one seed twice

Everything above this line came from single runs with seed 0, made by the first version of the training code. These runs are the real thing: the rewritten code, 40 stops each, 34 passes, 32-bit, seeds 1, 2 and 3. Then seed 1 again, and seed 1 once more in 16-bit.

| Run | Best validation score | At step | Score at the end | Seen, at the best moment | Minutes |
|---|---|---|---|---|---|
| seed 1 | 1.7490 | 7,239 | 1.7551 | 1.428 | 37.7 |
| seed 1, again | 1.7517 | 8,986 | 1.7614 | 1.379 | 35.8 |
| seed 2 | 1.7597 | 8,986 | 1.7663 | 1.374 | 35.8 |
| seed 3 | 1.7504 | 9,735 | 1.7584 | 1.363 | 36.0 |
| seed 1, 16-bit | 1.7592 | 8,238 | 1.7685 | 1.403 | 21.7 |

**The baseline: 1.7535 bits per character, with a standard deviation of 0.0054 over three seeds** (measured; the two runs of seed 1 are averaged first). The six-letter table scores 2.231. (A fourth seed, seed 0, joins below for a reason of its own. With it the mean is 1.7531 and the standard deviation 0.0045, and that is what `docs/results-char-34.json` records.)

**The twins.** Same seed, same code, same machine, two hours apart: 1.7490 and 1.7517. They differ by 0.0027, and they did not even have their best moment at the same stop. Part 4 found the cause (one operation on this GPU adds numbers up in a varying order) and measured it over 300 steps. This is what it grows into over 10,000: the twins are up to 0.019 apart at single stops, and up to 0.013 in the second half. It is not a fresh coin toss at each stop, either. The second run is the worse one at 18 of the last 21 stops, by 0.005 on average. Two runs that part company stay parted.

**A scare, and what it taught.** The pilot run in "How long should it train?" used seed 0 and the first version of the code, and scored 1.7404. That is lower than every run in the table above. A lucky seed, or had my rewrite made training worse? So I ran more. Each row is one full run of 34 passes:

| Code | Seed | Best validation score | Training loss, second half of the run |
|---|---|---|---|
| first version | 0 | 1.7404 | 1.0915 |
| first version | 0, again | 1.7423 | 1.0913 |
| rewritten | 0 | 1.7525 | 1.0916 |
| rewritten | 0, again | 1.7514 | 1.0914 |
| first version | 1 | 1.7571 | 1.0925 |
| rewritten | 1 | 1.7490 | 1.0927 |
| rewritten | 1, again | 1.7517 | 1.0924 |

After the first three rows it looked like a verdict against the rewrite: two runs of the old code, both about 0.010 better than the new code with the same seed. The last row but two settles it the other way. With seed 1 the old code is *worse* than the new, by 0.007. Neither version is better. They are different.

How can they be different at all? On the CPU, where arithmetic is repeatable, the two versions give *bit-identical* weights from the same seed: I trained a small model with each and compared every number. They do the same sums. And on the GPU the training loss follows the seed and ignores the version, as the last column shows.

Here is the pattern in that table. The same program with the same seed, run twice, repeats to within 0.003: three pairs, 0.0019, 0.0011 and 0.0027 apart. Change the seed and the score moves by about 0.005. Change the *program*, in ways that change no arithmetic at all, and the score moves just as far: 0.010 one way for seed 0, 0.007 the other way for seed 1. Part 4 found that one operation on this GPU adds numbers up in an order that is not fixed. It seems that the order depends on the rhythm in which the program feeds the GPU, which is steady for one program and different for another. That last sentence is my guess at the mechanism, and I have not tested it. The table is measured.

**So one run is not a measurement, and one program is not either.** On this machine an edit that changes nothing mathematically still re-rolls the dice, as surely as a new seed does. All nine 32-bit runs of this setting together: mean 1.7505, standard deviation 0.0062, from 1.7404 to 1.7597.

That changes how to read the pilots, which were made by the first version of the code and are being compared with runs of the second. The 17-pass run (1.7706) is behind all nine 34-pass runs, by 0.011 to 0.030. So "longer than 17" stands, against a threshold of 0.01, though with less room than 0.030 had seemed to promise. The 68-pass run (1.7577) beats one of the nine, by 0.002. So "68 is no better than 34" stands too, and more than half of "68 lost 0.017" was the luck of one 34-pass run. The rule picks 34 either way, because it asks for a gain of 0.01 before it will double. But both pilots were single draws from a bag 0.006 wide, and I had been reading their third decimal.

**Where the noise lives.** Taking each run's score apart, the way we did for the first run, shows where the bag's width comes from:

| At each run's best moment | Seed 1 (the twins' mean) | Seed 2 | Seed 3 | Standard deviation |
|---|---|---|---|---|
| Everything | 1.7504 | 1.7597 | 1.7504 | 0.0054 |
| Speaker-label lines | 2.315 | 2.490 | 2.356 | 0.09 |
| Everything else | 1.7147 | 1.7135 | 1.7121 | 0.0013 |

Most of the spread in the headline number is in the speaker labels: 6% of the characters, and how well a run happens to guess names it has never met. The twins differ by 0.005 on ordinary text and by 0.125 on the labels. At the last step the pilot owes 0.011 of its 0.013 lead over the first rewritten seed-0 run to the labels. At the two runs' best moments it is the other way round: 0.003 of the 0.012 is labels and 0.009 is ordinary text. (Those moments are 1,263 steps apart, so that is not quite like for like. Nor is it as tidy at the last step of every run, where ordinary text spreads from 1.711 to 1.717 and the labels from 2.41 to 2.55: the labels are then the bigger half of the noise, not nearly all of it.) Seed 2 is the "worst" seed because it is the most surprised by names it has never met: the labels account for all of its deficit. This is why part 3 decided to report the two kinds of line separately when tokenizers are compared, and it will matter there: the deciding number is the first row, as written down in advance, and most of its noise comes from 6% of the text.

**The best moments** fell at 72%, 90%, 90% and 97% of the way through, and every run in the table ended 0.006 to 0.010 worse than its best. (The three seed-0 runs ended 0.002 to 0.003 worse.) So by 34 passes the over-fitting has just begun, which is where the rule was meant to put us.

**16-bit.** The rule said: allow it for the tokenizer comparison only if the 16-bit run lands no further from the twins' mean than the larger of the twins' own difference (0.0027) and the spread between seeds (0.0054). It landed 0.0088 away. So the comparison in part 6 will run in 32-bit, at 36 minutes a run and not 22. To be fair to 16-bit: its 1.7592 is inside the range of the 32-bit runs. And the scare above shows that any change of program re-rolls the validation score by up to 0.010, which 16-bit arithmetic certainly is. So the validation score is no evidence that 16-bit learns worse. A quieter number hints at it. The training loss follows the seed and ignores the program: all three 32-bit runs of seed 1 agree on it to 0.0003. The 16-bit run of seed 1 is 0.0027 above them (1.0952 against 1.0924 to 1.0927). One run, so a hint and not a finding. The rule was written down first, and it says 32-bit.

**Speed.** With nothing else loading the machine, a 32-bit run sustains 85,400 tokens a second, which is 192 ms per step, and 16-bit sustains 153,800. The 41 stops cost about four minutes of each run.

## Where we are

New in the repo:

```
gp-thee/
├── src/gp_thee/train.py            the loop, the schedule, evaluation, checkpoints, a small sampler
├── src/gp_thee/evaluation.py       the score taken apart by play and by kind of line; the n-gram and compressor baselines
├── tests/test_train.py             122 tests, one of which is the loop written out again by hand
├── tests/test_evaluation.py        33 tests
├── tests/test_summarise_runs.py    3 tests of the rule that calls two arms different
├── scripts/train.py                one run
├── scripts/evaluate.py             a trained run's score, taken apart
├── scripts/baselines.py            how well Shakespeare can be predicted without a neural network
├── scripts/summarise_runs.py       seeds gathered into a mean and a spread
├── scripts/check_model.py          now also: --trained, the gate's checks on trained weights
├── docs/baselines.json             the baseline scores
└── docs/results-char-34.json       every baseline run's score
```

```bash
uv run pytest                                                       # 262 tests, about a minute, no GPU needed
uv run python scripts/baselines.py                                  # under a minute, no GPU
uv run python scripts/train.py --name my-run --passes 34 --seed 1   # about 36 minutes on an M5 Max
uv run python scripts/evaluate.py --run my-run
uv run python scripts/check_model.py --trained runs/my-run/best.pt
uv run python scripts/summarise_runs.py char-34-                    # the seeds gathered: mean, spread, twins
```

Five parts in, GP-Thee has finally learned something. On two plays it has never read it scores 1.75 bits per character, give or take 0.006 from one run to the next, where a table of six-letter counts scores 2.23 and a blind guess 6.60. About 0.09 of that 1.75 is names it could not have known (0.08 to 0.10, depending on the run): that is what the 858 labels with unknown names cost beyond what the 915 known ones do.

The limits, in one place. The recipe is someone else's, tuned for another file, and we changed nothing. The length of a run was chosen from single runs, each a draw from a bag 0.006 wide, on the same two plays that do the judging. Those two plays differ from each other by 0.10, eighteen times the spread between our runs, so the test works may well score a tenth of a bit away. And on this GPU any edit to the program, even one that changes no arithmetic, re-rolls the score as surely as a new seed.

What is owed, and where. Part 6 compares the tokenizers under the rules above. After that: whether the model recites its training text word for word, a proper sampler, and the test works, opened once, at the end.

Before we believed the number 1.75, it went through an independent evaluator, three rounds of planted faults, a second implementation of every baseline, nine runs where one had seemed enough, and a review of this post that caught its author quoting a sample from the wrong run. The six lines were the easy part.
