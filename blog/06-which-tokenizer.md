# Building GP-Thee, part 6: which tokenizer? A race whose rules were written before it was run

*Five ways of cutting Shakespeare into pieces, three training runs each, and a winner chosen by a rule that was fixed before the first of those runs. Plus what that rule could and could not have told us, which we also wrote down first, and an honest "I do not know" about why the winner won.*

[Part 3](03-the-tokenizer.md) built five tokenizers: single characters, and word fragments in vocabularies of 1,024, 1,536, 2,048 and 4,096 pieces, all learned from the training works alone. It ended with a promise. The vocabulary would be chosen by one number, averaged over three runs, and "if two tokenizers are closer than the spread between runs, the smaller vocabulary wins". [Part 5](05-training.md) trained the character model and found out what "the spread between runs" is for it on this machine: about 0.006 bits per character, whether the change is a new seed or a harmless edit to the program.

Every score in this post is part 5's measure, **validation bits per character**: the model's surprise at two plays it has never read. Lower is better. For scale: a table of six-letter counts with no neural network scores 2.23, and part 5's character model 1.75.

This part runs the race. The same note on credit as before: I wrote the code, and most of what was wrong with it, and with my explanations, was found by AI agents asked to attack them. They are the "auditors" and "reviewers" below.

## Why this race is hard to make fair

All five models are the same network from part 4. Only the embedding table changes size (part 3: one row of 384 numbers for every piece in the vocabulary), so the models have between 10.8 and 12.3 million parameters. What they are fed differs more than it seems. In the tables, as in the repo, the word-fragment tokenizers go by their file names: bpe-1024 is the vocabulary of 1,024 pieces, and so on. (BPE is byte-pair encoding, part 3's way of learning the fragments. A *piece* is an entry in the vocabulary. A *token* is one occurrence of a piece in the text.)

| Tokenizer | Pieces in the vocabulary | Characters per token | Training text, in tokens | One pass, in steps | What a window of 256 tokens holds |
|---|---|---|---|---|---|
| characters | 98 | 1.00 | 4,811,375 | 294 | 256 characters |
| bpe-1024 | 1,024 | 2.49 | 1,936,643 | 118 | about 640 characters |
| bpe-1536 | 1,536 | 2.66 | 1,787,863 | 109 | about 680 |
| bpe-2048 | 2,048 | 2.78 | 1,694,913 | 103 | about 710 |
| bpe-4096 | 4,096 | 3.07 | 1,507,176 | 92 | about 790 |

(All measured. The characters-per-token and window columns are measured on the validation plays. Parts 3 and 5 gave 640 to 820 for the window, which was on the training text, where more names are whole pieces.)

A word-fragment model reads the same works in a third to two fifths as many tokens. So one trip through the text is that many fewer steps, and a window of 256 tokens holds two and a half to three times as much Shakespeare. That second fact is an advantage we are handing the word-fragment models on purpose: the window is part of what a tokenizer buys you.

The first fact is the trap. Train every model for the same number of **steps** and the word-fragment models go round the text two and a half to three times as often. Train them for the same number of **passes** and they get a third to two fifths as many steps. Part 5 showed that the length of a run is a setting like any other, and that a run which is too short cannot tell you so: its log shows a score still falling, and not how far it had left to fall. So each tokenizer gets its own length, found by one rule with a seed that takes no part in the race:

> Start near 5,000 steps. While doubling the run improves the best validation score by more than 0.01 bits per character, double again, up to 20,000 steps. If a run's best moment falls before two thirds of its length, also try half that length. Keep the shortest length within 0.01 of the best.

This time the rule is a program, [scripts/find_length.py](../scripts/find_length.py), so that I could not bend it while watching the curves. Writing it as a program forced three readings of the sentence into the open. Is the double *always* tried? Yes: we cannot know that doubling does not help without trying it. Are halves halved again? Yes, down to 625 steps. Is "within 0.01" inclusive? Yes. All three went into the build log before the first run.

### What the rule found

These trial runs are the **pilots**. Each is a single run with seed 0, made to choose a length and for nothing else. For each vocabulary the rule asked for three: 5,000 steps first, then the double, which is always tried. Both 5,000-step runs had their best moment before two thirds of the way through, so the rule also asked for the half. Neither 2,500-step run peaked early, so the halving stopped there.

| Tokenizer | Steps | Passes | Best validation score | Best moment at | Score at the end |
|---|---|---|---|---|---|
| bpe-1024 | 2,500 | 21 | 1.8085 | 98% of the run | 1.8104 |
| bpe-1024 | 5,000 | 42 | 1.8001 | 57% | 1.8108 |
| bpe-1024 | 10,000 | 85 | 1.8045 | 30% | 1.9203 |
| bpe-4096 | 2,500 | 27 | 1.8515 | 90% | 1.8538 |
| bpe-4096 | 5,000 | 54 | 1.8733 | 30% | 1.9792 |
| bpe-4096 | 10,000 | 109 | 1.8759 | 15% | 2.2218 |

(All measured, seed 0, one run per row. Lower is better.)

**The rule picks 2,500 steps for both.** For bpe-4096, 2,500 is simply the best. For bpe-1024 the best is 5,000's 1.8001, and 2,500's 1.8085 is within 0.01 of it and shorter. As agreed beforehand, the two vocabularies in between borrow that length, and characters keep the 9,985 steps part 5 found.

Two things in this table are worth a second look.

**Word fragments over-fit early and hard.** Look at the last column. The 10,000-step run of the largest vocabulary ends at 2.22 bits per character on the validation plays. That is 0.35 worse than its own best moment, and about what the six-letter table scores with no neural network at all. All the while its *seen* score (part 5: the same ruler held to text it has trained on) kept falling, to 0.57. Seen falling while validation rises is what part 5 called over-fitting. Two cautions apply. Part 5's: the seen score does not show that a model recites its training text, and part 7 measures that. And one from the build log: it must not be compared between tokenizers, because the word-fragment pieces were themselves fitted to the training works. So the fair comparison is on the validation plays alone. After 68 passes the character model stood 0.04 above its own best score. After the same 68 passes the smallest vocabulary stood 0.09 above its best, and the largest 0.24. (Measured, one run each. Not quite like for like: the character run had finished cooling its learning rate, and the other two were four fifths and five eighths of the way through theirs.)

**A shorter run beat a longer one.** For the largest vocabulary, 2,500 steps scored *better* than 5,000 (1.8515 against 1.8733), even though the longer run passed through step 2,500 on its way (it stood at 1.90 there, already past its best) and we kept its best moment. One pair of runs could be luck: single runs wobble by about 0.006. But the 10,000-step pilot did no better (1.8759), and the three 2,500-step runs of the race itself scored 1.846 to 1.854. The likeliest reason is the one part 5 gave for its 68-pass run. The learning rate is stretched to fit the run. At step 1,500, where the long run had its best moment, its rate was still 0.00083 out of a peak of 0.001. The short run's was 0.00043 and falling. Over the next thousand steps the long run's seen score dropped from 1.43 to 1.22 while its validation score got worse. The short run cooled down, fitted the training text more slowly, and gained a little more on validation. That fits the logs. It is a reading, not a test: no run changed the schedule alone.

One admission, made in advance in the build log and repeated here. For bpe-1024 the rule stopped one doubling short of the best pilot, and that **arm** (part 5's word for one tokenizer and its runs) may carry a handicap. If the two pilots are taken at face value it is 0.008. They are single runs, so the true figure could be nothing or twice that. Its best moment came at 98% of the run, which is what an under-trained run looks like. An auditor watching the pilot noticed how thin the thread was: the 5,000-step run had two stops that tie to four decimal places, 1.8001 at step 2,875 and again at step 3,375, one on each side of the two-thirds line. Had the later one come out lowest, the rule would never have tried 2,500 steps, and this arm would have run at 5,000. The rule was written down first, so it stands. Remember the 0.008 when you see the margin.

## The rule

Written in the build log before the fifteen runs, and carried out by [scripts/summarise_runs.py](../scripts/summarise_runs.py):

- Each tokenizer is trained three times, with seeds 1, 2 and 3, at its own length, in 32-bit, with everything else at its default.
- Each run is judged at its best validation moment.
- The score is bits per character over all the validation text.
- The spread is the standard deviation over seeds, pooled over all five arms. Two arms differ only if their means are more than 1.82 pooled standard deviations apart. (Part 5 explained both halves. *Pooled* means one spread worked out from all fifteen runs, each measured against the mean of its own arm. 1.82 is what the ordinary 95% test comes to for five arms of three seeds.) With part 5's wobble of 0.006 that distance comes to about 0.011.
- The winner is the *smallest* vocabulary within that distance of the lowest mean.
- A run is thrown away only if its loss stops being a number. It is then made again with seed + 1000, and reported.
- All fifteen runs come from one commit of the code. Part 5 found that on this GPU an edit re-rolls a score as surely as a new seed, even an edit that changes no arithmetic. So the character runs were made again.

Reported beside it, and deciding nothing: the same test on everything except speaker-label lines (where part 5 found most of the noise), and the two **finalists** (characters, and whichever word-fragment vocabulary has the lowest mean) compared on the very same text. Both were auditors' proposals. A later auditor pointed out that the second had been promised and had no program, wrote a first working version, and listed the choices it would force, so that we could fix them before the runs and not after.

## What this race can and cannot show, said before the result

Here is something I had not seen done in a tutorial, and it is cheap. Before the fifteen runs, an auditor simulated the exact procedure above a few hundred thousand times, with a known truth planted in it. How often does the rule take the title away from characters when it should?

| If one word-fragment vocabulary is truly better than characters by | Characters lose the title |
|---|---|
| 0.03 bits per character | every time |
| 0.02 | 94 to 99.8% of the time |
| 0.01 | about half the time |
| 0.005 | one time in five |

(The ranges come from trying three sizes of wobble: 0.0045, 0.0054 and 0.0062.) Losing the title is not the same as the right vocabulary winning it. In the one case the auditor followed through, a vocabulary truly better by 0.02 was crowned nine times in ten. Truly better by 0.01, it was crowned one time in four, and a smaller neighbour that was no better took the title nearly as often.

So this race can see a difference of 0.02. It cannot see 0.01. "Characters win" will mean "nothing beat characters by more than about 0.011", not "characters are better". The burden of proof sits on the larger vocabulary, by design: a bigger table has to earn its place. One more lean had been named earlier still, in part 5: the recipe was tuned, by someone else, for a character model, and ties go to characters too.

(These words were not written blind. When the auditor wrote them the first pilot had finished, 0.06 behind part 5's character pilot, and the auditor could see it. By the time they went into the build log all six pilots had finished, 0.06 to 0.14 behind. They were written before any number that could decide.)

Two more things were written down then. The pilots that choose the lengths are single runs, so the length rule is itself a little lucky or unlucky, and it can cost an arm about 0.01. And the score is **exact for characters but only an upper bound for word fragments**: it can charge a word-fragment model a little too much, never too little. Such a model can spell the same text in more than one way. Our 1,024-piece tokenizer always writes the speaker label `BENVOLIO` as `B|EN|V|OL|IO`. The model could just as well produce `B|E|N|V|OL|IO`, or the word letter by letter, and the page would read the same. We charge it for the one spelling our tokenizer produces, and whatever probability it gave the others is lost. An auditor measured this on the first pilot by adding up every possible spelling of about 3,000 words and punctuation marks picked at random from the validation plays: about 0.008 bits per character (somewhere between 0.004 and 0.013). Four fifths of it came from just ten of them, every one a name the training works never contain. (One `BENVOLIO` alone was over-charged by 17 bits.) The measure stays as it was fixed, because it is the cost of the model as it will actually be used.

## What the auditors found this time

While the first pilot trained, three auditors attacked the plan. No blocker, and some good news worth having. A full-size pilot's score, worked out again on the CPU with the same scoring code, matched the GPU's to within 0.0000002. A scorer the auditor wrote from scratch, sharing no code with ours, agreed with ours on four small models. And the bookkeeping that turns a model's bits into bits per character gave the same answer whichever of the five tokenizers the numbers were pushed through. (The auditor fed it part 5's five-letter table: 2.2577 all five ways.) So the arithmetic of the measure treats every tokenizer alike. The slack above is a different matter: it belongs to the models, not to the arithmetic.

The programs around it were not ready. The script that makes the fifteen runs had only ever been dry-run, and:

- a run whose loss stopped being a number would have been *resumed* when the command was given again. In the auditors' CPU tests it blew up again every time, so the sweep could never get past it. On this GPU a resumed run is not an exact repeat, so it might have scraped through and been counted as if nothing had happened. The rule says discard it and re-run with another seed;
- a run made earlier was never checked against the length the plan asks for now, so one arm could have ended up with mixed lengths;
- the check that all runs share one commit looked only at runs that come earlier in the running order than the one about to be made. An auditor deleted the first run and made it again from a different commit. The fourteen later runs were never looked at, and the script said "all done".

I fixed all three. Then a helper agent wrote tests for the fixes, and the tests caught a bug in my fix: the script asked git where the code was *once*, at the start, so a commit landing in the middle of what I expected to be five hours of runs would have gone unnoticed. It asks before every run now. If you have read parts 3, 4 and 5 you know this story. It is the same one: code for the day things go wrong, tried only on days when they go right.

## The result

Fifteen runs, three and three quarter hours, all from one commit. None blew up, none was resumed.

| Tokenizer | Steps (passes) | Seed 1 | Seed 2 | Seed 3 | **Mean** | Spread |
|---|---|---|---|---|---|---|
| characters | 9,985 (34) | 1.7492 | 1.7571 | 1.7575 | **1.7546** | 0.0047 |
| bpe-1024 | 2,500 (21) | 1.7895 | 1.8006 | 1.8136 | **1.8012** | 0.0121 |
| bpe-1536 | 2,500 (23) | 1.8313 | 1.8219 | 1.8186 | **1.8240** | 0.0066 |
| bpe-2048 | 2,500 (24) | 1.8227 | 1.8291 | 1.8198 | **1.8239** | 0.0048 |
| bpe-4096 | 2,500 (27) | 1.8460 | 1.8539 | 1.8533 | **1.8511** | 0.0044 |

(All measured. The spread is the standard deviation of the three runs. Lower is better.)

**Characters win.** The pooled spread is 0.0071, so two arms differ beyond 0.0130. The nearest vocabulary, bpe-1024, is 0.0466 behind characters: more than three times that. Bigger vocabularies scored worse, with one tie: the two middle ones finished 0.0001 apart. Of the ten pairs of arms, that is the only pair the rule calls level.

For scale: in part 5 the 17-pass pilot finished 0.011 to 0.030 behind the nine 34-pass runs (0.020 behind their average), and the six-letter table is 0.48 behind the character model. So 0.047 is a real loss and a modest one. Every one of the fifteen models beats the six-letter table by 0.37 or more.

That is not what I expected when part 3 built four vocabularies and one character table. Part 3 quoted a rule of thumb from machine-translation research that pointed at a vocabulary of about 1,200 to 1,500 pieces for a text this size, and every large language model you have heard of uses word fragments.

Go back through what we wrote down beforehand, because this is what it was for:

- *The race can see 0.02 and cannot see 0.01.* The smallest gap is 0.047. This is not a close call.
- *bpe-1024 may carry a handicap from the length rule, 0.008 by the pilots' reckoning.* Two of its three runs did have their best moment at the very last step, the mark of a run that is too short. Against that, the three runs averaged 1.8012, level with the longer pilot's 1.8001. Give it the 0.008 anyway and it is still about 0.04 behind.
- *Word-fragment scores are an upper bound, slack about 0.008* (measured on one pilot, not on these runs). Give that back too: about 0.03 behind.
- *The test assumes every arm is equally noisy.* They were not quite. bpe-1024's three runs spread 0.012, the others 0.004 to 0.007. That pushed the threshold from the 0.011 we expected to 0.013. Does the verdict lean on the pooling? No. Judged only against its own spread and the characters', which is a harsher test with three runs each, bpe-1024 would have to come within about 0.026 of characters to tie. It is 0.047 away. What the harsher test does take away is bpe-1024's clear lead over the two middle vocabularies: 0.023 apart, about 0.025 needed.
- *The recipe was tuned for characters.* True, and untested. What is left after the give-backs is about 0.03. That is about what doubling the run length bought the character model in part 5 (0.011 to 0.030, depending on the run). A recipe tuned for word fragments might be worth that much, or might not. We do not know. The rule gives characters the title under *this* recipe, and that is all it was ever going to show.

None of those promises had to be bent after the fact, which is the point of making them.

What do the two finalists sound like? The opening of one sample each, seed 1 at its best moment, from the same prompt and the same dice as in part 5. I did not pick them.

Characters:

```
HAMLET.
The most brief may report all our loves; the court will be the
prisoner, and the fool be nothing but so weight in the villainy. I will infringe
to this action.
```

bpe-1024:

```
HAMLET.
What ho, Jack?

HORATIO.
I do.

HAMLET.
The letter is too late; the devil’s good.

HORATIO.
The antic poet.
```

Read aloud, I could not tell you which model scores better. (These two are 0.040 apart. The arms are 0.047 apart.) A sample is one roll of the dice, and the score is an average over all 274,727 characters of the two plays.

### Where the difference is

The secondary table, named in advance and deciding nothing, splits each score by play, and into speaker-label lines against everything else:

| | Everything | *All's Well* | *Romeo and Juliet* | Speaker-label lines | Everything else |
|---|---|---|---|---|---|
| characters | **1.7546** | **1.7080** | **1.7986** | 2.434 | **1.7117** |
| bpe-1024 | 1.8012 | 1.7286 | 1.8698 | **2.225** | 1.7745 |
| bpe-1536 | 1.8240 | 1.7471 | 1.8965 | 2.575 | 1.7764 |
| bpe-2048 | 1.8239 | 1.7564 | 1.8876 | 2.561 | 1.7772 |
| bpe-4096 | 1.8511 | 1.7798 | 1.9183 | 2.976 | 1.7799 |

(All measured, mean of three runs. Lower is better, and the best of each column is in bold.) Characters are ahead in both plays against every vocabulary.

**On ordinary text the four vocabularies cannot be told apart.** 1.7745 to 1.7799. In this column the pooled spread is 0.0036, so the same test calls two arms different beyond 0.0066, and the widest of the six pairs is 0.0054 apart. That is not proof that they are equal. With three runs each the test would only be sure to catch a gap of about 0.01, and the four means do line up in order of vocabulary size. If quadrupling the vocabulary costs anything on ordinary text, it is a twelfth of what separates all four from characters, who are 0.063 to 0.068 ahead.

**About nine tenths of the gaps among the vocabularies come from the names** (of the 0.050 between the smallest and the largest, 0.045 is speaker-label lines), and not quite the way part 3 predicted. This is the noisiest column in the post: bpe-1024's three runs scored 2.09, 2.23 and 2.35 on it. Put the same test to it (my sum afterwards, not part of the plan: pooled spread 0.083, so differences beyond 0.15 count) and two things stand. The largest vocabulary is clearly the worst at names, as part 3 said it would be. And the best of all five is bpe-1024, 0.21 ahead of characters. Hold that second one loosely. Part 5 showed that labels get worse the longer a model trains, and these arms trained for different lengths.

I had a guess ready for the names, and it mostly fails. The guess: at 1,536 pieces and above `HAMLET` is a single piece, so after a blank line the model bets on the whole names it knows, and `ROMEO` has to be spelled `RO|M|E|O` against that bet. A reviewer tested it and I measured it again. Whole names are rarer than I thought: 4 names are single pieces at 1,024 (`FALSTAFF`, `KING`, `DUKE`, `QUEEN`), 12 at 1,536, 33 at 2,048 and 151 at 4,096, heading 5%, 12%, 24% and 59% of the training speeches. The label scores are 2.22, 2.58, 2.56 and 2.98. Doubling the share of speeches they head, from 1,536 to 2,048, moved nothing. The big jump is between 1,024 and 1,536, where that share rises least, from 5% to 12%. Nor is it only the unknown names:

| Speaker labels, bits per character | Names the training works also use (915 labels) | Names they never use (858) |
|---|---|---|
| characters | 0.92 | 3.96 |
| bpe-1024 | 0.69 | 3.77 |
| bpe-1536 | 0.79 | 4.37 |
| bpe-2048 | 0.85 | 4.29 |
| bpe-4096 | 1.01 | 4.96 |

(Measured, mean of three runs.) The larger vocabularies are worse at names they *know* as well, and about half of bpe-1024's lead over characters is on known names. (Seed by seed that share runs from a third to all of it, so hold it loosely.) So "betting on whole names" may be part of the largest vocabulary's trouble. It does not explain the column.

The finalists, on the same text ([scripts/compare_finalists.py](../scripts/compare_finalists.py), every choice in it fixed beforehand). The gap is the 0.047 you already have. The new question is how much it depends on which passages the two validation plays happen to contain. The script cuts them into 56 blocks of 5,000 characters, makes up 10,000 new validation texts by drawing blocks at random, with repeats allowed, and works the gap out again on each. In 95% of them it lies between 0.036 and 0.059. As promised in advance, the caveat that goes with it: this holds the six trained models fixed. It measures the luck of the text, not the luck of training, and it cannot overturn the rule. It also only redraws the two plays we have, and they disagree about the size: characters are ahead by 0.021 in *All's Well* and by 0.071 in *Romeo and Juliet*. Characters are ahead in 80% of the blocks.

### An exploration, made after the verdict

Everything above was planned. This was not: I wanted to know *which words* the character model predicts better, so treat it as a description and not a test.

Comparing models that cut the text differently, word by word, has a trap in it, and I fell in. My first attempt shared each token's bits equally among its characters. It reported that word-fragment models are twice as bad as characters at *spaces* and better at every kind of word. That was nonsense: a word-fragment piece carries its leading space (` the`), so sharing its cost equally moved the cost of choosing the word onto the space. The fix is to use the one unit all five tokenizers share. Part 3's chunk rule cuts the text the same way for all of them (a word with its leading space, a run of punctuation, a line break), and no piece ever crosses a chunk's edge. So each model's bits can be added up exactly per chunk, and the chunks add up to the score.

| A word chunk, by how often the word occurs in the training works | Share of the text | Bits per word: characters | bpe-1024 | bpe-4096 |
|---|---|---|---|---|
| 1,000 times or more | 36% | 5.12 | 5.35 | 5.34 |
| 100 to 999 | 23% | 8.96 | 9.12 | 9.01 |
| 10 to 99 | 16% | 12.79 | 13.30 | 13.47 |
| 1 to 9 | 7% | 18.27 | 20.35 | 20.57 |
| never | 3% | 32.76 | 32.17 | 34.11 |

(Measured, mean of three runs per model. Fewer bits is better. The table leaves out punctuation, line breaks and speaker-label lines, about 13% of the text, which is why the shares do not add up to 100%.)

Characters are ahead in the first four rows. The gap is not a steady slope. Against bpe-1024 it is 4% on the commonest words, 2% in the next row, 4% in the next, and then 11% on words the training works use fewer than ten times. On words they never use, bpe-1024 is level, or a shade ahead. And common words are most of the text: in the units of the score, the top row alone accounts for 0.022 of bpe-1024's 0.047, and the rare-word row for 0.019. (On punctuation the vocabularies are slightly ahead.)

### Why? I do not know

The table rules out the first thing I reached for. My first guess was *scarce pieces*. A character model has 97 symbols to learn, and the 45 commonest, which make up 98% of the text, each turn up at least ten thousand times. A word-fragment model has a thousand or four thousand pieces to learn from a third to two fifths as many examples. At 4,096 pieces the middling piece turns up about a hundred times in all, and half the pieces fewer than that (measured).

But if scarce pieces were the trouble, the 4,096-piece model should be far worse on ordinary text than the 1,024-piece model, where the middling piece turns up 770 times and only one piece in twenty-five is that rare. It is 0.005 worse, which this race cannot tell from nothing. Rare words cost all four vocabularies the same 20.3 to 20.6 bits. And the largest single slice of bpe-1024's deficit is on the *commonest* words, such as ` the` and ` and`: single pieces it has met thousands of times.

So whatever costs 0.06 is something all four vocabularies share and characters do not. Candidates, none of them tested:

- **Fewer updates.** Every step is 64 windows of 256 tokens, so a word-fragment step holds two and a half to three times as many characters, and those models got a third to two fifths as many updates per pass through the text. They over-fit before they have taken many steps. In the longer pilots the best moment came at step 1,500 for the largest vocabulary and at about step 3,000 for the smallest, where the character pilots' came at about step 10,000.
- **The recipe.** Learning rate, dropout and batch were tuned, by someone else, for a character model. We changed nothing, for any arm.
- **Settings that count in steps fell unevenly.** The 100-step warm-up is 4% of a word-fragment run and 1% of the character run. Weight decay acts at every step, so it had four times as many steps to act on the character model.
- **The measure's slack**, about 0.008.

With a hundred times the text I would expect a different answer. That is a guess, and nothing here tests it. It is also not the main reason the models you have heard of use word fragments. They use them because a word-fragment model reads and writes the same text in a third to a quarter as many steps, and a window of the same size holds three to four times as much. At their scale that is most of the bill. Our character model pays it too. Every arm ran at 80,000 to 87,000 tokens a second (measured), so each character run took 35 minutes against 10, and GP-Thee will need two and a half times as many steps as bpe-1024 to write the same speech.

## What this does not show

- That characters are the better tokenizer in general. Only that nothing beat them here, under a recipe tuned by someone else for a character model, which we did not touch.
- That the word-fragment models were trained as well as they could be. Their lengths came from single pilot runs and a rule that stops at the shortest good length. A gentler learning rate, more dropout or a smaller batch might suit them better. Nor did we try a known remedy for this kind of over-fitting, which is to cut the training text a little differently every time the model reads it (it is called BPE-dropout). It would also train the model on the other spellings, which bears on the slack. We said beforehand that we would not tune, and we did not.
- Anything about other model sizes. One network, 11 million parameters.
- Anything about the test works, which are still unopened. Every choice so far, this one included, was made on the same two plays, and part 5 reported how much works differ: a standard deviation of 0.14 from one to the next.

## Where we are

**GP-Thee uses the character tokenizer**: 98 symbols, 10,757,760 parameters, 34 passes. The number 11M in its name was right all along.

New in the repo:

```
gp-thee/
├── scripts/find_length.py          the run-length rule of part 5, as a program
├── scripts/sweep.py                the fifteen runs, seed by seed, refusing to mix commits or lengths
├── scripts/compare_finalists.py    the two finalists on the same text
├── scripts/where_they_differ.py    the exploration: chunk by chunk, by how common the word is
├── tests/                          336 tests now; 75 of them are for find_length, sweep, summarise_runs and compare_finalists. The exploration script has none yet
├── docs/run_lengths.json           the lengths the rule found
├── docs/results-sweep.json         all fifteen runs and the verdict
├── docs/finalists.json             characters against bpe-1024 on the same text: the gap, its range, play by play
└── docs/where_they_differ.json     the exploration's table
```

```bash
uv run python scripts/find_length.py --tokenizer bpe-1024   # about an hour: the pilots the rule asks for
uv run python scripts/find_length.py --tokenizer bpe-4096   # another hour
uv run python scripts/sweep.py                              # just under four hours: fifteen runs. Commit nothing while it runs
for run in runs/sweep-*; do uv run python scripts/evaluate.py --run "$(basename "$run")"; done   # labels against everything else, per run
uv run python scripts/summarise_runs.py sweep-              # the verdict, and the same test on everything except speaker labels
uv run python scripts/compare_finalists.py
uv run python scripts/where_they_differ.py
```

Six parts in, the tokenizer is chosen, and with it the size of the model. (Which trained run carries the name is not chosen yet. That will need a rule of its own, written down first.) What the model has not yet had to do is answer for itself: does it recite its training works word for word, what does it write when asked properly, and how does it score on three works nobody has looked at? That is part 7.
