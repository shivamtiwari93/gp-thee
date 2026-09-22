# Building GP-Thee, part 8: the test we could only take once

*Three works have sat in a locked folder since part 2. This part opens them, once, to answer the only question seven parts were building toward: how good is GP-Thee-11M on Shakespeare it has never read? The answer is one number, and the ceremony to earn it honestly was larger than the number.*

[Part 7](07-does-it-recite.md) named the model — `sweep-char-seed-1`, GP-Thee-11M — and showed it does not recite its training text so much as reproduce its editors. One thing was still missing, and it is the thing the whole project was for. Every score in seven parts came from the works the model trained on, or from the two validation plays it was measured and selected against. Neither tells you what you actually want to know, which is how the model does on Shakespeare that took no part in building it.

For that you need text the model has never seen and no decision was ever made on. There are exactly three such works, and you can look at them exactly once.

## Why once

This is the part where the discipline of the whole series either means something or does not.

Three works — *The Life and Death of King John*, *The Tempest*, and *A Lover's Complaint* — were set aside in [part 2](02-the-data.md), before a single model was trained. Since then the code has physically refused to load them: `gp_thee.data.load_works("test")` raises rather than return them, unless you hand it a password. No script has scored them, no number in seven parts has come from them.

The reason you get one look is not superstition, it is arithmetic. The moment you measure something and let the result change what you do — pick a different checkpoint, tune a threshold, even just decide the number was "surprising" and look again — the works have taken part in a decision, and they are no longer held out. A held-out score is only held out until you use it. So the test works get spent the first time they are read, and everything you wanted from them has to come out of that single pass. Whatever the one run fails to capture is gone.

That constraint drove the whole design of `scripts/final_evaluation.py`, and it is worth seeing, because the machinery is the actual subject of this part. The number at the end is almost an afterthought by comparison.

## The one-way door

The script has three phases with a door in the middle, and the door only opens once.

**Before the door**, nothing is at stake. First a gate that refuses to start unless the release is committed, the working tree is clean, and the result file does not already exist — so a second run is impossible by construction, not by good intentions. Then a full dress rehearsal: every single thing the real measurement will do, run against the *validation* works instead of the test works. Every model re-scored, every baseline recomputed, every array written, the whole result serialised to JSON. If anything is going to crash or produce nonsense, it crashes here, on works we are allowed to look at, at no cost.

The rehearsal earns its keep in one specific way. It re-scores all twenty-one model checkpoints on the validation works and checks each against the number that checkpoint's own training log recorded during the sweep — and it demands the three runs behind the headline match **exactly, to all seventeen digits** (in the event, all twenty-one did). That sounds impossible for floating-point on a GPU, and [part 5](05-training.md) found that training the same model twice gives different numbers. But that non-determinism lives in the *backward* pass. Scoring is forward-only, and forward-only is bit-exact on this machine — I checked before writing the gate, and the rehearsal confirmed it twice:

```
sweep-char-seed-1/best 1.749198301521595 against 1.749198301521595  exact
sweep-char-seed-2/best 1.757111555111382 against 1.757111555111382  exact
...
the bar reproduces docs/baselines.json on all 11 rows
```

If a single digit were off, the pipeline would have drifted since the models were trained — a different torch, an edited data file, the wrong checkpoint — and the test number would be quietly meaningless. The exact match is the proof that the instrument reading the test works is the same instrument that read everything else.

**The door itself** is one line: the result file is created, holding a stub that says "in progress", *before* the test works are opened. If the run dies halfway, that stub is what remains — an honest record that the measurement was spent and did not finish, rather than a clean slate inviting a second try.

**After the door**, every measurement is wrapped so that a failure in one costs only that one, and the raw per-token numbers are written to disk the instant they exist, before any statistic is computed on them. The reasoning is the same throughout: a forward pass over the test works can never be redone, but arithmetic on a saved array can be redone forever. So the irreplaceable thing is captured first, and everything derived comes after.

### The measurement I would have shipped

I want to be honest about how close this came to going wrong, because it is the whole argument for the ceremony.

Before running it, I had four independent agents design the measurement and four more attack the design; then, with the script written but unrun, it was audited line by line, and three of those auditors independently found the same bug. My code computed the headline by looping over the per-work breakdown as though it were a dictionary. It is a list. The line would have raised an error — *after* the works were opened, and *before* the results were written. The test works would have been spent to produce a stub file and a traceback, and there would have been no second chance.

It was caught with the door still shut. But the lesson is not "I fixed a bug"; it is that the bug lived in a function the rehearsal never exercised, which is exactly why a shape error could have survived to the one run that mattered. So the rehearsal was extended to run *every* function the real measurement uses, including that one, on the validation works. A pre-registration you only follow where it is convenient is not one, and a rehearsal that skips the risky function is not a rehearsal.

Only after all of that did I type the password.

## The number

GP-Thee-11M, on the three works it had never read, scores **1.7958 bits per character**. On the two validation plays it was selected on, it scored 1.7492.

Higher is worse, so the model does measurably worse on the test works — but they are not the same kind of text, and that is most of the story:

| | bits per character |
|---|---|
| *The Life and Death of King John* — a history | 1.6921 |
| *The Tempest* — a late romance | 1.9044 |
| *A Lover's Complaint* — a narrative poem | 1.9240 |
| **all three together** | **1.7958** |

Of the three, the history is the easiest and the poem the hardest — the history scores below the model's whole validation figure, the poem well above it. The gap between the easiest and hardest of these three works is 0.23 bits per character — which is larger than every difference between models the entire project ever argued about. Whatever "how good is the model" means, it means less than "which play is it reading". Leave any one of the three works out and the combined figure swings between 1.72 and 1.91. That is the real width of a three-work sample, and it is worth keeping in view whenever a single decimal is quoted as *the* score.

For scale, here is what the number is good *against*, on this exact text, measured the way [part 5](05-training.md) measured it on validation — simple predictors that never saw a neural network. The n-grams are fitted on the training works; the compressors are shown squeezing the test text cold, having read nothing:

| predictor | bits per character on the test works |
|---|---|
| blind guess (1 of 97 characters) | 6.60 |
| character frequencies | 4.79 |
| 6-gram (the best n-gram) | 2.32 |
| bzip2 | 2.58 |
| xz | 2.91 |
| **GP-Thee-11M** | **1.80** |

The model beats every one of them, by at least 0.52 bits — a wider margin than the 0.48 it managed on validation. Even the fairer comparison, a compressor that reads all the training works *first* before squeezing the test text, only reaches 2.37, still more than half a bit behind. This was [the project's step 9](../docs/PLAN.md) success criterion, written down at the very start: *the model beats the n-grams and the compressors*. It is now met on the works that decided nothing.

## The four things I wrote down before looking

The number is one line. The reason the run computes a dozen other things is that four questions already in print could be *changed* by what the test works said, and the only way to keep that honest is to fix both answers to each in advance — so that neither could be narrated after the fact. All four were written into the build log before the door opened. Here is how they came out.

**Does it recite? — confirmed, on a clean control.** Part 7's headline was that the model reproduces its editors' furniture, not Shakespeare's verse: the longest passage of the poet's own words it could be made to write back was 44 characters from the works it trained on, against 0 from the validation plays. But part 7 admitted the validation zero was suspect, because those plays were *chosen* for sharing no long passage with the training works. The test works were not chosen that way. So this was the real test of part 7's claim — and on the three works it never read, the model reproduces **nothing**: zero runs of even 40 agreed characters, where the training works had fifty-two. The verdict holds on a control that was not rigged to produce it. (The test works do share 26 overlapping fifty-character windows with the training works — real editorial overlap, essentially a single ~75-character passage, the accepted *King John* / *Winter's Tale* scene heading. The model reproduces none of it either.)

**Which run is best? — the released one, again.** The release was picked as the best of three eligible runs on validation. On the test works the three score 1.7958, 1.7971, and 1.8046 — and the released run is again the best of them. That is worth exactly one sentence: it is one draw of the dice agreeing with another, over a spread of 0.0088 that the project's own threshold (0.0130) calls a tie. The released model is what a written-down rule selected, not a model demonstrated to be the best; I would have said the same thing had it come last.

**Do characters still beat word-fragments? — yes, on new kinds of text.** [Part 6](06-which-tokenizer.md) decided single characters beat every word-fragment vocabulary, but it decided that on two plays. The test set adds a history, a romance and a poem, and the character model beats every fragment model on them too — 1.80 against the best fragment model's 1.89. The conclusion survives text it was not decided on.

**What did all that selecting actually buy?** Two layers of selection sit on the validation score — the best checkpoint within the run, and the best run of three. Part 7 priced the between-run choice by simulation (best of three flatters a score by about 0.005, best of eleven by about 0.010) and the within-run best-of-stops layer by direct measurement (about 0.006 to 0.008). The test works let me look directly. On validation the released run sits 0.0054 below the three-run mean; on test, 0.0034. The edge that selection gave the number does not fully carry to text that chose nothing — which is exactly what "the number is a little optimistic" is supposed to mean, now measured rather than simulated.

Two small honesty checks rode along on the same single pass. Re-scored on the CPU instead of the GPU, the headline moves by less than one part in a hundred million. And the scoring window, the one knob nobody tunes, moves it by about 0.05 at its extremes — held fixed everywhere, so it biases no comparison. Both are in the record; neither changes anything.

## What the number is, and what it is not

GP-Thee-11M predicts a held-out page of Shakespeare at 1.80 bits per character. A fair coin per character would be 1 bit; the raw alphabet is 6.6; a good general-purpose compressor is 2.6. So the model has learned a great deal about how Shakespeare's characters follow one another — enough to halve the compressor's surprise above that one-bit floor — from five megabytes of text and eleven million parameters, with no pretraining, no outside data, and a tokenizer, model and training loop all built from nothing across this series.

It is also, cheerfully, not much of anything else. It cannot follow an instruction, it has never been asked a question, and it reproduces not one line of the verse it was trained on. It is an in-character autocomplete for a universe that contains only Shakespeare, and now we know precisely how well it autocompletes: 1.80 bits per character, on a history, a romance and a poem it had never met, beating every predictor that came before it and reciting none of them.

That was the question. The works are spent. There is no number after this one, which is the whole reason it took a locked folder and a one-way door to earn.

## Where this leaves the project

Eight parts, and GP-Thee is complete: a corpus cleaned and audited, a split frozen before any training, a tokenizer chosen by a rule, a model and training loop built and checked, a recipe nobody tuned, eleven runs, one of them named and released, a direct answer to whether it is a parrot, and now a single honest score on text it never saw. Every number in the series was measured before it was written, and the ones that were wrong — and several were — were caught and corrected in the open.

The model itself, GP-Thee-11M, is on Hugging Face; the entire project, including every training log, every score, and the per-character arrays behind this final number, is on GitHub. It was built for no reason other than to see whether it could be done honestly, start to finish, on a laptop. It could.

*The complete build log, with all twenty entries and every measurement, is in [docs/BUILD_LOG.md](../docs/BUILD_LOG.md). The final evaluation's full record, including the per-token surprise of every model on every test work, is in [docs/final-evaluation.json](../docs/final-evaluation.json) and [docs/final-evaluation/](../docs/final-evaluation/).*
