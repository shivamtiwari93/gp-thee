# Building GP-Thee, part 7: does it recite, can it speak, and which one do we release?

*A model trained thirty-four times over five megabytes of Shakespeare: has it learned the plays, or learned them by heart? This part measures that, builds the sampler that lets you ask the model things, and picks the one run of eleven that gets the name. The test works are still unopened.*

[Part 6](06-which-tokenizer.md) settled the last design question: GP-Thee reads and writes single characters. Eleven trained runs of that design now sit in the repository, and one of them has to become the thing people download. Before that can happen, the model has to answer the question anyone sensible asks about a model trained on one small library thirty-four times over.

**Does it just remember the text?**

The same note on credit as before. I wrote the code; most of what was wrong with it, and with my reasoning, was found by AI agents asked to attack it. In this part one of them caught the worst methodological mistake of the whole project, and it was mine.

## The measurement I nearly published, and why it was worthless

Part 2 held out two whole plays, *All's Well That Ends Well* and *Romeo and Juliet*, that no model has ever trained on. So here is an obvious way to find out whether copying means anything: measure how much of those two plays already appears, word for word, in the other thirty-nine works. That is the rate an innocent writer of Shakespeare would show, just from Shakespeare's habits and his editors' formulas.

I measured it. **Not one fifty-character passage of the two validation plays occurs anywhere in the thirty-nine training works.** The longest shared run in either direction is forty-seven characters, and it is not even verse. Here it is with its line breaks written as `\n`, which is how every passage in this part is shown, because the full stop it opens with and the space it closes with both count:

```
.\n\n[_Exeunt._]\n\nSCENE III. The same. A room in 
```

A beautiful result: the innocent rate at fifty characters is zero, so any fifty-character match from the model is real copying. I had the paragraph written.

It is circular, and worthless. Part 2's own splitting script **requires** this of the held-out works: it looks for any fifty-character run a held-out work shares with any other work, and refuses to write the split unless the only ones it finds are ones already examined and accepted by name. The five held-out works passed on the first try and the lists were never revised — but had they not passed, they would have been. The validation plays score zero because scoring zero was a condition of their being the validation plays, not a discovery about them. A reader could not possibly have known that. I did know it, and did not think of it, until a reviewer did.

(One overlap *was* found and accepted, and it is not in a validation play: sixty-three characters shared between *King John* and *The Winter's Tale*, an editor's scene heading. Hold that thought until part 8.)

The honest baseline uses works that were selected for nothing: **each of the thirty-nine training works against the other thirty-eight.**

Fifty characters is the unit, but a single width tells you nothing about how close anything came, so here is the whole curve — the share of windows of each length that also occur in one of the other works.

| Run length | All thirty-nine works | Without the five that reprint one another | The poet's own words only |
|---|---|---|---|
| 20 characters | 0.011287 | 0.011228 | — |
| 25 | 0.003673 | 0.003428 | — |
| 30 | 0.001687 | 0.001416 | 0.0000217 |
| 40 | 0.000457 | 0.000247 | 0.0000197 |
| **50** | **0.000190** | **0.000046** | **0.0000068** |
| 60 | 0.000100 | 0.000006 | 0 |
| 80 | 0.000046 | 0 | 0 |
| 100 | 0.000015 | 0 | 0 |

Read down the first column and you can watch Shakespeare's habits run out. One window in ninety at twenty characters, one in five thousand at fifty, one in sixty-five thousand at a hundred. The last column is empty at twenty and twenty-five characters for a dull reason rather than an interesting one: a window has to carry twenty-five letters to count as the poet's, and a window of twenty-five characters cannot, because some of them are spaces.

(*Measured* three times now: by the reviewer who designed it, by me, and a third time by a different program written to produce this curve — one that hashes every window and confirms each hit against the text, against one that indexes every window by the text itself. All three agree at fifty characters to the digit: 916 shared windows of 4,809,425, and 202 of 4,425,256 once the five reprinting works are set aside.) The five exceptions are real: *The Passionate Pilgrim* reprints sonnets that also appear in *The Sonnets* and *Love's Labour's Lost*, and *Venus and Adonis* and *The Rape of Lucrece* share their dedication to Henry Wriothesley. Take those away and what is left is mostly the editor, not the poet: "Dramatis Personæ" headers, "ACT I / SCENE I. London. An ante-chamber", and the like.

So the innocent rate is about one window in twenty thousand, and nearly all of it is furniture: of the 202 shared windows left once those five works are set aside, 30 are Shakespeare's own words. That is the bar.

## What counts as reciting

Three decisions, all made before the model was measured and written into [the build log](../docs/BUILD_LOG.md):

**A copy is raw characters, and the whole curve is reported.** Fifty characters is the unit part 2 promised, but fifty on its own is a near-certain zero for the model and for an innocent writer alike, so it cannot show how close either came. The counts at twenty, twenty-five, thirty, forty, fifty, sixty, eighty and a hundred characters are all published together, and so is the longest match.

**The poet and the editor are counted separately, and neither is a footnote.** Most of what a Shakespeare text file contains was not written by Shakespeare: cast lists, act and scene headings, entrances and exits, and the speaker label before every single line. That text is short, formulaic and repeated across works, which makes it the easiest thing in the corpus to reproduce and the least interesting. About a tenth of the corpus is editorial furniture by the rule I published (*measured*: 10.3% of the training works, 10.1% of the validation plays, which is a reassuring symmetry). A passage also has to carry at least twenty-five letters to count, or a run of indentation scores as a copy.

This split is not a technicality. It decides the headline. With the results in front of you it would be possible to write "it reproduces sixty-six characters of its training text word for word" and equally possible to write "the longest complete line of Shakespeare it reproduces is eighteen characters", and both would be true.

**The verdict was fixed in advance,** so it could not be chosen afterwards:

> The model **recites** if the longest confirmed passage of Shakespeare's own words it reproduces from the training works reaches fifty characters, and is at least twice the longest such passage it reproduces from the two plays it never read. It **does not recite** if that figure is under fifty characters and exceeds the validation figure by no more than ten. Anything between is reported as it stands, with every passage printed.

One honest caveat about this pre-registration, which is weaker than parts 5 and 6. The reviewer who designed the measurement prototyped it at about a tenth of full scale, so model numbers existed before the rule was written, and I had seen them. The rule is still committed before the full measurement, and the build log says all of this. But nobody should read it as blind.

## Asking the model, properly

Sampling is the wrong instrument for this question, and it is worth saying why before we get to the right one.

When you ask a model for text and then look for copied passages, you learn what the model *happens* to say. It can know a passage perfectly well and never volunteer it. The right question is what it can be *made* to say, and that needs a different approach: walk the model along the true training text and ask, at every one of 4.8 million positions, whether the character that really comes next is the one it would have written. A run of such positions is a **candidate**.

Then comes the step that an earlier version of this skipped, and which changed the answer by a third. A candidate is not yet an extraction. The screen judges each position with somewhere between 128 and 255 characters of run-up, because that is how the scoring windows overlap; a model actually generating has the full 256. That is enough to change its mind. So every long candidate is **confirmed**: from the real 256 characters before it, the model is made to write, with no dice at all, and we count how many characters it gets right before it diverges. Only confirmed lengths are ever reported, and the difference is not cosmetic. *Measured*, over the training works: fourteen candidates reach fifty characters and nine of them survive being written out. The longest candidate of all runs to ninety-nine characters — and collapses to twenty-seven. The sixty-six characters that end up being the headline come from a different candidate, one of exactly sixty-six, which the model types back without a slip.

## The answer

Here is the scan, over every character of both texts.

| | The 39 works it trained on | The 2 plays it never read |
|---|---|---|
| Its first guess is the right character | 69.72% of the time | 63.63% |
| Candidate runs of 40 characters or more | 52 | 1 |
| Runs of 40 or more that survived being written out | 34 | 1 |
| Longest, once it had to write it out | **66 characters** | 40 |
| Runs of 50 characters or more, confirmed | 9 | 0 |

Sixty-six characters is a real extraction: hand the model the 256 characters that precede it and it will type all sixty-six back, with no dice, exactly. Here it is, and here are the other eight, with line breaks shown as \n so that each stays on one line:

```
 66  IRD PART OF KING HENRY THE SIXTH\n\n\n\n\nDramatis Personæ\n\nKING HENRY 
 59  exandria. A Room in the Palace.\n\nEnter Antony and Cleopatra
 58  _Exeunt Antipholus of Syracuse and Dromio of Syracuse._]\n\n
 57  andarus’ house.\n\nEnter Pandarus and Cressida.\n\nPANDARUS.\n
 56  \nEnter Antipholus of Syracuse.\n\nANTIPHOLUS OF SYRACUSE.\n
 54  _]\n\nSCENE III. The same. A Room in the Palace.\n\nEnter 
 51  ntipholus of Syracuse.\n\nANTIPHOLUS OF SYRACUSE.\nThe
 51  \n\nSECOND FRIEND.\nA thousand pieces!\n\nFIRST FRIEND.\n
 50  COND\n\n\n\n\nDramatis Personæ\n\nKING RICHARD THE SECOND
```

A title page. A cast list. Scene headings. Entrances. Speaker labels. **Not one line of verse.** *Measured*, character by character: seven of the nine carry nothing of Shakespeare's at all except the blank lines between speeches. The seventh carries three letters, the word `The` that begins a speech it never finishes. The eighth carries one short line, `A thousand pieces!` — eighteen characters, and the only complete line of his own that this model has ever been shown to reproduce.

Go looking for the longest passage that is mostly the poet's, and you get forty-four characters, which turn out to span two different speeches:

```
or he hath done me wrong.

KING HENRY.
What 
```

Thirty-two of those characters are Shakespeare's: the end of one line, and the first word of the next. The twelve in the middle are a speaker label and its line break. **The longest unbroken stretch of Shakespeare's verse this model reproduces anywhere in five megabytes is twenty-five characters long, and it is "or he hath done me wrong."**

For scale, the innocent baseline was one window in twenty thousand. The model has clearly learned something about its training works that it did not learn about the two plays it never read: 69.72% against 63.63% is six points, and six points is a lot. But when you open up what is inside those six points, it is cast lists and scene headings.

**GP-Thee has not memorised Shakespeare. It has memorised his editors.**

That makes sense once you see it. The training works alone hold 730 lines beginning `SCENE`, and "Dramatis Personæ" heads thirty-six of the forty-one works this project may look at. A speaker label is the most predictable line on any page of a play. Those are the passages a model with 10.8 million parameters and thirty-four passes can afford to store exactly. A pentameter line is not.

### What the rule said, and why I am not giving you a clean yes or no

The verdict was fixed in advance, and it lands between its two answers. Forty-four characters is under fifty, so not "it recites". But forty-four exceeds the validation figure of zero by more than ten, so not "it does not recite" either. The rule says: report it as it stands, with every passage printed. That is what the table above is.

I could tell you the honest summary is "no, it does not recite", and I believe that is right. But the rule I wrote *before* the measurement does not quite say so, and the reason is instructive: the validation ceiling came out at exactly zero, and my second clause asked the training figure to come within ten of it. Almost nothing could have satisfied that. A rule written after seeing the zero would have been written differently. This one was not, so it stands, and you get the numbers instead of a verdict.

### The bug that flattered the model

The first time this ran, it reported five confirmed passages as "mostly the poet's". The rule requires me to look at every long passage by hand, so I did:

```
um and colours. Enter King Henry, Gloucester     ("Drum and colours. Enter King Henry...")
LOUCESTER, brother to the King.\nDUKE OF          (a cast list)
UCIUS, servant of Timon’s creditors               (a cast list)
BER, Conspirator against Caesar.\nC               (a cast list)
```

Four of the five were the editor's furniture, counted as Shakespeare's. Two faults, both mine. My rule said the front matter runs "up to the first heading", and I had put `Dramatis Person` in the list of headings, so the cut-off landed on the cast list's own title and left the entire cast list unmarked. And my list of words that begin a stage direction did not include `Drum`.

Both errors pushed the same way: they made the model look as though it were reproducing Shakespeare when it was reproducing his editors. That is the direction I would have liked the answer to go, which is exactly why I should not be the last check on it. The rule is fixed, the four passages above are named in a test so they cannot come back, and the whole measurement was run again from scratch.

A second reviewer, reading this draft, caught a quieter failure of the same kind. The rule promised the whole run-length curve and my code asked for two of the eight lengths, so five of the numbers this post claimed to publish did not exist anywhere. Nobody would have noticed; the two published lengths told the same story. That is precisely what makes it worth naming, because a pre-registration you keep only where it is convenient is not one. The curve above, and the grid below, are the full eight.

### Sampling is the weaker instrument, and here is the proof

Asking the model for text and looking for copies is the obvious way to do this, and it would have badly understated the answer.

**Temperature** is how closely the dice follow the model's own odds: at 1.0 they follow them exactly, below 1.0 the model plays safe, and at zero there are no dice at all and it takes its likeliest character every time — which is what the confirmation step does, and what "greedily" means below. The grid ran all four of 0, 0.5, 0.8 and 1.0, against four kinds of prompt.

Prompted with its own training text at temperature 0.8, the model produced **one** copied passage of fifty characters in twenty thousand characters of output. The scan finds thirty-four places in the corpus where the model can be *made* to write forty characters or more, and nine where it writes fifty or more. (Fifty-two candidate runs reach forty characters; eighteen of them do not survive being written out.)

Every copy of fifty characters or more that sampling did find is, again, a scene heading. The longest was fifty-six characters, written greedily from a validation prompt:

```
t._]\n\nSCENE III. The same. A Room in the Palace.\n\nEnter 
```

Still, the grid is worth having for one thing the scan cannot give you: what the model copies when nobody is trying to make it copy. Three deviations have to be stated first. The written rule asked for 200,000 characters in each of the sixteen cells. The run generated 20,000 in eight blocks per cell; the seven separator line breaks make 20,007 measured characters per cell, or 320,112 altogether. The smaller grid took 34 minutes 24 seconds, including five minutes when the test suite was running beside it, so the promised size would have taken roughly five to six hours. The grid is therefore one tenth of the size fixed in advance. Its suite cells used the first eight of the ten fixed prompts, omitting the instruction and blank-line prompts, and its training and validation excerpts began at positions from a fixed seeded draw rather than the promised arithmetic placement. The positions were still reproducible and not hand-picked. The exhaustive scan above — the primary instrument — is unaffected: it still examined every position in all 4.8 million training characters and both validation plays.

Here is the whole sampled curve against the innocent baseline from the top of this post. The last column is what a writer copying at the innocent rate would have produced in the same amount of text. It is a yardstick, not a null hypothesis: the baseline was measured on Shakespeare's real text, and this is not Shakespeare's real text.

| Run length | Windows the model copied | Its rate | An innocent writer's rate | Expected at that rate |
|---|---|---|---|---|
| 20 characters | 11,766 | 0.036791 | 0.011228 | 3,591 |
| 25 | 3,080 | 0.009633 | 0.003428 | 1,096 |
| 30 | 1,037 | 0.003244 | 0.001416 | 453 |
| 40 | 136 | 0.000426 | 0.000247 | 79 |
| **50** | **14** | **0.000044** | **0.000046** | **14.6** |
| 60 | 0 | 0 | 0.000006 | 1.9 |
| 80 and 100 | 0 | 0 | 0 | 0 |

At twenty characters the model copies more than three times as much as an innocent writer, which is not a scandal — it is what a language model trained on this text is *for*, and it is the reassurance that the instrument is not simply printing zeros. Then watch the two columns converge. By fifty characters they have met: fourteen copies observed against fourteen and a half expected. Past sixty the model is at zero where Shakespeare's own works are still repeating each other.

And the column that is not in the table, because it is one number: **of those 320,112 characters, the count of windows that are Shakespeare's own words copied verbatim is zero, at every length.** At thirty and forty characters the innocent rate would have produced about seven and about six.

There is a sensitivity check under all of this, which the rule also asked for: the same counts after lowercasing everything and reducing every run of whitespace to a single space, so that a copy retyped with different layout still counts. Pooled, that gives six copies of fifty characters against fourteen raw. It cuts both ways, which is why it is worth having — normalising collapses the line breaks, so a fifty-character normalised window covers more words than a raw one and is harder to match, and the scene headings the raw count finds are mostly line breaks.



## Meanwhile: a sampler you can actually use

The model has had a sampler since part 5, but a minimal one, built to watch a run: same prompt every time, one temperature, no way to ask it anything. Part 7 adds the real one, [src/gp_thee/sampling.py](../src/gp_thee/sampling.py) and [scripts/sample.py](../scripts/sample.py).

The interesting part is not the sampling. It is what happens to your prompt before the model sees it.

**This universe has ninety-seven characters in it.** Your keyboard does not. The apostrophe in `I'll` is a straight one; Shakespeare's text uses `’`. Your quotes are straight; the corpus uses `“` and `”`. There may be a tab in there, or a non-breaking space, or an en dash where the corpus only has an em dash. And there may be a `ñ`, which this universe simply does not contain.

So a prompt passes through a normaliser first, which does one of three things to every character:

- leaves it alone, if the corpus already uses it;
- replaces it with what this corpus uses instead, and says so;
- **refuses the whole prompt and names the character**, with its codepoint and the nearest plain letter if there is one.

```
$ uv run python scripts/sample.py --run sweep-char-seed-1 --prompt "naïve"
that prompt cannot be spelled in this universe.
'ï' (latin small letter i with diaeresis, U+00EF) is not one of the 97 characters in the
complete works, so the model has never seen it and cannot be asked about it. The nearest
letter this universe has is 'i'.
```

Refusing is the point. Silently dropping the character would leave you wondering what the model was really given, and a project that has spent six parts insisting on measurement should not start guessing on the last mile. Every substitution is reported beside the sample, so the text you read is always the text the model saw.

Two rules that are not about spelling. **Trailing spaces are stripped.** That rule was written in [part 3](03-the-tokenizer.md) for the word-fragment tokenizers, where a space is glued to the word after it, so a prompt ending in one asks the model to continue a word that has not started. For the character model that won part 6 a space is an ordinary token of its own — *measured*: 740,476 of the 4,811,336 training characters, one in six and a half — and asks for nothing unusual. The rule is kept anyway, so that every checkpoint in this project is prompted the same way, and the strip is reported beside the sample like any other change. And **START is never prepended**. START means "a new work begins here"; an ordinary prompt is the middle of one.

### The ten prompts

The same ten prompts now run against every checkpoint, so two models can be set side by side. They are fixed for the rest of the project. Here is the released model on four of them, at the published default of temperature 0.8.

Give it a speaker label and it writes a play:

```
HAMLET.
The most brief may report all our loves; the court will be the
prisoner, and the fool be nothing but so weight in the villainy. I will infringe
to this action.

OSRIC.
Sir, I am in my called the likeness of your injuries for the
peril of your eye.
```

Give it a sonnet number that does not exist, and it writes a sonnet:

```
                    155

When first I looked upon thy task,
Thou look’st so deep as foul as majesty,
Had thou first died with stubborn tongue of blood,
Thy tale, silver and milk-venom stol’n,
```

Give it a modern sentence and watch it flee to the sixteenth century within one clause:

```
The meeting is at nine o’clock, and the budget him
without a slave?

SERVANT.
I thank you.
```

And ask it to follow an instruction, which it cannot do, because nothing in its universe has ever asked anyone to do anything:

```
Write me a poem about a cat.

HOTSPUR.
I pray you, mark how the catalogue
May have had broke our heads and have not torn.
```

It knows cats perfectly well — "The cat will mew, and dog will have his day" is in there, and twenty-nine more besides — but asked for a poem about one it reached for `catalogue` and got on with the scene. That is the whole model in four lines: it is an in-character autocomplete, not an assistant, and it has never once been asked a question.

### Speaking as a character

The nearest thing to a conversation this model can hold is a scene, so that is exactly how the wrapper builds it: the transcript is a page of a play and nothing else. Each turn adds a blank line, the speaker's name in capitals, a full stop, and their line. Then the model writes the next speech and stops at the blank line where the next speaker would begin.

```
ROMEO.
But soft, what light through yonder window breaks?

JULIET.
I have a headache and no money.

ROMAN.
No, but ’tis a gentleman that we are all advanced; and what would you
have said?
```

Nobody cast ROMAN. With no answerer named, the model writes the speaker label itself and we report who it picked. Headaches and lack of money both occur in the corpus; what the model failed to do was treat Juliet's sentence as a complaint that called for a relevant answer. It knew only that a line of dialogue was owed and roughly what one sounds like.

## Which run becomes GP-Thee-11M

Eleven character runs at thirty-four passes exist, and one of them has to be the model people download. This is the last and easiest place in the whole project to flatter the result, because the highest-scoring run of the eleven is [the pilot part 5 showed was a lucky draw](05-training.md).

So the rule was written down first and applied by a program, [scripts/choose_release.py](../scripts/choose_release.py):

- **Eligible** means made by the commit that decided the tokenizer, with a clean tree, one of the pre-registered seeds 1, 2 and 3, 32-bit, every other setting at its default. Three runs qualify, and they are the three the published comparison used.
- The released run is the eligible run with the **lowest validation score**, as its own result file recorded it during training.

| Eligible | Seed | Validation | |
|---|---|---|---|
| sweep-char-seed-1 | 1 | 1.7492 | **GP-Thee-11M** |
| sweep-char-seed-2 | 2 | 1.7571 | |
| sweep-char-seed-3 | 3 | 1.7575 | |

Eight runs are excluded, each with its reason recorded, including the 1.7404 that a different commit and a lucky seed produced.

**And now the part that matters more than the choice.** Two layers of selection sit on that 1.7492, and neither sits on the test score that part 8 will report. Within the run, the best of forty-one stops was kept. Between runs, the best of three was kept. Simulated with this project's own run-to-run spread, picking the best of three flatters a score by about 0.005, and the best of eleven by about 0.010. (*Measured* against both spreads this series has published, 0.0054 from the seed study and 0.0071 from the tokenizer sweep: 0.0046 to 0.0060 for the best of three, 0.0086 to 0.0112 for the best of eleven.) The three eligible runs are 0.0083 apart, which [part 6's own test](06-which-tokenizer.md) calls a tie.

So the release buys nothing this project can demonstrate, and the model card will say so. What part 8 reports will be two numbers, not one: the released model's own test score, and the mean and spread of all three eligible runs. The first is the artifact. The second is the recipe.

## A promise withdrawn

[Part 2](02-the-data.md) said the released model would be retrained at the end on all forty-four works, so that users would get a model that had read everything. That promise is withdrawn, and the reason is the whole argument of this series.

A model trained on all forty-four works can never be evaluated honestly, because the three test works would be inside it. It would be the one model in the project whose quality is an expectation rather than a measurement. Making that the headline artifact, on the last page of a series that spent six parts insisting on the difference, would be a poor joke. The gain would be 10.6% more training text (*measured*: 507,888 characters on top of 4,811,336 — the two validation plays as well as the three test works, since a forty-four-work model swallows both). What that buys is unknown, and unknowable by this project's own rules: the only works that could measure it are the works it would have eaten.

**GP-Thee-11M is the thirty-nine-work model, with a real test score.** If a forty-four-work version is ever made it will be a separate download with a different name, and its card will begin by saying it has no held-out score and never can have one.

## Where we are

New in the repo:

```
gp-thee/
├── src/gp_thee/sampling.py         the prompt normaliser, the sampler, the conversation wrapper, the ten prompts
├── src/gp_thee/memorisation.py     finding a copy, telling the poet from the editor, and the scan
├── scripts/sample.py               ask the model for text, or run the ten prompts against a checkpoint
├── scripts/memorisation.py         does it recite? writes docs/memorisation.json
├── scripts/baseline_curve.py       what an innocent writer of Shakespeare scores, at every length
├── scripts/choose_release.py       which run is GP-Thee-11M, by the rule; writes docs/release.json
├── docs/release.json               the released run, the eight excluded ones, and what the choice cost
├── docs/memorisation.json          every confirmed passage, the innocent baseline, the sampled grid
├── docs/baseline-curve.json        the innocent baseline at all eight lengths, work by work
├── docs/sampled-grid-*.txt         every character the model wrote for the grid, so you can re-count it
└── docs/candidates.json            every candidate of 50 characters or more, and what it confirmed to
```

```bash
uv run python scripts/sample.py --run sweep-char-seed-1 --prompt $'\n\nHAMLET.\n'
uv run python scripts/sample.py --run sweep-char-seed-1 --suite     # the ten fixed prompts
uv run python scripts/sample.py --run sweep-char-seed-1 --scene     # speak as a character
uv run python scripts/choose_release.py
uv run python scripts/baseline_curve.py                             # under a minute
uv run python scripts/memorisation.py                               # about forty minutes
```

Seven parts in, GP-Thee is finished: a tokenizer chosen by a rule, a recipe nobody tuned, eleven runs, one of them named, and a direct answer to the question of whether it is a parrot. What it has never been asked to do is read anything outside the thirty-nine works it trained on and the two it was measured against.

Three works have been out of reach since part 2. *King John*, *The Tempest*, *A Lover's Complaint*. No model has read a character of them, no score in seven parts has come from them, and the code refuses to load them without a password that appears in exactly one place. What this series has published about them is their size in the split table, and nothing else. In part 8 we type it, once.
