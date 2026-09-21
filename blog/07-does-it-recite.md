# Building GP-Thee, part 7: does it recite, can it speak, and which one do we release?

*A model trained thirty-four times over five megabytes of Shakespeare: has it learned the plays, or learned them by heart? This part measures that, builds the sampler that lets you ask the model things, and picks the one run of eleven that gets the name. The test works are still unopened.*

[Part 6](06-which-tokenizer.md) settled the last design question: GP-Thee reads and writes single characters. Eleven trained runs of that design now sit in the repository, and one of them has to become the thing people download. Before that can happen, the model has to answer the question anyone sensible asks about a model trained on one small library thirty-four times over.

**Does it just remember the text?**

The same note on credit as before. I wrote the code; most of what was wrong with it, and with my reasoning, was found by AI agents asked to attack it. In this part one of them caught the worst methodological mistake of the whole project, and it was mine.

## The measurement I nearly published, and why it was worthless

Part 2 held out two whole plays, *All's Well That Ends Well* and *Romeo and Juliet*, that no model has ever trained on. So here is an obvious way to find out whether copying means anything: measure how much of those two plays already appears, word for word, in the other thirty-nine works. That is the rate an innocent writer of Shakespeare would show, just from Shakespeare's habits and his editors' formulas.

I measured it. **Not one fifty-character passage of the two validation plays occurs anywhere in the thirty-nine training works.** The longest shared run in either direction is forty-seven characters, and it is not even verse:

```
[_Exeunt._]

SCENE III. The same. A room in
```

A beautiful result: the innocent rate at fifty characters is zero, so any fifty-character match from the model is real copying. I had the paragraph written.

It is circular, and worthless. Part 2's own splitting script **chose** the held-out works by exactly this test: for each candidate work, look for any fifty-character run that also appears in another work, and reject it if there is one. The validation plays score zero because scoring zero was the condition of their being the validation plays. A reader could not possibly have known that. I did know it, and did not think of it, until a reviewer did.

The honest baseline uses works that were selected for nothing: **each of the thirty-nine training works against the other thirty-eight.**

| | Share of fifty-character windows that occur elsewhere |
|---|---|
| All thirty-nine works | 0.000190 |
| Without the five that genuinely reprint one another | 0.000046 |
| The poet's words only, without those five | about 0.000003 |

(*Measured*, twice, by a reviewer and by me, to the same digits.) The five exceptions are real: *The Passionate Pilgrim* reprints sonnets that also appear in *The Sonnets* and *Love's Labour's Lost*, and *Venus and Adonis* and *The Rape of Lucrece* share their dedication to Henry Wriothesley. Take those away and what is left is mostly the editor, not the poet: "Dramatis Personæ" headers, "ACT I / SCENE I. London. An ante-chamber", and the like.

So the innocent rate is about one window in twenty thousand, and nearly all of it is furniture. That is the bar.

## What counts as reciting

Three decisions, all made before the model was measured and written into [the build log](../docs/BUILD_LOG.md):

**A copy is raw characters, and the whole curve is reported.** Fifty characters is the unit part 2 promised, but fifty on its own is a near-certain zero for the model and for an innocent writer alike, so it cannot show how close either came. The counts at twenty, twenty-five, thirty, forty, fifty, sixty, eighty and a hundred characters are all published together, and so is the longest match.

**The poet and the editor are counted separately, and neither is a footnote.** Most of what a Shakespeare text file contains was not written by Shakespeare: cast lists, act and scene headings, entrances and exits, and the speaker label before every single line. That text is short, formulaic and repeated across works, which makes it the easiest thing in the corpus to reproduce and the least interesting. About a tenth of the corpus is editorial furniture by the rule we published (*measured*: 9.6% of the training works, 9.3% of a validation play, which is a reassuring symmetry). A passage also has to carry at least twenty-five letters to count, or a run of indentation scores as a copy.

This split is not a technicality. It decides the headline. With the results in front of you it would be possible to write "it reproduces sixty-six characters of its training text word for word" and equally possible to write "the longest line of Shakespeare it reproduces is eighteen characters", and both would be true.

**The verdict was fixed in advance,** so it could not be chosen afterwards:

> The model **recites** if the longest confirmed passage of Shakespeare's own words it reproduces from the training works reaches fifty characters, and is at least twice the longest such passage it reproduces from the two plays it never read. It **does not recite** if that figure is under fifty characters and exceeds the validation figure by no more than ten. Anything between is reported as it stands, with every passage printed.

One honest caveat about this pre-registration, which is weaker than parts 5 and 6. The reviewer who designed the measurement prototyped it at about a tenth of full scale, so model numbers existed before the rule was written, and I had seen them. The rule is still committed before the full measurement, and the build log says all of this. But nobody should read it as blind.

## Asking the model, properly

Sampling is the wrong instrument for this question, and it is worth saying why before we get to the right one.

When you ask a model for text and then look for copied passages, you learn what the model *happens* to say. It can know a passage perfectly well and never volunteer it. The right question is what it can be *made* to say, and that needs a different approach: walk the model along the true training text and ask, at every one of 4.8 million positions, whether the character that really comes next is the one it would have written. A run of such positions is a **candidate**.

Then comes the step that an earlier version of this skipped, and which changed the answer by a third. A candidate is not yet an extraction. The screen judges each position with somewhere between 128 and 255 characters of run-up, because that is how the scoring windows overlap; a model actually generating has the full 256. That is enough to change its mind. So every long candidate is **confirmed**: from the real 256 characters before it, the model is made to write, with no dice at all, and we count how many characters it gets right before it diverges. Only confirmed lengths are ever reported. In the reviewer's prototype, fourteen candidates of fifty characters or more became nine confirmed ones, and the longest fell from ninety-nine characters to sixty-six.

[[RESULTS]]

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

Two rules that are not about spelling. **Trailing spaces are stripped**, because in this project's tokenizers a space belongs to the token *after* it (part 3): a prompt ending in a space is asking the model to continue a word that has not started. And **START is never prepended**. START means "a new work begins here"; an ordinary prompt is the middle of one.

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

It found the only `cat` it knows. That is the whole model in four lines: it is an in-character autocomplete, not an assistant, and it has never once been asked a question.

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

Nobody cast ROMAN. With no answerer named, the model writes the speaker label itself and we report who it picked. It did not care for Juliet's headache, and it has never heard of money problems, but it knew that a line of dialogue was owed and roughly what one sounds like.

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

**And now the part that matters more than the choice.** Two layers of selection sit on that 1.7492, and neither sits on the test score that part 8 will report. Within the run, the best of forty-one stops was kept. Between runs, the best of three was kept. Simulated with this project's own run-to-run spread, picking the best of three flatters a score by about 0.005, and the best of eleven by about 0.010. The three eligible runs are 0.0083 apart, which [part 6's own test](06-which-tokenizer.md) calls a tie.

So the release buys nothing this project can demonstrate, and the model card will say so. What part 8 reports will be two numbers, not one: the released model's own test score, and the mean and spread of all three eligible runs. The first is the artifact. The second is the recipe.

## A promise withdrawn

[Part 2](02-the-data.md) said the released model would be retrained at the end on all forty-four works, so that users would get a model that had read everything. That promise is withdrawn, and the reason is the whole argument of this series.

A model trained on all forty-four works can never be evaluated honestly, because the three test works would be inside it. It would be the one model in the project whose quality is an expectation rather than a measurement. Making that the headline artifact, on the last page of a series that spent six parts insisting on the difference, would be a poor joke. The gain is 4.8% more training text, which is about the size of the noise we spent two parts measuring.

**GP-Thee-11M is the thirty-nine-work model, with a real test score.** If a forty-four-work version is ever made it will be a separate download with a different name, and its card will begin by saying it has no held-out score and never can have one.

[[WHERE]]
