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

## Meet the model: what 1.80 bits actually looks like

A number that small deserves to be shown rather than asserted. Everything below was generated by
[scripts/demonstrate.py](../scripts/demonstrate.py) and recorded in
[docs/demonstrations.json](../docs/demonstrations.json), so every sample on this page is real output from the
released checkpoint, with its prompt, its temperature and its seed. Nothing here is typed by hand or picked from
a dozen tries. Where a sample runs longer than the page can carry, the block ends with a `…` line; the complete
text of every one, with its prompt, seed and temperature, is in
[docs/demonstrations.txt](../docs/demonstrations.txt).

### It is, literally, a guess about the next character

A language model is not a thing that "knows" text. It is a thing that, given some characters, produces a
probability for every character that could come next — all ninety-seven of them. That is the whole mechanism.
Here is GP-Thee-11M doing it, four times:

| after it has read… | its next-character guess |
|---|---|
| `…HAMLET.\nTo be, or not to b` | **`e`** 97.3% · `r` 1.2% · `l` 0.5% · `a` 0.3% |
| `SCENE III. Rome. The Capit` | **`o`** 99.9% · `i` 0.0% · `u` 0.0% |
| `…ROMEO.\nBut soft, what li` | **`k`** 61.6% · `e` 15.4% · `f` 7.8% · `g` 7.0% |
| `…KING HENRY.\nOnce more unto the b` | `o` 23.8% · `a` 23.7% · `e` 18.6% · `r` 14.6% |

Read those four rows slowly, because they contain the entire argument of this section.

In the first two it is nearly certain, and rightly: after `not to b` almost nothing but `e` can follow, and
`Capit` is going to be `Capitol`. In the third it is genuinely torn — `k` for `light`, but 15% on `e` and 8% on
`f`, because `what li…` could still become `lie`, `life`, `liege`. That hesitation is not a defect; it is the
model holding several possible sentences open at once.

Now the fourth row. `Once more unto the b` is the opening of the most famous speech in *Henry V*, and the next
letter is `r`, for `breach`. The model puts `r` **fourth**, at 14.6%, behind `o`, `a` and `e` — a near-uniform
shrug across four letters. It does not know the line. It has read that play dozens of times and it does not know
the line. Hold onto that; we will come back to it.

### Give it a page of a play and it writes a play

The model has no notion of a request. What it has is a very strong notion of what a printed page of Shakespeare
looks like. So the way to use it is to start one.

Hand it nothing but a speaker label:

```
$ uv run python scripts/sample.py --run sweep-char-seed-1 --prompt $'\n\nHAMLET.\n'

HAMLET.
The most brief may report all our loves; the court will be the
prisoner, and the fool be nothing but so weight in the villainy. I will infringe
to this action.

OSRIC.
Sir, I am in my called the likeness of your injuries for the
peril of your eye.

HAMLET.
I will, my lord.

OSRIC.
Will you go?

HORA
```

It gave Hamlet a speech, brought in Osric, gave *him* a speech, alternated the turns, kept the blank lines and
the full stops after the names, and was halfway through introducing `HORATIO` when it ran out of the characters
I allowed it. Neither name is made up: Osric and Horatio are the two courtiers who share Act V with Hamlet, and
the training text alternates their labels for long stretches in exactly this pattern. The model rebuilt the shape
of that scene without copying a word of it. It has learned the *form* of a play almost perfectly. It has learned
to say nothing.

Hand it a scene heading and it builds the furniture too:

```
$ ... --prompt $'SCENE III. Rome. The Capitol.\n\nEnter CAESAR and the Conspirators.\n\nCAESAR.\n' --seed 7

SCENE III. Rome. The Capitol.

Enter CAESAR and the Conspirators.

CAESAR.
Now, where now? Come, get you gone;
Bid them by the marks them readily.

[_Exeunt._]

SCENE VI. Rome. A Street

Enter Caesar, Caesar, Agrippa and a Capitol; Lepidus and Agrippa.

CAESAR.
Hail! How now, my noble lord! Where’s Caesar call’d?

CAESAR.
Hail, hail! Lord Antony of Caesar!
…
```

Everything structural is right: the scene ends with `[_Exeunt._]`, a new scene opens with a heading, an entrance
direction follows, then a speaker. Everything *meaningful* is wrong. It numbered the next scene VI instead of IV.
It entered "Caesar, Caesar, Agrippa and a Capitol" — the same man twice and a building as a person. Then Caesar
asks where Caesar is, and answers himself. This is the most honest picture of the model in the whole post: a
perfect imitation of the shape of a page, with no one home.

Give it an entrance and it invents the exits too:

```
$ ... --prompt $'\n\n[_Enter a Messenger._]\n\nMESSENGER.\n' --seed 3

[_Enter a Messenger._]

MESSENGER.
And yet so please you.
In the way I dismalled myself, to fight
Against contempt, and may be buried.

[_Exit, gives him a letter._]

ROSS.
Believe me, and let me have night.
[_To Theseus._] You must have them much chase in your rule and see.
```

`[_Exit, gives him a letter._]` is not a direction any edition prints. Then `ROSS`, whom nobody entered, answers a
messenger who has just left, and aims an aside at Theseus, who is in a different play. Note `dismalled`; we come
back to it.

And because it is an autocomplete, anything you type gets continued rather than obeyed:

```
$ ... --prompt "Write me a poem about a cat."

Write me a poem about a cat. I will use it a cure, and let
the most excellent lady to hear a curer of my mirth, for well I see it,
so it is a woman that could be a thing so good a more a woman.
…
```

It does not write a poem about a cat. It treats your sentence as the opening line of a scene and carries on. Same
with a modern one — watch how fast it escapes to the sixteenth century:

```
$ ... --prompt "The meeting is at nine o'clock, and the budget"

The meeting is at nine o’clock, and the budget that begin between the
old buds and twenty can make the valour. He shall be thrown forth.
Ah, that can win their wings through the naked battle gates, and all
these jollities are without a piece of b
```

Four words in, "budget" has become "old buds" and we are at the battle gates. It has no other register to fall
into.

It will also write verse if you give it verse-shaped bait. Here is a sonnet number that does not exist — the
sequence stops at 154:

```
$ ... --prompt $'\n\n\n                    155\n\nWhen first I looked upon thy'

                    155

When first I looked upon thy tender company,
Comes in perfect thoughts that makes haste more wiser hold;
And I, for well I see thou art a cold rotten for some monarch.
Here in the altar trials he lies,
And in the way of the commonwealth lies;
Anon he will subscribe thee for his part;
…
```

Line breaks in roughly the right places, a vaguely iambic pulse, sonnet-ish diction, and not one line that means
anything. "A cold rotten for some monarch" is the model's own.

### The temperature dial, which is the only knob that matters

Temperature decides how closely the dice follow the model's own odds. At 0 there are no dice at all: it takes its
likeliest character every single time. Above 1 it takes chances. Here is one prompt — `JULIET.` — at four
settings, same seed:

**Temperature 0** — it takes the safest path, and the safest path is a loop:

```
JULIET.
I will not see thee another man in the world.

JULIET.
I will not see thee another man in the world.

JULIET.
I will not see thee another man in the world.
…
```

**Temperature 0.5** — coherent, cautious, and slightly obsessed with one word:

```
JULIET.
Is not this the matter?

SILVIUS.
Ay, indeed.

JULIET.
Ay, but the truth is the matter of the matter in the laws.

SILVIUS.
Why, then the laws is most strangely bound.
…
```

**Temperature 0.8** (the published default) — livelier, and the cast starts sliding:

```
JULIET.
Thy good will, and princes!

SILVIA.
What dost thou think of me, my friend, thy country’s by the mother’s
belly.

JULIA.
I think he hath not so much past so flat as thou wouldst weep.
…
```

Note `JULIET` → `SILVIA` → `JULIA`. It is not tracking who is on stage; it is producing plausible-looking names.

**Temperature 1.2** — it gambles, and invents:

```
JULIET.
Thyself!

HOLOFERNES.
Now, they are, their properest, go your trespass;
Be not in every errand of other life,
Divide what it could.

LUCETTA.
Not more sorrow, an ’tre shower,
An adoxing Clown is waste
```

`’tre` is not a word. Neither is `adoxing`. That is the dial doing exactly what it says: at low temperature the
model is repetitive and safe, at high temperature it is inventive and wrong, and the interesting behaviour is in
between. There is no setting at which it becomes coherent, because coherence is not what 11 million parameters on
5 MB buys you.

### Speaking as a character

The nearest thing to a conversation this model can have is a scene, so that is literally how the wrapper builds
one: the transcript *is* a page of a play. You give turns, it writes the next speech and stops at the blank line.

```
$ uv run python scripts/sample.py --run sweep-char-seed-1 --scene

ROMEO.
But soft, what light through yonder window breaks?

JULIET.
I have a headache and no money.

ROMAN.
No, but ’tis a gentleman that we are all advanced; and what would you
have said?
```

Nobody cast `ROMAN`. With no answerer named, the model picks the speaker itself and we report who it chose — it
wanted a Roman, presumably because "Roman" is a word it has seen at the head of many lines. It ignored the
headache and the money entirely, because it has never encountered either complaint, but it knew a line of
dialogue was owed and roughly what one sounds like.

Name the answerer and it obliges:

```
$ ... --scene --as MACBETH

MACBETH.
Then liv’d me, and pale and made me think
That to be dream’d or butt for thee I cannot
Be thought of my life, as my uncle Suffolk,
But gives some money for thy comforts, whom
A happy beauty’s down with thy strength tonight.
```

Macbeth in blank verse, with "my uncle Suffolk" — a character from *Henry VI* — and a stray echo of the word
"money" it just saw. Style: convincing. Content: nobody's.

### This universe has ninety-seven characters in it, and your keyboard does not

One feature exists purely because the closed-universe rule demanded it. The model's alphabet is the 97 characters
that occur in the training works. Your keyboard can produce many more, and quietly feeding it a character it has
never seen would be a small lie. So every prompt passes through a normaliser first, which either converts, or
refuses and says why:

```
$ ... --prompt "I'll not be \"quoted\""
prompt as fed: 'I’ll not be “quoted”'
  "'" (apostrophe, U+0027) became '’' (right single quotation mark, U+2019)
  a straight double quote became “
  a straight double quote became ”

$ ... --prompt "naïve"
that prompt cannot be spelled in this universe.
'ï' (latin small letter i with diaeresis, U+00EF) is not one of the 97 characters in the
complete works, so the model has never seen it and cannot be asked about it. The nearest
letter this universe has is 'i'.
```

Shakespeare's editors used curly quotes, so a straight apostrophe is silently corrected to `’` and reported. A
tab becomes a space. But `ï` and `ñ` have no equivalent, so `naïve` and `Señor` are refused outright rather than
mangled. Refusing is the point: a project that spent seven parts insisting on measurement should not start
guessing on the last mile.

## "Isn't it just replaying the plays?"

This is the right question to ask, and the honest reason to ask it is that the output above *looks* like
Shakespeare. Fluent Elizabethan diction with correct play formatting pattern-matches, in a reader's head, to
"real Shakespeare" — so the obvious suspicion is that the model is a very elaborate lookup table.

It is not, and this is measurable rather than arguable. For every sample in this post I asked: **what is the longest
stretch of it that occurs anywhere in the 4.8 million characters it trained on?**

| sample | length | longest verbatim run | and that run is… |
|---|---|---|---|
| the HAMLET scene | 300 chars | **22** | `the likeness of your` |
| the Caesar scene | 330 chars | **34** | `.\n\n[_Exeunt._]\n\nSCENE VI. Rome. A` |
| `To be, or not to be…` | 240 chars | **19** | `.\nGod be with you,` |
| sonnet 155 | 260 chars | **20** | `of the commonwealth` |
| the modern sentence | 200 chars | **19** | `He shall be thrown` |
| the cat poem | 200 chars | **16** | `most excellent` |
| the stage direction | 240 chars | **18** | `to fight\nAgainst` |

(Runs are shown with `\n` for a line break; a space at either boundary is trimmed here, so some read a character
or two shorter than the count beside them. The counts are the record's.)

Out of two to three hundred characters at a time, the longest copied run is **sixteen to thirty-four characters**
— and look at *what* those runs are. Either stage furniture (`[_Exeunt._]`, a scene heading) or four-word phrases
so common they are nobody's property. Not one is a line of verse. This is the same result part 7 got by the
opposite method: what GP-Thee memorised is the *printing conventions* of an edition, not the poetry.

### It invents words, which a lookup table cannot

The stronger evidence runs the other way. I took every word longer than five letters out of the samples above —
76 of them — and looked each one up in the complete training works. Three appear **nowhere**:

- **`jollities`** — absent. But `jollity` occurs 6 times. It formed a plural Shakespeare never wrote.
- **`dismalled`** — absent. But `dismal` occurs. It conjugated an adjective into a verb.
- **`adoxing`** — absent, and not a word in any language. It assembled it out of English-shaped parts.

That is not copying, and it is not random either: `jollity → jollities` is a correct English plural applied to a
word the model only ever saw in the singular. Generalising morphology to a form you have never seen is precisely
what a language model does and precisely what a lookup table cannot. The same goes for whole phrases — `naked
battle gates`, `Caesar, Caesar, Agrippa`, `I will infringe` and `a cold rotten for some monarch` all occur
zero times in the works it trained on.

### The three lines it should know best, and doesn't

Start it on the most quoted sentence in the language and it cannot finish it:

```
$ ... --prompt $'\n\nHAMLET.\nTo be, or not to be, that is the'

HAMLET.
To be, or not to be, that is the tender of it. I have heard
Of love to wear a burden for it at all.

OSRIC.
And thou art not here, my lord. Farewell.

HAMLET.
God be with you, for love of your love.

OSRIC.
Ay, love of your love. I fear you love me, I am not assur’d of yo
```

*That is the tender of it.* The word is `question`, it is the very next one the model had to write, and it reaches
instead for a phrase that means nothing. Now the same trick on Henry V — recall the probability table at the top
of this section, where `Once more unto the b` put `r` fourth at 14.6%. And finally, the test the objection
deserves most. If the model were reciting, the famous lines would be the easiest
thing in the world for it. So I took the exact run-up to the single most famous line in *Julius
Caesar* and made it write greedily — temperature 0, its single likeliest continuation, no dice:

```
given:            "…and at last by Marcus\nBrutus._]\n\nCAESAR.\n"
what really follows:  "_Et tu, Brute?_—Then fall, Caesar!"
what the model wrote: "I am sorry for you."   (and then again, and again)
```

It had the cue, it had the speaker, it had read the play, and it produced a polite condolence on a loop. Put that
beside `the tender of it` and beside `r` for `breach` in fourth place, and the picture is consistent: **not one
of the three comes back from the probe that should retrieve it most easily.**

Read that for exactly what it is. Three prompts, decoded once each, cannot prove a line is absent — and the table
at the top of this section says so itself, because `r` at 14.6% means that sampled rather than taken greedily,
`breach` does come up, roughly one draw in seven. The systematic version of the question is part 7's scan, which
walks all 4.8 million training positions instead of trying three, and there the longest unbroken stretch of the
poet's verse the model reproduces anywhere is twenty-five characters. What GP-Thee holds is not the line; it is
the language the line is made of.

Which is, if you think about it, the same fact as 1.80 bits per character. Take the objection at its strongest.
Not a table of whole plays — that one is already dead, because the model shares not a single run of forty
characters with these three works, so a replayer would have nothing to reach for. The strong version is a table of
*fragments* with a fallback, and the table at the top of this post already contains one: the 6-gram is nothing
but counts of every five-character context in the training works, plus a rule for what to do when a context is
new. Fitted on the same 39 works and turned loose on the same three, it scores 2.32 — and no order does better;
5 through 8 land between 2.32 and 2.53. That is the ceiling on lookup, the best table anyone can build out of this
training text. GP-Thee scores 1.80 against it. Those 0.52 bits are what you only get by learning how Elizabethan
English *works*.

## What it cannot do

A fair accounting, because the samples above are flattering in one direction and the limits are the more useful
half:

1. **It cannot follow an instruction.** Nothing in its universe has ever asked anyone to do anything. Requests
   get autocompleted as dialogue.
2. **It cannot answer a question**, for the same reason. There is no question-answer pattern in a playtext, only
   characters talking past each other.
3. **It has no memory of who is on stage.** `JULIET` became `SILVIA` became `JULIA` within four lines; Caesar
   greeted Caesar. It tracks the *shape* of turn-taking, not the participants.
4. **It cannot hold a thought across two sentences.** Every sample is locally fluent and globally meaningless.
   Its context is 256 characters — about three lines of verse — and it has no plot, no goal and no state.
5. **It knows nothing outside Shakespeare.** No facts, no arithmetic, no modern world. Ask about a budget and you
   get battle gates.
6. **It cannot spell outside 97 characters.** No `ï`, no `ñ`, no emoji — and by design it refuses rather than
   approximating.
7. **It repeats itself when made to be careful.** At temperature 0 it loops; the only cure is randomness, which
   trades repetition for nonsense.

None of these are bugs to be fixed. They are what 11 million parameters trained on 5 MB of one author is: an
in-character autocomplete with a superb ear and nothing to say.

*(One honest note about the machinery behind this section. The first run of `demonstrate.py` reported `e` at 81%
after `not to b` instead of 97%. The bug was mine and it is a classic: `load_checkpoint` does not put the model in
evaluation mode, and I called the model directly instead of through the sampler, so dropout was still switched on
— 20% of the network's values were being zeroed at random while I measured its confidence. The generated text was
never affected, because `generate()` sets eval mode itself. Caught, fixed, and noted here because a series about
measuring honestly should say when its own instrument was misread.)*

## What the number is, and what it is not

GP-Thee-11M predicts a held-out page of Shakespeare at 1.80 bits per character. A fair coin per character would be 1 bit; the raw alphabet is 6.6; a good general-purpose compressor is 2.6. So the model has learned a great deal about how Shakespeare's characters follow one another — enough to halve the compressor's surprise above that one-bit floor — from five megabytes of text and eleven million parameters, with no pretraining, no outside data, and a tokenizer, model and training loop all built from nothing across this series.

It is also, cheerfully, not much of anything else. It cannot follow an instruction, it has never been asked a question, and it reproduces not one line of the verse it was trained on. It is an in-character autocomplete for a universe that contains only Shakespeare, and now we know precisely how well it autocompletes: 1.80 bits per character, on a history, a romance and a poem it had never met, beating every predictor that came before it and reciting none of them.

That was the question. The works are spent. There is no number after this one, which is the whole reason it took a locked folder and a one-way door to earn.

## Where this leaves the project

Eight parts, and GP-Thee is complete: a corpus cleaned and audited, a split frozen before any training, a tokenizer chosen by a rule, a model and training loop built and checked, a recipe nobody tuned, eleven runs, one of them named and released, a direct answer to whether it is a parrot, and now a single honest score on text it never saw. Every number in the series was measured before it was written, and the ones that were wrong — and several were — were caught and corrected in the open.

The model itself, GP-Thee-11M, is on Hugging Face; the entire project, including every training log, every score, and the per-character arrays behind this final number, is on GitHub. It was built for no reason other than to see whether it could be done honestly, start to finish, on a laptop. It could.

*The complete build log, with all twenty entries and every measurement, is in [docs/BUILD_LOG.md](../docs/BUILD_LOG.md). The final evaluation's full record, including the per-token surprise of every model on every test work, is in [docs/final-evaluation.json](../docs/final-evaluation.json) and [docs/final-evaluation/](../docs/final-evaluation/).*
