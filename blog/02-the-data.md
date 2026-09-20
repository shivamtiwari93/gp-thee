# Building GP-Thee, part 2: the data

*Finding the entire universe, reading it before touching it, removing 0.75% of it, and choosing the five works the model we measure will never be allowed to read.*

[Part 1](01-prerequisites-and-setup.md) ended with a working environment and a 5.4 MB text file. This part turns that file into training data. No model gets trained here either. That is normal: on a real project most of the work, and most of the ways to fool yourself, are in the data.

The rule of the project, once more: Shakespeare is the only text that exists. So this one file is everything the model will ever know.

## Finding the universe

We wanted four things from a dataset: all the works, a licence that lets us publish, the structure of the plays kept intact (who speaks, which scene), and one consistent edition.

**Project Gutenberg's eBook #100**, *The Complete Works of William Shakespeare*, has all four. It is one plain-text file of 5,422,721 bytes and about 963,000 words, in the public domain in the United States. (It is 5,359,444 *characters*. The two numbers differ because curly quotes and accented letters take two or three bytes each.) It holds 44 works: the 154 Sonnets, 38 plays and 5 poems.

What we looked at and turned down:

| Source | Why not |
|---|---|
| "tiny shakespeare", the 1 MB file used in most tutorials | About a fifth of the text. No sonnets, no poems, no *Hamlet*, no *Macbeth*. |
| "Complete works" datasets on Hugging Face | Each one we checked had a defect: shuffled rows with wrong labels, an old edition with a copyright notice embedded in the text hundreds of times, 36 plays only, or the text pre-cut into 80-character windows. |
| Folger Shakespeare | Beautifully clean, but licensed for non-commercial use only, and missing two poems. Wrong for a project we intend to publish. |
| A popular Kaggle spreadsheet of the plays | 36 plays, no poems, licence unknown, and over 6,000 stage directions attributed to whoever spoke last. |

One edition, not several. It is tempting to add a second edition to get more data, such as the Folger Library's modern text or the First Folio of 1623 in its original spelling. It would not be more data. It would be the same plays twice. Where the two editions spell a word the same way, the second copy is only a repeat, and we can repeat text for free by training for another pass. Where they differ, the model would meet `upon` in one copy and `vpon` in the other, have to learn both, and mix them when it writes. And later in this post we set some plays aside to test the model on. Every one of those would have a twin in the other edition that must be set aside too, or the test is spoiled.

## What is in the file

Measured on the cleaned text described below:

| Genre | Works | Share of the text |
|---|---|---|
| Tragedy | 11 | 29.1% |
| History | 10 | 27.1% |
| Comedy | 12 | 26.5% |
| Romance | 5 | 12.2% |
| Poetry | 6 | 5.1% |

The longest work is *Hamlet* (177,142 characters). The shortest is *The Phoenix and the Turtle* (2,071).

The genre labels are our addition, and they are knowledge from outside the universe. They are used for one thing only: when we set works aside for testing, the labels let us take them from every genre. The model never sees them.

The structure a model can learn from is all there. Here is the start of *Hamlet* as it appears in our cleaned corpus:

```
ACT I

SCENE I. Elsinore. A platform before the Castle.


Enter Francisco and Barnardo, two sentinels.

BARNARDO.
Who’s there?

FRANCISCO.
Nay, answer me. Stand and unfold yourself.
```

A speaker's name in capitals on its own line, then the speech, then a blank line. That pattern repeats more than 30,000 times. It is what will let us steer the model later: write `HAMLET.` on a line, and the most likely continuation is a speech. How much of Hamlet's own voice comes with the name is something we will have to measure. One limit is known already: the model sees only the last 256 characters, so in a long speech the name scrolls out of view.

## Read the file before you clean it

The obvious way to clean a text file is to look at the first screen, write some patterns, and run them. We did something slower. Before writing any rule, we had the file mapped by AI agents: separate Claude Code sessions, each given one question and no sight of the others' work. Four studied where works begin and end, what an editor added, which characters occur, and which passages appear twice. Four more re-measured every claim with their own code and tried to break every proposed rule.

The file turned out to be full of traps. Each of these would have damaged the corpus without any error message.

**The play that starts 41 lines late.** The table of contents lists `KING RICHARD THE SECOND`. The play itself is headed `THE LIFE AND DEATH OF KING RICHARD THE SECOND`. The shorter title does appear on a line of its own, 41 lines further down, as the first name in the cast list. A script that splits the file by searching for each title finds all 44, reports success, and quietly glues the opening of *Richard II* onto the end of the previous play.

**Blank lines cannot find the works.** Works are separated by about four blank lines. So are acts: the file has 311 runs of four or more blank lines, and only 42 of them separate works. We had also assumed that works end with `THE END`, because the first one does. It is the only one. Two more end with `FINIS`, and the other 41 just stop.

**The contents list that looks like the play.** Every play opens with an editor's list of its acts and scenes. 85 of those `ACT` lines are byte-for-byte identical to the real act headings later in the play. Capital letters and indentation do not tell them apart either. Only position does: the list runs from the line `Contents` to the line before `Dramatis Personæ` ("the persons of the play", the heading of every cast list).

**The pattern that would have deleted the sonnet numbers.** An editor numbered roughly every fourth line of *Venus and Adonis* in the margin:

```
Hunting he lov’d, but love he laugh’d to scorn;        4
```

Those numbers are 68% of all the digits in the file, and they had to go. The first pattern proposed, back at the research stage, was "two or more spaces, then digits, at the end of a line". A fact-checking agent ran it before anyone used it. It matched 447 lines: 293 in the poem, and all 154 sonnet numbers, which sit alone on a line after 20 spaces. It would have deleted them. The stricter pattern that replaced it spared the sonnets, but when the file was mapped it turned out to find only 293 of the poem's 294 numbered lines. The missing one was line 195658, which has a *single* space before its number. The gap, it turned out, runs anywhere from 1 to 22 spaces. Two of the numbers are also simply wrong: a line labelled 448 is line 450, and one labelled 790 is line 788.

The final rule matches the shape, never the value, and is confined to that one poem. The general lesson is the one that shaped the script: **a cleaning rule must state how many lines it expects to touch.** A rule that says "294, all inside *Venus and Adonis*" cannot silently eat the sonnets.

## The rule, and the rules

One sentence governs everything: **remove what an editor or transcriber added, keep what Shakespeare's readers would see, and never reword a line of verse or prose.**

| | Lines | Words | Characters | Distinct characters |
|---|---|---|---|---|
| Raw file | 196,022 | 963,478 | 5,359,444 | 100 |
| Cleaned | 194,293 | 957,139 | 5,319,224 | 97 |

0.75% of the file was removed.

**Removed:** Project Gutenberg's marker lines, the title block and table of contents, each play's contents list (1,488 lines, the bulk of it), the three `THE END` / `FINIS` lines, the 294 margin numbers, a poem title that was printed twice, and one row of asterisks an editor used to mark a lost line.

**Tidied:** a stray single space at the start of 2,510 lines, which had given the most common stage directions two spellings (`[_Exeunt._]` and ` [_Exeunt._]`); 47 doubled spaces; ditto marks in the *Julius Caesar* cast list, where `CASSIUS, ” ” ”` now reads `CASSIUS, Conspirator against Caesar.`; five typing slips, including a speaker called `ANDARUS.` who is `PANDARUS.` in his other 152 speeches. After the fixes, every `[` in the corpus has its `]`. Most of this second list did not come from reading the file. It came from the audit described below.

**Kept on purpose:** titles, cast lists, act and scene headings, speaker names, stage directions with the underscores that mark italics, all verse indentation, the sonnet numbers, dedications, curly quotes, accented letters, and the spelling exactly as found.

The first two parts of the rule pull against each other. Act and scene headings, many stage directions and the italic underscores were added by editors or transcribers. But they are what a reader sees on the page, and they are the structure we want the model to learn. So for anything that is part of the play as read, "keep what readers would see" wins. For front matter such as the contents lists, which only repeat headings that appear again inside the play, "remove what an editor added" wins. Cast lists were the closest call. Editors compiled them, not Shakespeare. We kept them because they are how a reader meets a play, and they are 0.60% of the text.

The output is 44 files, one per work, in [data/processed/works/](../data/processed/works/). You can open *Hamlet* and read it. Beside them the script writes `manifest.json`: every file's size and checksum, and how many lines each rule touched. The full list of rules is in [data/processed/README.md](../data/processed/README.md).

## A script that refuses to guess

[scripts/prepare_data.py](../scripts/prepare_data.py) is built around the idea that this is the only corpus there is, so a silent mistake is the worst outcome.

- **A checksum guarantees the words.** A checksum is a fingerprint of a file's exact bytes. The script refuses to run unless the raw file's fingerprint matches the file we studied.
- **Assertions pin the structure.** An assertion is a line of code that stops the program if something expected is not true. Every rule states the exact number of lines it touches. Some of those numbers come from outside the file: 38 plays times 5 acts is 190 act headings, and there are 154 sonnets. Those can prove a rule wrong. The rest (1,488 contents lines, 2,510 stray spaces) were measured once and then checked independently in the audits below. They cannot prove a rule right, but they stop it from changing quietly.
- **Line numbers never move.** Rules mark lines as dropped or edit them in place. Files are assembled only at the end, so no rule can shift the lines another rule is aiming at.
- **Nothing is written until every check has passed.**

That last line was not true in the first version, and we only know because we had the script attacked.

## Auditing the output, twice

We asked four more agents to audit the result, each writing its own tools rather than trusting the script's.

One aligned all 196,022 raw lines against the output and accounted for every dropped and edited line. One checked each work against what is known about it: five acts per play, 154 sonnets of which number 99 has 15 lines and number 126 has 12, 265 stanzas in *Lucrece*. One hunted for anything an editor had added that survived. And one ran **mutation tests**: corrupt a copy of the input on purpose, and see whether the script notices. (These were run with the checksum test switched off. With it on, every corruption is caught on the first line, and we learn nothing about the other checks.)

No text had been lost. But two of the auditors found real problems.

The clutter hunter found what eight agents reading the file beforehand had missed: the stray single space on about 2,500 lines, the doubled spaces, the ditto marks, and two places where an editor had put a word in square brackets. Those became the "Tidied" rules above.

The mutation tester found two faults in the script itself.

**The script wrote its files before running its last twelve checks.** A late failure left a damaged corpus on disk next to a manifest describing the old one, which is exactly what the script's own documentation promised could not happen.

**The work finder trusted the first match.** A title line planted in the middle of another play was accepted without complaint, and 909 lines of *Cymbeline* ended up in *Hamlet*'s file. This cannot happen while the checksum holds. It would happen on the day someone updates the checksum to a newer edition, which is exactly the day the checks are supposed to earn their keep. The finder now demands exactly one match per title.

Three of our own checks also turned out to be unable to fail, whatever the input. One asserted that the 44 works had been found in order, but the search for each work began where the previous one ended, so they could not be out of order. We removed two of the three and rebuilt that one so that it can fail. A check that cannot fail protects nothing, and it makes a script look safer than it is.

The new rules changed the output, so we audited it again instead of assuming. The second round, measured by an independent alignment:

- 2,874 lines edited, 1,881 dropped, and 152 blank lines added: four after each of the 38 play titles, so that every play opens the same way.
- Across all 2,874 edits, the only characters changed apart from spaces, tabs and the margin digits were **33 removed and 223 added**. Removed: 27 ditto marks, 5 editor's brackets, one straight apostrophe. Added: 219 characters of spelled-out cast-list descriptions, the missing P and I in two speaker names, one closing bracket, one curly apostrophe. None of them is in a line of verse or prose.
- 110 deliberate corruptions. Every structural one on our list was caught, and every failed run left the existing files untouched. The few that slipped through, such as `Finis` in a different spelling or a stray digit in the middle of a line, became new checks.

What no structural check can catch is a changed word. If someone alters "To be" to "To bee", the act count is still 5. That is the checksum's job, which is why there are two kinds of protection and not one.

## What "only Shakespeare" means

One question had been waiting since the research stage, and mapping the duplicated passages made it impossible to put off. Is everything in *The Complete Works of William Shakespeare* by William Shakespeare?

No.

- ***The Passionate Pilgrim*** was published under his name in 1599. Only 5 of its 20 poems are securely his, and those 5 are variants of text that is already in the corpus: Sonnets 138 and 144, and three poems from *Love's Labour's Lost*. The rest are by Richard Barnfield, Bartholomew Griffin, Christopher Marlowe, Walter Raleigh and people unknown.
- **Six plays are widely thought to be collaborations**, about 14% of the corpus by words: *The Two Noble Kinsmen* and *Henry VIII* with John Fletcher, *Pericles*, *Timon of Athens*, *Titus Andronicus* and *1 Henry VI*.

These attributions come from scholarship, not from anything in the file. Cutting those plays would cost a seventh of an already tiny corpus, on the strength of scene-by-scene attributions that scholars still dispute.

So we made a decision and wrote it down: **the universe is the canon as published under his name.** *The Passionate Pilgrim* stays, all twenty poems. What we will not do is *measure* the model on it, or on any of those six plays. That brings us to the split.

## The split, and why these five works

### Why split at all

The score we will use is called the **loss**: how surprised the model is, on average, by each next character. Lower is better, and zero means it was never surprised.

A model with 10.8 million parameters has 43 MB of weights, eight times the size of the corpus. Part 1 described the same shape starting to memorise a 1 MB sample. If we judged the model on text it had trained on, one that had learned nothing but how to recite would score close to zero, and that would tell us nothing about whether it had learned how Shakespeare writes. So some works are set aside, the model never trains on them, and it is judged on those. Works set aside like this are called **held-out**.

We need two held-out sets, not one.

| Set | What it is for |
|---|---|
| **Train** | what the model learns from |
| **Validation** | looked at *during* development: when to stop training, which settings are better |
| **Test** | looked at only at the very end, and never used to make a choice |

Suppose we try thirty settings (how much dropout, how big a model) and keep the one that scores best on the validation plays. The winner is partly the best setting and partly the one that got lucky on those two particular plays. The model never trained on them, but our choices did. After dozens of choices, the validation score is a little better than the model deserves. The test set is the exam that none of our choices has ever been checked against. The rule that keeps it that way: nothing we decide may depend on it. The moment a test score changes a decision, it has become a second validation set.

### Why whole works, and why chosen ones

The common shortcut, used by most tutorials, is to take the last 10% of the file. We measured what that would give us. The file holds the Sonnets, then the plays in alphabetical order, then the poems. So the last 10% is whatever comes late in the alphabet: 71% of *The Two Gentlemen of Verona*, all of *The Two Noble Kinsmen*, all of *The Winter's Tale*, and all five poems.

| Genre | Whole corpus | The last 10% |
|---|---|---|
| Tragedy | 29.1% | 0% |
| History | 27.1% | 0% |
| Comedy | 26.5% | 13.5% |
| Romance | 12.2% | 53.5% |
| Poetry | 5.1% | 33.0% |

Nobody chose that sample, so nothing about it is balanced. Tragedy and history are 56% of the corpus, and they would be graded nowhere: both held-out sets would have to be carved from this one tail. The sample starts 29% of the way into *The Two Gentlemen of Verona*, so the model would train on the opening of a play and then be graded on the rest of it. And it contains a likely collaboration, and *The Passionate Pilgrim*, five of whose poems reappear almost word for word in the training text.

The other shortcut is to hold out random chunks. A random chunk of *Hamlet* sits between two chunks the model trained on: same scene, same speakers, same names, sometimes the same sentence continuing. The model has not seen the chunk itself, but it has memorised everything around it, and that is enough to flatter the score. The standard name for this is **data leakage**: information about the exam reaching the student by a side door.

Holding out whole works asks the honest question: faced with a play it has never read, how well does it predict the next character?

The honest question is also the harder one. A new play brings names the model has never met. `ROMEO.`, `PAROLLES.` and `BERTRAM.` appear nowhere in the 39 training works, and speaker names like these are about 3% of the validation characters. (We measured this on the validation works only. The test works stay unexamined until the end.) So expect our numbers to look worse than the ones in tutorials. nanoGPT's well-known 1.47 was measured on a different file, with a different alphabet, using the last-10% shortcut. It is not a target for us, and our number cannot be compared with it. To give ours a meaning, we will score some simple baselines on the same five works.

### The criteria

1. **Leave no genre ungraded.** A comedy and a tragedy for validation. A history, a romance and a poem for test. This has a price, and we should name it. With two or three works per set, neither set can look like the whole corpus. Validation has no history, romance or poetry. Test has no comedy or tragedy, which is the kind of gap we just criticised in the last-10% split, only smaller and chosen on purpose. What the rule buys is that every kind of writing is graded somewhere, and that the final exam is on kinds of play we never tuned on, which makes it stricter, not kinder. What it costs: validation and test are scores on different texts, so they cannot be compared with each other. If test comes out worse than validation, that may only mean *King John* is harder to predict than *All's Well*. So we will report the score of each of the five works separately.
2. **His, by the mainstream view.** None of the six suspected collaborations, and not *The Passionate Pilgrim*. A score on a scene John Fletcher wrote does not measure what we care about. Two of our five have had doubters all the same. In 2012 two Oxford scholars argued that Thomas Middleton had a hand in *All's Well That Ends Well*, and one major edition has since accepted it. In 2007 another scholar gave *A Lover's Complaint* to John Davies of Hereford, and one complete-works edition left it out for that reason. Both claims are disputed, most editions print both works as Shakespeare's, and so do we. They are the least certain of the five.
3. **No passage shared with any other work.** "When my love swears that she is made of truth" opens Sonnet 138 and also the first poem of *The Passionate Pilgrim*. Hold one out and train on the other, and the model has seen part of its exam. Seven works share Shakespeare's own words with another work, so they must stay in training: the Sonnets, *Love's Labour's Lost* and *The Passionate Pilgrim* (the reprinted poems), *2 Henry IV* and *Richard II* (one quotes the other), *Lucrece* and *Venus and Adonis* (the same dedication, which is his). Another 16 works share only an editor's words with some other play, such as "SCENE II. Another part of the forest." We do not count those.
4. **About 5% each.** A quarter of a million characters is enough predictions that the average itself is not noisy, and it leaves 90% for learning. A typical play is 2 to 3% of the corpus, so each set is two or three works.

### From the criteria to five titles

Those rules still leave a lot of choice: 11 comedies, 9 tragedies, 6 histories, 3 romances and 2 poems, and a few thousand combinations that pass every check. Here is how the five were actually picked.

- **The poem was forced.** Only two poems are eligible, and *The Phoenix and the Turtle* is 2,071 characters, too short to score anything. That leaves *A Lover's Complaint*.
- **The history was nearly forced.** *King John* is the only eligible history that stands alone. Every other one belongs to a sequence. Holding out *Henry V* while training on both parts of *Henry IV* would hand the model its characters and their history.
- **The comedy and the tragedy have the cleanest record.** When the file was mapped, a stricter search was also run: no 40-character run of dialogue, and no run of 8 identical words, shared with any other work. Among the eligible plays, exactly one comedy and one tragedy passed it: *All's Well That Ends Well* and *Romeo and Juliet*. Validation is the set we will consult hundreds of times, so it got the two cleanest plays.
- **The romance was a judgement call.** *The Tempest* and *Cymbeline* both passed the stricter search. We took the shorter one, which leaves a little more text for training.

None of the five shares a character with a training play, though Shakespeare reuses names: the training works have another Helena, another Juliet and several Antonios.

The choice was made once, before any model existed.

| Set | Works | Characters | Share |
|---|---|---|---|
| Validation | *All's Well That Ends Well* (comedy), *Romeo and Juliet* (tragedy) | 274,727 | 5.2% |
| Test | *King John* (history), *The Tempest* (romance), *A Lover's Complaint* (poem) | 233,161 | 4.4% |
| Train | the other 39 works | 4,811,336 | 90.5% |

(Each share is rounded on its own, so they add up to 100.1%.)

### Checking for leaks, by measurement

[scripts/make_split.py](../scripts/make_split.py) writes this split to a file and re-runs the third criterion as a test every time it runs. "Shared passage" needs a definition. Ours is 50 characters in a row, about a line and a half of verse, after each work has been stripped down to lowercase letters. The script compares every 50-character run of each held-out work against all 43 other works. That catches a reprinted poem or a quotation. It does not catch anything shorter, and it should not: "enter a messenger" turns up in play after play, and a model that has learned it has learned Shakespeare, not the exam.

Four of the five share no run that long with any other work. *King John* shares one, of about 60 characters, with *The Winter's Tale*:

```
exeunt scene ii the same a room of state in the palace enter
```

That is the `Exeunt` that closes one scene, an editor's heading for the next, and the word "Enter". It is not a line of either play. The script accepts that one overlap by name and fails on any other.

### What it costs

The model we measure will never have read *Romeo and Juliet*. For a Shakespeare model that is a real loss, and it is the price of an honest number.

The released model does not have to pay it. Held-out text does two jobs: it gives us a score, and it tells us when to stop training, before the model starts learning its text by heart. Once the measured run has told us how many steps that takes, we can train a second model, **GP-Thee-11M**, on all 44 works with the same settings and the same number of steps. This is a standard move, and it has a price of its own. The released model's score on unseen Shakespeare cannot be measured, because there is no unseen Shakespeare left. We expect it to be about as good as the measured model, since it is the same recipe with 10% more text. That is an expectation, not a measurement. With nothing held out, we also cannot watch it for memorising, so we will check that its training curve tracks the measured run's, and test how much of its output is copied from the corpus.

One more honesty note. Two or three works is a small exam, and two kinds of wobble hide in it. The first is chance. A model's parameters start out as random numbers and the training text is fed in random order, so two runs with identical settings end with slightly different scores. When we compare two settings we will train each one three times and report the average and the spread. A difference smaller than the spread is not a result. The second is which works we happened to pick, and repeating runs does nothing for it. It does not hurt comparisons, because both settings sit the same exam. It does limit what the final number means: it is a score on these five works, not on Shakespeare.

The split is frozen. The five titles were committed to the repository before any model was trained, and we will not change them whatever the results look like. If we allowed ourselves to swap a play after seeing scores, we could end up picking the exam our model happens to pass, without ever meaning to. That is the same trap as the validation set.

## The alphabet

After cleaning, the corpus uses 97 distinct characters. Our first tokenizer will be the simplest possible: one token per character, so the vocabulary is those 97. That is three fewer than part 1 assumed, because the asterisk, the tab and the one straight apostrophe are gone. At 384 parameters per character, the model has 10,757,376 parameters, not 10,758,528. It is still 10.8 million, and still GP-Thee-11M.

The 97 are very unevenly used. 23 of them occur fewer than 50 times in 5.3 million characters: eight of the ten digits, `&`, `œ` (o and e joined into one character), and a handful of accented letters such as `é` and `ç`. A model cannot learn much about a character it meets a few dozen times, and some of these appear only twice. It also hardly matters to the score: together those 23 are 490 characters, under 0.01% of the text. All 97 occur in the 39 training works, so nothing in any held-out work can be new to the model's alphabet.

From here on, anything learned from text is learned from the training works only. That includes the tokenizer. Whether to move from characters to fragments of words is part 3.

## Where we are

New in the repo:

```
gp-thee/
├── data/processed/works/         44 cleaned works, one file each
├── data/processed/manifest.json  every file's size, genre and checksum; every rule's count
├── data/processed/split.json     the frozen train / validation / test split
├── data/processed/README.md      every rule, every keep decision
├── scripts/prepare_data.py       raw file in, cleaned works out, or nothing at all
└── scripts/make_split.py         writes the split and re-runs the leak check
```

To reproduce all of it:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/make_split.py
```

Still no model. Next, in part 3: teaching a computer to read, one character at a time.
