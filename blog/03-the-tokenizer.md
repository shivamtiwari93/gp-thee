# Building GP-Thee, part 3: teaching a computer to read

*A model cannot read letters. This part builds the thing that turns Shakespeare into numbers and back: first one character at a time, then in fragments of words, learned from the plays themselves.*

[Part 2](02-the-data.md) ended with 44 clean works and a rule: from here on, anything learned from text is learned from the 39 training works only. The first thing that learns from text is not the model. It is the tokenizer.

## Why a model needs one

A neural network does arithmetic. It cannot be handed the letter `T`. It can be handed the number 41, which it uses to look up a row in a table (the embedding table from part 1), and that row of 384 numbers is what the model actually works with.

So before any training, we need a fixed, reversible agreement: this piece of text is this number. A **tokenizer** is that agreement. The list of pieces it knows is the **vocabulary**. Turning text into numbers is **encoding**, and going back is **decoding**.

One distinction to hold on to, because the rest of the post depends on it. An entry in the vocabulary is a **piece**. Each time a piece occurs in a text, that occurrence is a **token**. The word ` the` is one piece and, in our training works, 22,693 tokens.

The rule of the project applies here with full force. Many people building a small model borrow a tokenizer, often GPT-2's, whose 50,257 pieces were learned from millions of web pages. Ours may only know what is in the training works.

## The simplest tokenizer: one character, one number

Sort the 97 distinct characters of the training works, and number them. The newline is 0, the space is 1, `!` is 2. The last few are `“` and `”` and `…`.

```
"To be"   →   [41, 65, 1, 52, 55]
```

That is the whole idea, and it is a perfectly good tokenizer. Nothing is learned except which characters exist. It is what the best-known Shakespeare tutorials use, nanoGPT's among them.

Its cost is how little the model sees at once. Our model looks back over 256 tokens. With one token per character, that is 256 characters. The median non-blank line in the training works is 39 characters, so the model sees about six lines of verse, and it must spell every word letter by letter, every time.

## Two decisions before going further

**There is no "unknown" token.** Older tokenizers have a catch-all token for anything unrecognised. GPT-2 and its descendants avoid one by starting from the 256 possible bytes instead of from characters, so any text at all can be encoded. We do neither. Our training text uses only 99 of the 256 byte values, so 157 rows of the embedding table would never be trained, and a curly apostrophe is three bytes, so a model producing bytes could emit a third of one. In a universe with a closed alphabet of 97 characters, refusing is simpler. Encoding and then decoding gives back exactly the text you started with, for any text in that alphabet. A character outside it is an error, not a guess.

This has a funny consequence. You cannot type an ordinary apostrophe at this model. The training works contain 23,019 apostrophes and every one of them is the curly kind, `’`. The straight one on your keyboard, `'`, does not exist in this universe. Nor do `"`, `<`, `@`, `#` or `+`. When we build the part that takes prompts from people, it will need to translate. The tokenizer itself stays strict.

**One special token, `<|start|>`.** It is placed before every work when the corpus is laid out as one long stream of numbers. It gives the model something to hold on to: after START comes a title. It also gives us a way to say "begin a new work", and in principle a way to know when the model thinks a work is over, which is when it produces START itself.

Expect the first of those to work better than the last. START is followed by a title 39 times in the training stream, and a title is easy to recognise. But the model gets only 38 examples of a work *ending*, each one different, and part 2 removed the `THE END` lines. START is the most under-trained token we have, so the sampling code will not rely on the model to stop by itself.

START must never be confused with text, and the way to guarantee that is to make it untypeable. Shakespeare's alphabet has no `<`, no `|` and no `>`, so no text can ever spell it. The code does not take this on trust: a tokenizer refuses to be built from any alphabet that could spell START.

(My first version of that check was wrong in an instructive way. It refused any alphabet containing *any* character of `<|start|>`, which includes `s`, `t`, `a` and `r`. It rejected Shakespeare. The check only needs one character of START to be missing from the alphabet, not all of them.)

With START, the character vocabulary is 98, not 97. That is one more row in the embedding table, 384 more parameters, for a total of 10,757,760. Every vocabulary change in this series is priced the same way: 384 parameters per piece. It is still GP-Thee-11M.

## Fragments of words: byte-pair encoding

The idea that modern language models use is older than they are. It was invented in 1994 by Philip Gage as a way to compress files: find the most common pair of bytes, replace it with one unused byte, repeat. Hence the name **byte-pair encoding**, or BPE. In 2016 Sennrich, Haddow and Birch turned it on text for machine translation, merging characters instead of bytes and keeping the list of merges as a vocabulary. GPT-2 carried it into language models. Ours is the 2016 kind: it merges characters, not bytes.

Start with single characters. Count every pair of neighbours in the text. Take the most frequent pair and glue it together into a new piece, everywhere. That step is one **merge**. Count again. Repeat, and stop when the vocabulary is as big as you want. That size is the one knob.

Here it is on the string `aaabdaaabac`, written with a gap between tokens:

```
a a a b d a a a b a c
```

The most frequent pair is `a a`, four times. Overlaps count: each `a a a` holds the pair twice, even though only one of the two can be glued. (That is how the standard implementations count, and ours does the same. Count it only twice and it would tie with `a b`.) Glue it, working left to right: `aa a b d aa a b a c`. Now `aa a` and `a b` both occur twice. We break ties by a fixed rule, the pair that comes last in sorted order, and `aa a` wins: `aaa b d aaa b a c`. Then `aaa b`, twice: `aaab d aaab a c`. Every remaining pair occurs once, so there is nothing left worth learning. Three merges, and the text went from 11 tokens to 5.

That example is a test in our repository. The answer was worked out by hand first, and the code has to agree.

To encode *new* text, split it into characters and replay the merges in the order they were learned. Nothing is counted at that stage: a pair is glued only if it is on the list.

We built four sizes: 1,024, 1,536, 2,048 and 4,096 pieces. Each vocabulary is the 97 characters, plus START, plus one new piece per merge, so 4,096 means 3,998 merges. And because merges are learned most-frequent-first, the 1,024 vocabulary is simply the first 926 merges of the same list. We ran the loop once.

### What it learns from Shakespeare

Run the same loop on the 39 training works, with one extra rule about where gluing is allowed, which is the next section. Watch the order in which things appear:

| Merge | Piece | |
|---|---|---|
| 1 | ` t` | a space, then t: the start of the most common words |
| 2 | `he` | |
| 12 | two newlines | a blank line: the gap before a speech, four times in five |
| 13 | ` the` | the first whole word longer than one letter (` a` was merge 3) |
| 36 | ` and` | |
| 98 | ` thou` | |
| 139 | `[_` | the opening of a stage direction |
| 170 | `._]` | and its close |
| 209 | ` lord` | |
| 230 | ` love` | |
| 513 | ` death` | |
| 572 | ` ’tis` | |
| 982 | `HAMLET` | a speaker's name, in capitals |

Nobody gave it a dictionary, or told it that `[_` opens a stage direction. It found ` thou`, ` death` and `HAMLET` because they are frequent. It did get one hint, and it matters more than it looks: a rule about where a word may begin and end. That rule is the next section, along with what BPE finds when it is left without it.

The longest pieces it ever learns are ` Northumberland` and a run of 19 spaces. The indent before every sonnet number is 20 spaces, and the twentieth travels with the number, for a reason we are about to come to.

A famous line, under each tokenizer:

```
To be, or not to be, that is the question:
Whether ’tis nobler in the mind to suffer
```

| Tokenizer | Tokens for these 84 characters |
|---|---|
| characters | 84 |
| 1,024 pieces | 30 |
| 2,048 pieces | 28 |
| 4,096 pieces | 26 |

With 2,048 pieces, where `|` marks a cut and `⏎` is the line break, which is a token of its own:

```
To| be|,| or| not| to| be|,| that| is| the| qu|estion|:|⏎|W|he|ther| ’tis| no|bl|er| in| the| mind| to| suff|er
```

At that size the model's 256-token window holds about 727 characters, eighteen lines instead of six.

## The rule that costs compression, and why we keep it

Look at the pieces above. Each word carries the space *before* it: ` be`, ` or`, ` not`. No piece has a space between two words, and no piece mixes a line break with text. That is not an accident. It is a rule we impose.

Before any merging, the text is cut into **chunks**: a word with its leading space, a run of punctuation, a run of line breaks, indentation. Merges happen inside a chunk and never across one. The usual name for this step is **pre-tokenization**, and our pattern is a cousin of GPT-2's.

Why does the space go in front and not behind? Because a word is followed by a comma, a full stop or a line break far more often than it follows one. In the training works 84% of words have a space in front of them and only 73% have one behind. Putting the space in front gives more words one spelling.

Why have the rule at all? My first explanation, written as a comment in the code, was a guess presented as a fact. It said that without the rule the most frequent merges would glue the end of one word to the start of the next, such as `e ` followed by `t`. I had not run it. One of the AI auditors (their work is described below) did, and that particular pair is never merged. The general idea was right and the example was invented, which is a good reason to measure. So we did ([scripts/compare_chunk_rule.py](../scripts/compare_chunk_rule.py)), and the measurement turned up something I had not expected. At 2,048 pieces:

| | No chunk rule | With the rule |
|---|---|---|
| Characters per token, training works | 3.03 | 2.84 |
| Characters per token, validation works | 3.00 | 2.79 |
| Merged pieces seen 100+ times in training | 96.6% | 91.2% |
| Pieces that straddle two words | 237 | 0 |
| Pieces mixing a line break with text | 204 | 0 |
| Ways a common word gets cut up (average over the 150 most common) | 12.4 | 2.0 |
| Ways `come` gets cut up | 45 | 2 |

Read the first three rows carefully: **without the rule, the tokenizer looks better.** The text compresses better, on works it has never seen as well as on the ones it learned from, so the 256-token window would hold about 775 characters instead of 727. Its vocabulary is even better used. If those were the goal, the rule would be a mistake.

The last rows are why we keep it. Without the rule, the very first merge is `e` plus the space *after* it, and pieces such as `, and ` and `I have ` follow. The word `come` then reaches the model as `come `, or `come to `, or ` c` + `ome `, or 42 other things, depending on its neighbours. With the rule it is ` come`, and on the rare occasions it starts a line, `come`.

That is our reason, and it is a bet, not a result. Those last rows are a test the rule cannot fail: with chunks, a word has only a few forms by construction. And 45 overstates the damage, because the four commonest cuts cover three quarters of the times `come` appears. What we are betting is that a model with 5 MB to learn from is better off meeting each word in one form than seeing 7% more text per window. Only training models both ways could settle it, and it is on the list of experiments.

### The apostrophe is a letter

One detail of the chunk rule is specific to this text. GPT-2's rule, which most later tokenizers inherited, knows seven modern English endings (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`) and splits them off, so `don't` becomes `don` and `'t`. Any other apostrophe is punctuation. On Shakespeare that rule would cut `'tis` into `'t` and `is`, and `o'er` into three pieces.

In Shakespeare the apostrophe is doing the work of a letter, constantly: `’tis`, `o’er`, `ne’er`, `lov’d`, `th’`. So in our rule it counts as one, and ` ’tis` becomes a single piece at merge 572.

The worry with that choice is closing quotation marks, which are the same character and would end up glued to the word before them. We measured it: of those 23,019 apostrophes, somewhere between 75 and 150 are closing quotes stuck to a word, depending on how strictly you count. That is about half a percent at most.

## Learning from the training works only

Everything the tokenizer learns, it learns from the 39 training works: the alphabet, and every merge. The validation works are only ever encoded with the result. The test works are not opened at all.

A promise like that needs a test that could fail. Ours re-learns all 3,998 merges from the training works alone and demands the identical list, in order. Is that sensitive enough to notice a leak? We checked, with the validation works standing in for the leak: fitting on the 39 training works plus the 2 validation works changes merge number 33, and 3,848 of the 3,998 positions in the list. (Most of that is one early change shifting everything after it. Only 169 pieces differ. But the order is what the test checks.)

You can see the promise at work in a single name. With 2,048 pieces:

```
HAMLET.   →   HAMLET | .
ROMEO.    →   RO | M | E | O | .
```

*Hamlet* is a training work, so the name was frequent enough to earn its own piece. *Romeo and Juliet* is held out for validation, so the tokenizer has never seen the name and has to spell it out. (`RO` is a piece only because `ROSALIND`, `RODERIGO` and `ROSENCRANTZ` are in the training works. And at 1,024 pieces even `HAMLET` is two tokens, `HAM` and `LET`: it is merge 982, and that vocabulary stops at merge 926.)

That is not a flaw to be fixed. It is what an honest split looks like from the tokenizer's side: a new play brings new names, and the model will have to cope with them, just as a reader would.

## How big should the vocabulary be?

| Tokenizer | Vocabulary | Characters per token, training / validation | What 256 tokens cover | Merged pieces seen 100+ times in training | Parameters |
|---|---|---|---|---|---|
| characters | 98 | 1.00 / 1.00 | 256 characters | | 10,757,760 |
| BPE | 1,024 | 2.48 / 2.49 | 636 characters | 98.3% | 11,113,344 |
| BPE | 1,536 | 2.69 / 2.66 | 689 characters | 95.0% | 11,309,952 |
| BPE | 2,048 | 2.84 / 2.79 | 727 characters | 91.2% | 11,506,560 |
| BPE | 4,096 | 3.19 / 3.07 | 817 characters | 50.6% | 12,292,992 |

A merged piece is any piece made by a merge: everything except the 97 single characters and START.

Bigger vocabularies let the model see further. They also cost parameters, and, more seriously, they spread a small corpus thin. Every piece has its own row in the embedding table, and a row only finds out what its piece means from the times the piece appears. (Because of part 1's weight tying, every row is also nudged at every step, as one of the answers the model did not pick. That teaches a rare row to keep quiet, not what it is for.) At 4,096 pieces, half the merged pieces appear fewer than 100 times in the entire training text, and 125 never appear at all, because a longer piece swallowed them: `YRACUS` and `YRACUSE` exist only on the way to ` SYRACUSE`.

A rule of thumb from machine-translation research (Gowda and May, 2020) says: use the largest vocabulary in which 95% of the pieces still have at least 100 training examples. Counted the paper's way, over every piece, that lands at about 1,200 for us. But 26 of our pieces can never reach 100 at any size: 25 rare characters such as `œ`, and START. So we count merged pieces only, and counted that way it lands at about 1,536. We had planned to try 1,024, 2,048 and 4,096, which jump straight over both answers, so 1,536 joined the list. The rule was found for a different task and we have bent it, so we treat it as a hint about where to look, not an answer.

Two things about these vocabularies surprised us.

**Three merges in four are decided by a tie.** The reason is mostly arithmetic. Early merges are landslides: ` t` wins with 97,647 occurrences, the first tie does not happen until merge 172, and only 41 of the first 500 merges are ties. By merge 1,950 the winning count is down to 149, thousands of pairs are still in the running, and several usually share the top count exactly. From there to the end, 96% of merges are ties. Names add a few more: once a fragment that occurs nowhere else has formed, such as `YRAC`, every way of extending it occurs exactly as often as the name does.

So the last entries of any vocabulary size are an arbitrary choice among equally frequent pairs. We checked whether it matters. Flipping the tie rule changes 130 of the 3,998 pieces, and moves the token count by 4 in 1.5 million.

**Character names are a sixth of the big vocabulary.** All-capitals pieces, mostly speaker names and fragments of them, are 9% of the 1,024 vocabulary, 13% of 2,048 and 16% of 4,096. From 1,536 pieces upwards they are also almost the whole reason the tokenizer compresses the training works better than the validation works: the training cast has its own pieces, and the validation cast does not. Leave out the speaker-name lines, and at 2,048 pieces the gap disappears.

### How we will actually choose

Not from this table. We will train the same model with each tokenizer and compare them on the validation works. Three things are fixed now, before any model exists, so that the result cannot steer the rules.

**The measure.** A model's raw score is the loss from part 2: its average surprise *per token*. That cannot be compared across tokenizers. A guess among 2,048 large pieces is a harder guess than one among 98 characters, so its per-token surprise is naturally higher, and there are fewer guesses. A model that predicts 28 large tokens and one that predicts 84 small ones are not sitting the same exam. But however the text is cut, every model ends up putting a probability on the same whole text. So we add up the surprise over the whole text and divide by the number of *characters*, which is the same for every tokenizer. That is **bits per character**. (One bit is the surprise of a fair coin flip. PyTorch measures surprise with natural logarithms, so the total is divided by ln 2, about 0.693, to turn it into bits.)

**How long each model trains.** A bigger vocabulary makes the corpus shorter: 4.8 million tokens as characters, 1.7 million at 2,048 pieces, 1.5 million at 4,096. The same number of training steps is therefore many more passes over the text for a big vocabulary than for a small one. So tokenizers are not compared at a fixed step count. Each run keeps its best checkpoint on the validation works, and those are compared.

**The deciding number.** Validation bits per character over all the text, averaged over three runs from different random starts. If two tokenizers are closer than the spread between runs, the smaller vocabulary wins. Because unseen names hit the larger vocabularies hardest, we will also report the score separately for speaker-name lines and for everything else, and for each of the two plays. Those are there to explain the result, not to pick it.

## Making it fast

The obvious way to write BPE recounts every pair in 4.8 million characters before every one of 3,998 merges. It works. In plain Python it takes about 0.8 seconds per merge, which is the better part of an hour. Two standard shortcuts bring it to under six seconds.

**Count distinct chunks, not running text.** ` the` occurs more than 22,000 times, but it only needs merging once. The training works contain 1.2 million chunks and only 38,398 distinct ones. Work on those, and weight each by how often it occurs. That alone turns an hour into a couple of minutes.

**After a merge, only recount what changed.** Gluing `t` and `h` can only affect chunks that contained `t h`. Keep a note of which chunks have contained which pair, and visit only those. (We never tidy the note, so some visits are to chunks that no longer hold the pair. A wasted visit is harmless: the chunk's pairs are taken out of the counts and put straight back.) That turns minutes into seconds.

Shortcuts are where bugs live, so a slower version that recounts every chunk before every merge lives in our tests, and the fast one has to match it exactly: on 200 random strings, and on the first 120 merges of Shakespeare. All 3,998 the slow way takes minutes, so that comparison was run once, in the audit.

## The audit, and what it found in my tests

As with the data script, four AI agents were asked to attack the result, each writing its own tools.

One wrote an independent BPE from the description alone, the slow way, and ran all 3,998 merges. **Identical, in order.** One threw 658,544 strings at the encoder: every short combination of the kinds of character we have (letters, digits, spaces, line breaks, apostrophes, punctuation), and hundreds of thousands of random ones weighted towards the awkward kinds. No failures. One tried to prove a leak from the held-out works and instead proved there was none: refitting on the training works reproduces our saved files byte for byte.

The most useful finding was about my tests, not my code. The auditor broke the tokenizer on purpose, in a copy, to see whether the tests would notice. This is **mutation testing** in the strict sense. Part 2 corrupted the *input* to see whether the script's checks would notice. Here the *code* is broken to see whether the tests notice. Same idea, one level up.

- An encoder that ignored every merge after the first 1,500 passed all 28 tests.
- An encoder that scattered stray START tokens through its output passed all 28 tests.

The reason is worth understanding. My main test was the round trip: encode, decode, compare with the original. But a round trip accepts *any* way of cutting the text up, as long as the pieces join back together: `HAM` + `LET` decodes to the same text as `HAMLET`. And my decoder hid START by default, so extra ones vanished before the comparison.

Neither fault loses a character, which is why the round trip passed. Both would have quietly spoiled training. With the first, 450 of the 2,048 pieces are never produced, so their rows are never trained and every token count in this post is wrong. With the second, the model is told that works begin at random places.

The suite now has 51 tests. Round trips decode with START visible. One test pins the exact pieces of a known line. Token counts are checked against the published report. The slower reference BPE is part of the suite. Saved files are compared byte for byte with what the code writes.

### Looking at the test works, three times

This is the part of the post I would rather not write.

Part 2 set the rule that the test works are not examined until the very end. The audit found that part 2 itself had broken it: it quoted two statistics measured on the test works, and named two speaking parts from them as names the model would never have seen. One of those figures was the share of the test text taken up by unseen speaker names, which is exactly where these tokenizers differ. Had that number been in my head while choosing a vocabulary size, the test works would have had a vote.

I removed the figures, wrote the confession, and then did it again, twice, in the first draft of this very post. To show that our leak test is sensitive, I fitted a tokenizer on all 44 works, which means on the test works, and published two numbers that depend on their text. And I quoted the number of apostrophes in the whole corpus a few paragraphs away from the number in the training works, so that the test works' share fell out by subtraction. A reviewer caught both. Neither number influenced anything, and no model exists yet. But three slips is a pattern, and the pattern is that good intentions are not a control.

So the rule is now enforced by code instead of by me. Every script and test loads works through one function ([src/gp_thee/data.py](../src/gp_thee/data.py)), and asking it for the test works raises an error unless the caller states that this is the final evaluation. The tokenizer build no longer opens the test works at all. It does not need to: the alphabet of the whole corpus was recorded when it was cleaned, before the split existed, so the script can prove that every work is encodable without reading it. No token files are saved for the test works, since a file's size would give away its token count. And the sensitivity check above uses the validation works as its stand-in for a leak.

## Two things that will bite later

**A prompt that ends in a space.** A space normally belongs to the *next* token, so `To be, or not to ` ends with a space token on its own. The model has met that token, but rarely and in odd company: about 3,500 times in 1.7 million training tokens, and when a word follows, it is one of the few hundred whose leading space never got glued on (the commonest continuation is `ord`, as in ` order`). After a trailing space the model does not expect ` be`. The right prompt is `To be, or not to`, and the model supplies ` be` itself. The sampling code will strip trailing spaces.

**Straight quotes**, as promised above. The sampling code will turn `'` into `’` before encoding. For `"` it will have to decide between `“` and `”`.

## Where we are

New in the repo:

```
gp-thee/
├── src/gp_thee/tokenizer.py        both tokenizers, the chunk rule, the merge learner: about 220 lines
├── src/gp_thee/data.py             loads works by set, and keeps the test works locked
├── scripts/build_tokenizers.py     fits the tokenizers on the training works, saves them, checks them
├── scripts/compare_chunk_rule.py   what the chunk rule costs and what it buys
├── data/tokenizers/                five saved tokenizers and a report
└── tests/test_tokenizer.py         51 tests, including BPE the slow way
```

```bash
uv run python scripts/build_tokenizers.py
uv run pytest
```

We still have no model. But the text is now numbers, five different ways, and we know how we will choose between them. Next, in part 4: the model itself, all 10.8 million parameters of it, or up to 12.3 million, depending on which tokenizer wins.
