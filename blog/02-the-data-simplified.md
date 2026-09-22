# Building GP-Thee, part 2: the data — simplified

*We choose one source, turn it into an auditable corpus, and freeze an honest train/validation/test split before any model exists.*

> This is the simplified edition. For every cleaning rule, measurement, and edge case, read [the original Part 2](02-the-data.md).

## What you will learn

- Why choosing and licensing the source is part of model design.
- How to clean text without silently rewriting it.
- Why independent audits and deliberately broken inputs are stronger than a long list of assertions.
- How validation and test data serve different purposes.
- Why whole works make a harder but more honest exam than random text chunks.

## Data work is model work

[Part 1 — simplified](01-prerequisites-and-setup-simplified.md) ended with a reproducible environment and a 5.4 MB source file. Part 2 still does not train a model. It decides what counts as the corpus, what gets removed, and which text the model must never read.

The experiment's rule remains strict: Shakespeare is the only text in this universe. The source, cleaning code, and split must make that claim inspectable.

## Choosing one defensible source

The dataset needed four properties:

1. All the works, not a tutorial excerpt.
2. A licence compatible with publication.
3. Play structure such as speakers and scenes.
4. One consistent edition.

Project Gutenberg's eBook #100, *The Complete Works of William Shakespeare*, was the best fit. It is **5,422,721 bytes**, about **963,000 words**, and **5,359,444 characters** before cleaning. The totals differ because curly punctuation can take more than one byte. It contains 44 works: the Sonnets as one work, 38 plays, and 5 poems.

Other sources lost on important tradeoffs:

| Source | Why it was rejected |
|---|---|
| “tiny shakespeare” | Only about one fifth of the text; no sonnets, poems, *Hamlet*, or *Macbeth*. |
| Hugging Face “complete works” datasets checked | Shuffled labels, repeated copyright text, missing plays, or text already cut into short windows. |
| Folger Shakespeare | Clean, but non-commercially licensed and missing two poems. |
| A popular Kaggle spreadsheet | Only 36 plays, unknown licence, and thousands of stage directions assigned to the previous speaker. |

Adding a second edition would appear to create more data, but it would mostly duplicate the same plays. Identical passages would be repeated, while spelling differences such as `upon` and `vpon` would teach competing conventions. It would also weaken evaluation: a held-out play in one edition could still appear in training through its twin. One edition makes provenance and leakage easier to reason about.

## Preserve the structure the product needs

The cleaned corpus keeps play structure rather than flattening everything into prose. A typical passage has an act, a scene, a stage direction, a speaker name on its own line, and a speech. Speaker labels repeat more than 30,000 times.

That layout is useful training signal. Later, a prompt ending with `HAMLET.` on its own line can encourage the model to continue with a speech. The context window is only 256 characters, so the name will disappear during a long speech; the structure helps, but it does not give the model durable knowledge of who is speaking.

The project also assigned five genre labels: tragedy, history, comedy, romance, and poetry. Those labels are outside knowledge, so the model never sees them. They exist only to help construct held-out sets that cover every kind of writing.

## Map the file before changing it

The tempting cleaning workflow is to inspect the first page, write a few regular expressions, and run them over everything. That is exactly how a plausible script destroys data silently.

Before any cleaning rule was written, separate AI-assisted research sessions mapped boundaries, editorial additions, characters, and duplicated passages. Other sessions re-measured the findings and tried to break the proposed rules.

Four examples shaped the approach:

**A title search can succeed and still split the wrong place.** The contents page calls one work `KING RICHARD THE SECOND`; its real heading is `THE LIFE AND DEATH OF KING RICHARD THE SECOND`. The shorter title appears again 41 lines later in the cast list. Taking the first exact match would quietly attach the play's opening to the previous work.

**Blank lines are not boundaries.** The file has 311 runs of four or more blank lines, but only 42 separate works. Most plays do not end with `THE END`; only one does, two use `FINIS`, and 41 simply stop.

**Editorial contents can look exactly like the play.** Each play begins with a list of acts and scenes. Eighty-five headings there are byte-for-byte identical to real headings later. Only their position—from `Contents` to `Dramatis Personæ`—identifies them as a contents list.

**A broad digit rule would erase the sonnets.** *Venus and Adonis* has 294 editor-added margin numbers. A proposed rule for spaces followed by digits matched 447 lines: the 293 obvious margin numbers and all 154 sonnet numbers. A stricter version still missed one margin number because it had only one leading space. The final rule is restricted to that poem, matches layout rather than numeric value, and requires exactly 294 hits.

That last requirement became the general rule: every cleaning operation must say how many lines it expects to touch and where they belong.

## Define “clean” before cleaning

The governing sentence is:

> Remove what an editor or transcriber added, keep what Shakespeare's readers would see, and never reword a line of verse or prose.

The result was small but carefully bounded:

| | Raw | Cleaned |
|---|---:|---:|
| Lines | 196,022 | 194,293 |
| Words | 963,478 | 957,139 |
| Characters | 5,359,444 | 5,319,224 |
| Distinct characters | 100 | 97 |

Only **0.75%** of the file was removed.

Removed material includes Project Gutenberg wrappers, the file-level title and contents, each play's repeated contents list, three end markers, 294 margin numbers, one duplicated poem title, and an editor's row of asterisks.

Small mechanical defects were tidied: a stray leading space on 2,510 lines, 47 doubled spaces, ditto marks in a cast list, five typing slips, and unmatched editorial brackets. One speaker labelled `ANDARUS.` became `PANDARUS.`, matching his other 152 speeches.

The project deliberately kept work titles, cast lists, act and scene headings, speaker names, stage directions, verse indentation, sonnet numbers, dedications, curly punctuation, accents, and original spelling.

Some decisions were genuinely ambiguous. Cast lists are editorial, not Shakespeare's words, but they are how readers meet the characters and represent only 0.60% of the text. They stayed. The goal was not to manufacture a historically pure text; it was to preserve the edition as a reader encounters it while removing repeated navigation and transcription noise.

The output is 44 files in [`data/processed/works/`](../data/processed/works/), plus a manifest of sizes, checksums, genres, and rule counts. The complete policy lives in [`data/processed/README.md`](../data/processed/README.md).

## Build a pipeline that refuses to guess

[`scripts/prepare_data.py`](../scripts/prepare_data.py) treats silent success as the main failure mode.

- A **checksum**, a fingerprint of exact file bytes, rejects any raw file other than the one studied.
- Structural checks require known counts, such as 154 sonnets and exact rule-hit totals.
- Lines are marked for editing or removal without changing their original positions.
- Output is assembled only after every check passes.

That last promise was false in the first version. The script wrote files before its final twelve checks, so a late failure could leave a damaged corpus beside an old manifest.

An independent mutation audit found it. A **mutation test** deliberately changes an input and asks whether the safety checks notice. With the top-level checksum disabled so deeper checks could be exercised, the auditors tried 110 corruptions.

They found another serious issue: the work finder accepted the first heading match. Planting a title inside another play moved 909 lines of *Cymbeline* into *Hamlet*. A normal run was protected by the checksum, but the bug would matter whenever the project intentionally adopted a newer source file. The finder now requires exactly one valid match per title.

Three checks were also found to be unable to fail because earlier code had already guaranteed their conditions. Two were deleted and one was rebuilt. A green check is not evidence if no possible bad input can turn it red.

After fixes, the output was audited again from scratch. The independent alignment accounted for **2,874 edited lines, 1,881 dropped lines, and 152 added blank lines**. Outside whitespace, tabs, and margin digits, edits removed 33 characters and added 223; none changed a line of verse or prose. Every structural mutation on the final list was caught, and failed runs left the existing corpus untouched.

The checksum and structural checks solve different problems. Structure can catch a missing act but not a changed word such as `To be` becoming `To bee`. The checksum catches byte changes; the assertions explain whether the expected structure still exists.

## Decide what “Shakespeare” means

The book's title does not settle authorship.

Only 5 of the 20 poems in *The Passionate Pilgrim* are securely Shakespeare's, and those five repeat material already elsewhere in the corpus. Six plays are widely considered collaborations, representing about 14% of the words.

Removing disputed scenes would discard a large part of an already small dataset based on scholarly attributions that are not always agreed. The project therefore defined its universe as **the canon published under Shakespeare's name**. All 44 works remain in the corpus. Suspected collaborations and *The Passionate Pilgrim* are excluded from the held-out evaluation, where authorship matters most.

This is a product boundary, not a discovery made by the model. Writing it down prevents the definition from changing after results arrive.

## Split before anyone can see a score

The model needs three sets:

| Set | Purpose |
|---|---|
| **Train** | Text used to change the model's parameters. |
| **Validation** | Text consulted during development to choose settings and when to stop. |
| **Test** | A final exam that must not influence any choice. |

Validation is not a final exam. If thirty configurations are tried and the best validation score wins, some of that win is luck on those particular works. The untouched test set estimates how the chosen process performs on text that made no decisions.

Random chunks are poor held-out data here. A chunk of *Hamlet* can sit between training chunks from the same scene, with the same speakers and perhaps the rest of the same sentence. The model has not seen the exact slice, but the exam is surrounded by clues. That is **data leakage**.

Taking the last 10% of the file is also misleading. Because of source ordering, it contains no tragedy or history, is 53.5% romance and 33.0% poetry, begins partway through a play, and includes duplicated or disputed material. Holding out entire works asks the harder, clearer question: can the model predict a play it has never read?

## Choosing the five held-out works

The split was frozen using four criteria:

1. Cover every genre somewhere in evaluation.
2. Avoid the six suspected collaborations and *The Passionate Pilgrim*.
3. Avoid Shakespearean passages duplicated in another work.
4. Keep validation and test near 5% of the corpus each.

Those constraints produced the following split before any model existed:

| Set | Works | Characters | Share |
|---|---|---:|---:|
| Validation | *All's Well That Ends Well*; *Romeo and Juliet* | 274,727 | 5.2% |
| Test | *King John*; *The Tempest*; *A Lover's Complaint* | 233,161 | 4.4% |
| Train | The other 39 works | 4,811,336 | 90.5% |

The shares are rounded separately, so they total 100.1%.

The poem was effectively forced because the other eligible poem was only 2,071 characters. *King John* was the only eligible standalone history. The comedy and tragedy were the only eligible examples passing a stricter search for shared dialogue, so the frequently consulted validation set received them. Both *The Tempest* and *Cymbeline* qualified as romances; the shorter *Tempest* left more training text.

This creates an intentional limitation. Validation has no history, romance, or poetry, while test has no comedy or tragedy. Their overall scores describe different text, so each work must also be reported separately.

[`scripts/make_split.py`](../scripts/make_split.py) checks for leakage using every 50-character sequence after lowercasing and reducing text to letters and spaces. Four held-out works share none with any other work. *King John* shares one roughly 60-character sequence with *The Winter's Tale*: `Exeunt`, a scene heading, and `Enter`. It is editorial structure rather than either play's dialogue, so the script permits that one named overlap and rejects every other one.

## The cost of an honest split

The measured model will never train on *Romeo and Juliet*. That is a real cost for a Shakespeare model, paid to earn a meaningful score.

At this point in the project, the plan was to use the measured run to choose a training length, then retrain the public model on all 44 works. That would give users a model that had read the entire canon, but it could never receive an honest held-out score. The original Part 2 labels that quality as an expectation, not a measurement.

The series later [withdrew that promise](07-does-it-recite.md#a-promise-withdrawn): GP-Thee-11M remained the measured 39-work model because releasing an unmeasurable all-44 replacement would contradict the project's central rule. Stating both moments preserves the chronology: the all-44 release was the plan when the split was designed, not the final decision.

The exam is also small. Repeating a training run can measure variation from random starting weights and random batch order, but it cannot reveal how results would change under five different held-out works. The final score is therefore a score on these works, not a universal score for Shakespeare.

## The cleaned alphabet

Cleaning leaves **97 distinct characters**, three fewer than Part 1's raw-file estimate because the asterisk, tab, and one straight apostrophe disappeared. Twenty-three characters occur fewer than 50 times, together accounting for only 490 characters—under 0.01% of the corpus.

Every one of the 97 appears in the 39 training works, so held-out text introduces no unseen character. From this point onward, anything learned from text—including the tokenizer—must be learned from training works only.

With 97 vocabulary entries, the pre-START design at this point contains **10,757,376 parameters**. Part 3 adds START, a special work-boundary token, as entry 98; the final character model therefore contains **10,757,760 parameters**.

## Reproduce the result

From the repository root, rebuild the cleaned works and then recreate the frozen split:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/make_split.py
```

Together these took about four seconds on the project machine. They deterministically replace generated files under `data/processed/works/`, `manifest.json`, and `split.json`; they never alter `data/raw/100-0.txt`. The first command either produces all cleaned artifacts after every check passes or leaves the previous output untouched. Its summary should report **44 files**, **5,319,224 cleaned characters**, **97 distinct characters**, and **0.75% removed**. The second writes the frozen split and re-runs its leakage test. It should report **39 / 2 / 3 works** for train / validation / test and one accepted overlap: the editorial scene-heading sequence described above.

The generated files are designed to be inspected, not treated as opaque pipeline output. This small script reads their two manifests and prints the facts later stages depend on:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

root = Path("data/processed")
manifest = json.loads((root / "manifest.json").read_text())
split = json.loads((root / "split.json").read_text())

print("cleaned works:", manifest["output"]["works"])
print("cleaned characters:", f'{manifest["output"]["chars"]:,}')
for name, details in split["sets"].items():
    print(name, details["works"], "works", f'{details["chars"]:,}', "characters")
PY
```

This inspection is read-only and takes well under a second. Expected output:

```text
cleaned works: 44
cleaned characters: 5,319,224
train 39 works 4,811,336 characters
validation 2 works 274,727 characters
test 3 works 233,161 characters
```

Finally, open a real output rather than trusting only the totals:

```bash
sed -n '1,24p' data/processed/works/08-the-tragedy-of-hamlet-prince-of-denmark.txt
```

This command is also read-only and should return immediately. You should see the title followed by `Dramatis Personæ` and the cast—not a flattened bag of sentences. This is a quick human check that the structure the product needs survived cleaning.

## Takeaways

- Dataset choice combines coverage, licence, structure, consistency, and auditability.
- Cleaning needs a written policy before it needs regular expressions.
- Exact hit counts, checksums, independent alignment, and mutation tests protect against different failure modes.
- A test that cannot fail adds confidence without adding safety.
- Whole-work holdouts are harder than random chunks but make the evaluation question clearer.
- Validation guides choices; test data must not guide anything.
- A small, genre-aware split still has sampling limits, so per-work results matter.
- The original all-44 release plan was explicitly an expectation and was later revoked rather than presented as measured quality.

At the end of Part 2 there is still no trained model. There is something more valuable first: a corpus whose edits are accounted for and an exam chosen before anyone could optimize for its answers.

[← Part 1: prerequisites and setup — simplified](01-prerequisites-and-setup-simplified.md) · [Next: Part 3, teaching a computer to read — simplified →](03-the-tokenizer-simplified.md)
