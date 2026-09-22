# Building GP-Thee, part 7: does it recite? — simplified

*A small model read the same five megabytes of Shakespeare thirty-four times. We tested whether it learned the language—or merely stored passages word for word.*

> This is the simplified edition. The [full technical version](07-does-it-recite.md) includes every copied passage, the complete run-length curves, sampler output, audit history, and release-selection record.

## What you will learn

- How to test memorisation without relying on a few generated samples.
- Why a seemingly perfect control group turned out to be circular.
- What GP-Thee copied from its training data—and what it did not.
- How we limited eligibility before selecting the best of three final runs.
- Why we kept a measurable 39-work model instead of retraining on all 44 works.

## The question everyone should ask

GP-Thee has 10.8 million parameters and trained on only 39 works. It made 34 passes through that small corpus.

So when its output looks Shakespearean, the obvious question is:

> Did the model learn patterns of language, or did it simply memorise the plays?

Looking at a handful of samples is not enough. A model might remember a passage perfectly and never happen to generate it. We needed a systematic test.

## The first baseline was invalid

The project had two validation plays that no model trained on. At first they looked like an ideal control: check how often a passage from those plays already appears in the 39 training works.

At 50 characters, the answer was zero.

That result looked excellent. Any 50-character match produced by the model could be called memorisation.

It was also circular.

Back in Part 2, the validation plays were selected partly because they did not share long passages with the training works. A zero-overlap result was therefore not a new discovery; it was one of the conditions used to choose those plays.

The honest baseline had to use works selected for no such reason: each of the 39 training works compared with the other 38.

After excluding five works that genuinely reprint one another, about one 50-character window in 20,000 also appears in another work. Most of those repetitions are not poetry. They are editorial patterns such as cast lists, scene headings, entrances, exits, and speaker labels.

That became the baseline.

## What counts as reciting?

We fixed three rules before the full measurement.

### 1. Measure raw characters at several lengths

The main unit was a 50-character exact match, but the analysis also counted 20, 25, 30, 40, 60, 80, and 100 characters. A single threshold can hide how close the model came.

### 2. Separate Shakespeare from his editors

Roughly a tenth of the corpus is structural material added by editors:

- cast lists;
- act and scene headings;
- entrances and exits;
- speaker labels.

This material is repetitive and easy to predict. Reproducing `SCENE III. The same.` is not the same achievement as reproducing a line of verse. We therefore reported both the full text and passages containing mostly Shakespeare's own words.

### 3. Confirm every candidate by making the model write it

The systematic scan walks through all 4.8 million training characters. At every position it asks whether the real next character is also the model's first choice. A long streak becomes a **candidate**.

But a candidate is not yet a confirmed copy. The screening windows and the generation window provide slightly different amounts of context, which can change the model's next choice.

So every long candidate was tested again: give the model the real preceding 256 characters, remove randomness, and make it generate until it differs. Only this confirmed length was reported.

That second step mattered. Fourteen training candidates reached 50 characters during screening, but only nine survived actual generation.

## The result

| | 39 training works | 2 validation plays |
|---|---:|---:|
| First-choice character is correct | 69.72% | 63.63% |
| Candidate runs of at least 40 characters | 52 | 1 |
| Confirmed runs of at least 40 | 34 | 1 |
| Longest confirmed passage | **66 characters** | 40 |
| Confirmed runs of at least 50 | 9 | 0 |

The model clearly predicts its training text better. But the content of its longest copies is more revealing than their length.

The 66-character passage is the beginning of a title page and cast list. The other long matches are scene headings, entrances, exits, and speaker labels. Among the nine confirmed copies of at least 50 characters, not one is a line of verse.

The longest passage containing mostly Shakespeare's words is 44 characters, but it crosses a speaker boundary:

```text
or he hath done me wrong.

KING HENRY.
What
```

The longest uninterrupted stretch of Shakespeare's own verse is only 25 characters:

```text
or he hath done me wrong.
```

The clearest summary is:

> GP-Thee's long exact copies were repeated structure added by editors, not long passages of Shakespeare's verse.

That result makes sense. The training works contain hundreds of scene headings and thousands of speaker labels. Those patterns are short, formulaic, and repeated. They are cheaper for a small model to store exactly than a unique poetic line.

## Why the preregistered rule did not give a neat answer

The verdict rule was written before the full measurement:

- “Recites” required at least 50 characters of Shakespeare's own words and at least twice the validation length.
- “Does not recite” required fewer than 50 characters and no more than 10 characters above validation.
- Anything else had to be reported without forcing it into yes or no.

The training result was 44 characters and the validation result was zero. That is below 50, but more than 10 above zero.

So the formal rule returned **between the two answers**.

In plain language, the content argues against long-form verse recitation, but the preregistered rule is formally inconclusive. Changing it after seeing the inconvenient zero would have defeated its purpose, so we kept the “between” result and published the passages.

This is a useful product lesson: a metric can be sensible in theory and still behave badly at an edge case. If the rule was fixed before the data, report the edge case rather than quietly repairing the rule.

## A bug that made the model look better

The first full run labelled several cast-list passages as “mostly the poet's.”

Two mistakes caused it:

- the front-matter detector stopped at the cast list's own heading, leaving the list itself unmarked;
- the stage-direction detector did not recognise `Drum`.

Both mistakes leaned in the flattering direction: editorial text was counted as Shakespeare's writing.

The passages were inspected because the protocol required every long match to be printed. The classification was fixed, named examples were added to the tests, and the measurement was rerun.

The important lesson is not that bugs happen. It is that qualitative inspection was part of the measurement design, so a plausible but wrong number did not become the headline.

## Why random sampling is a weaker test

We also generated text at **temperatures** 0, 0.5, 0.8, and 1.0 from four kinds of prompts. Temperature controls randomness: zero always selects the model's first choice, while higher values allow more varied choices.

Sampling found only one 50-character copy when the model was prompted with training text at temperature 0.8. The exhaustive scan found nine positions where it could be made to write 50 or more characters.

Sampling asks, “What did the model happen to say?” The scan asks, “What can the model be made to reproduce?” The second question is stronger for measuring memorisation.

The sampled grid still told a useful story. Across 320,112 generated characters, exact matches were common at 20 characters but fell rapidly with length. At 50 characters, the model produced 14 matching windows. The leave-one-out baseline gives 14.6 as a rough yardstick, not a true null expectation, because generated text does not have the same distribution as real Shakespeare. At measured widths of 30 characters and above, none of the copied windows was classified as Shakespeare's own words.

The secondary grid also had three disclosed deviations from its original plan:

- 20,000 generated characters per cell instead of 200,000;
- the first eight prompts from the ten-prompt suite rather than all ten;
- fixed seeded prompt positions rather than the planned arithmetic placement.

The positions were reproducible and not hand-picked. The primary exhaustive scan still ran at full scale, so these deviations do not affect the main scan or the formal verdict.

## Turning the model into something people can try

This part also built a public sampler.

The model knows only 97 text characters. A modern keyboard knows many more. Before a prompt reaches the model, a normaliser therefore:

- keeps a character the corpus already uses;
- converts a known equivalent, such as a straight apostrophe to the corpus's curly apostrophe, and reports the change;
- refuses an unsupported character instead of silently deleting or guessing it.

For example, `naïve` is refused because `ï` is outside the model's alphabet. That may feel strict, but it makes the interface honest: the user always knows what the model actually received.

Ten fixed prompts exercise the same checkpoint behaviours every time: dialogue, a stage direction, a sonnet opening, modern prose, an instruction, and more. The model is good at continuing the *form* of a play. It does not reliably follow instructions because its training universe contains no instruction-following examples.

Ask it to write a poem about a cat and it continues the sentence as Shakespearean dialogue. Give it a speaker label and it produces speeches, names, blank lines, and stage directions.

It is an autocomplete, not an assistant.

## Try it yourself: sample without training

Run this command from the repository root after `uv sync`. A fresh clone contains the trusted loader and model definition, while the first run downloads the public weights into Hugging Face's normal cache. It uses the same prompt normaliser as the project and a fixed seed. Change `prompt`, `temperature`, or `seed` and run it again:

```bash
uv run python -B - <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, "release/gp-thee-11m")
from load_release import load
from gp_thee.sampling import generate
from gp_thee.tokenizer import CharTokenizer

revision = "352de5f38ddf55cedbe3ac7712dd5321ecedd4cb"
folder = Path(snapshot_download("shivamtiwari93/gp-thee-11m", revision=revision))
model, document = load("cpu", folder)
tokenizer = CharTokenizer(document["chars"])
prompt = "\n\nHAMLET.\n"
sample = generate(model, tokenizer, prompt, characters=120, temperature=0.8, seed=0)
print(sample["prompt"] + sample["continuation"])
print("stopped because:", sample["why"])
print("normaliser notes:", sample["notes"] or "none")
PY
```

The fixed command begins `The most brief may report all our loves…`. The revision is pinned to the audited public release, the repository tree stays unchanged, and the held-out test works are never opened. `-B` prevents Python from leaving a bytecode cache beside the release loader.

If you trained a model locally—or have the original machine's checkpoint—`scripts/sample.py` provides the friendlier command-line interface:

```bash
# These commands require runs/sweep-char-seed-1/best.pt; a fresh clone does not contain it.
uv run python scripts/sample.py --run sweep-char-seed-1 --prompt $'\n\nHAMLET.\n' --characters 240 --temperature 0.8 --seed 0
uv run python scripts/sample.py --run sweep-char-seed-1 --scene --as MACBETH
uv run python scripts/sample.py --run sweep-char-seed-1 --suite
```

The last command writes the ten fixed prompts and outputs to `runs/sweep-char-seed-1/suite.txt` and `suite.json`.

### Inspect the memorisation evidence

You do not need a checkpoint to audit the reported scan. This read-only command prints the deciding measurements and every confirmed training copy of at least 50 characters from the committed record:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("docs/memorisation.json").read_text())
scan = report["runs"]["sweep-char-seed-1/best"]["scan"]
for label in ("training works", "validation plays"):
    row = scan[label]
    print(f"{label}: agreement={row['agreement']:.2%}, candidates 40+={row['candidates_of_40_or_more']}, "
          f"confirmed 50+={row['confirmed_of_50_or_more']}, longest={row['longest_confirmed']}, "
          f"longest poet-only={row['longest_confirmed_poet_only']}")
print("formal verdict:", report["verdict"]["verdict"])
print("\nConfirmed training copies of 50+ characters:")
for hit in scan["training works"]["confirmed"]:
    if hit["characters"] >= 50:
        print(f"{hit['characters']:>2} chars | editor={hit['the_editors_words']} | {hit['text']!r}")
PY
```

It should recover 69.72% versus 63.63% agreement, nine versus zero confirmed copies of at least 50 characters, longest passages of 66 versus 40 characters, and the formal `between the two` verdict. All nine printed 50-character copies are marked as editorial material. The full scan can be regenerated with `scripts/memorisation.py` only when a local `.pt` checkpoint exists; that script rewrites the JSON record, so use a disposable copy if you want to compare it with the committed result.

## Choosing the released run

Eleven character-model runs existed, including pilots made under different circumstances. Releasing the single lowest score among all eleven would select the luckiest number.

The release rule was written first:

- Only the three runs from the final tokenizer experiment were eligible.
- They had to use seeds 1, 2, and 3, the same committed code, 32-bit arithmetic, and default settings.
- The eligible run with the lowest recorded validation score would be released.

| Run | Validation bits per character | Result |
|---|---:|---|
| seed 1 | **1.7492** | released as GP-Thee-11M |
| seed 2 | 1.7571 | eligible |
| seed 3 | 1.7575 | eligible |

The three are only 0.0083 apart, inside the project's own “tie” threshold of 0.013. The released run is the output of a rule, not a model proven superior to the other two.

Selection still makes its validation number slightly optimistic. Choosing the best of three runs is expected to improve the displayed score by about 0.005 through luck alone. Part 8 measures how much of that advantage survives on the untouched test works.

## Why we did not retrain on all 44 works

An earlier plan promised a final model trained on every work. We withdrew it.

Adding the validation and test works would provide 10.6% more text, but it would destroy the only honest evaluation set. The public model would become the one model in the project whose quality could never be measured.

So GP-Thee-11M remains the 39-work model with a real held-out score. Any future all-44 model must be a separate artifact with a different name and a clear warning that it has no held-out evaluation.

## Lessons for product and engineering teams

**Test capabilities directly.** A handful of attractive demos cannot measure memorisation.

**Audit controls for circularity.** A control selected to produce a particular result is not independent evidence.

**Inspect what a metric counts.** “66 copied characters” sounds alarming until you see that they are a cast-list heading.

**Publish protocol deviations.** A smaller secondary experiment can still be useful if its changed scope is explicit.

**Do not sacrifice measurement for a better-looking artifact.** More training data was not worth losing the ability to say how well the release works.

## Takeaways

- The model predicts training text better, but its long exact copies are editorial structure rather than verse.
- Its longest uninterrupted reproduction of Shakespeare's own words is 25 characters.
- Exhaustive scanning found memorised passages that ordinary sampling usually missed.
- The formal preregistered verdict landed between yes and no; the complete evidence was published instead of changing the rule.
- GP-Thee-11M is the best eligible run by a rule fixed in advance, and it remains a measurable 39-work model.

[← Part 6: choosing the tokenizer — simplified](06-which-tokenizer-simplified.md) · [Next: Part 8, the final test — simplified →](08-the-final-test-simplified.md)
