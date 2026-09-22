# Building GP-Thee, part 8: the one-time final test — simplified

*Three works stayed locked away through the entire project. We opened them once to learn how GP-Thee performs on Shakespeare that influenced no decision.*

> This is the simplified edition. The [full technical version](08-the-final-test.md) contains the complete one-way-door protocol, every score, longer model samples, probability probes, and the final audit history.

## What you will learn

- Why a held-out test set can only remain independent until it influences a decision.
- How we designed a one-time evaluation that could fail honestly.
- What GP-Thee's 1.80 bits-per-character score means.
- Which earlier conclusions survived on new text.
- What the model can imitate, where it fails, and why it is not simply replaying its training set.

## Why the test could happen only once

Three works were set aside before any model was trained:

- *The Life and Death of King John*;
- *The Tempest*;
- *A Lover's Complaint*.

The code refused to load them during development. They did not choose the tokenizer, training length, checkpoint, memorisation threshold, or released run.

That independence disappears as soon as a result changes a decision. If we looked at the test score, adjusted something, and tested again, the test works would become another validation set.

So the final evaluation had one job: collect every planned measurement in one controlled invocation, then never run again.

This is the same principle product teams use for launch metrics, security exercises, and regulated acceptance tests. If the result can trigger iteration, the measurement is part of development—not an independent final check.

## The one-way door

The final-evaluation script was built around three stages.

### Before the door: refuse drift and rehearse everything

The script would not start unless:

- `docs/release.json` was committed and unchanged, fixing which run had been selected;
- the relevant source, scripts, and data were unchanged;
- every required checkpoint existed, and the released checkpoint matched its recorded checksum;
- no final result file already existed.

It then rehearsed the entire pipeline on the validation works, where failure was still affordable. Twenty-one checkpoints were rescored. Their validation results matched the recorded training results exactly on this machine, and all baseline calculations and output structures were exercised.

A line-by-line pre-run audit caught a serious bug before the test works opened: code building the final headline treated a list like a dictionary. The rehearsal had not exercised that function, so it was extended to run the complete output path. Without the audit and extension, the one real run would have opened the works and then crashed before producing a usable result.

### The door: record that the test has been spent

Before loading the test works, the script created the final result file with an `in progress` marker. If the run died, that marker would remain. A second invocation would refuse to start.

Failure would therefore be visible, not erased to create another attempt.

### After the door: isolate failures and preserve evidence

Every measurement ran inside its own guarded block. One failure would be recorded while the remaining blocks continued.

In the run that actually happened, all 21 planned model evaluations succeeded and their raw per-token evidence was saved. A later audit found that each evaluation calculated two summaries before saving that evidence. Nothing was lost, but a failure at that point could have erased one model's raw results. The code was hardened afterwards to save immediately after scoring, and a regression test now forces the summary calculation to fail and checks that the raw file still exists.

That hardening does not rewrite history or justify another run. The final record contains the exact commit and script hash that produced the one real result.

## Inspect the final evidence without reopening the test

Do **not** run `scripts/final_evaluation.py`. The one permitted invocation has already happened, and its result is committed. The commands in this section only read that sealed record or run demonstrations against the released weights; none invokes the final-evaluation gate.

From the repository root, this read-only query reconstructs the two headlines, the per-work scores, the baseline and tokenizer comparisons, the test memorisation result, the released seed's rank, and the best-versus-last checkpoint comparison:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

import numpy as np

record = json.loads(Path("docs/final-evaluation.json").read_text())
artifact = record["headline"]["the_artifact"]
recipe = record["headline"]["the_recipe"]
best = record["arms"]["sweep-char-seed-1/best"]
ngram = record["the_bar_on_this_text"]["6-gram (looks back 5)"]["bits_per_character"]
scan = record["memorisation"]["scan"]
best_bpe = min(row["test_bpc"] for row in record["arms"].values()
               if row["tokenizer"].startswith("bpe"))
surprise = np.load(Path("docs") / best["surprise_file"])
characters = best["taken_apart"]["all"]["characters"]
recomputed = float(surprise.sum() / np.log(2) / characters)

print(f"released artifact: {artifact['test_bits_per_character']:.4f} bpc")
print(f"recomputed from the raw array: {recomputed:.15f} bpc")
print(f"three-run recipe:  {recipe['mean']:.4f} bpc (spread {recipe['spread']:.4f})")
for work in best["taken_apart"]["works"]:
    print(f"{work['title']}: {work['bits_per_character']:.4f} bpc")
print(f"best tested 6-gram: {ngram:.4f}; margin: {ngram - artifact['test_bits_per_character']:.4f}")
print(f"best BPE run: {best_bpe:.4f}; character lead: {best_bpe - artifact['test_bits_per_character']:.4f}")
print(f"released seed's test rank: {recipe['released_rank_on_test']} of {len(recipe['test'])}")
print(f"test memorisation scan: {scan['candidates_of_40_or_more']} candidates of 40+, "
      f"{scan['longest_confirmed']} confirmed")
for seed in (1, 2, 3):
    selected = record["arms"][f"sweep-char-seed-{seed}/best"]["test_bpc"]
    last = record["arms"][f"sweep-char-seed-{seed}/last"]["test_bpc"]
    print(f"seed {seed}: best {selected:.4f}, last {last:.4f}, last-best {last-selected:+.4f}")
PY
```

The raw array should recompute the headline as 1.795786796476007. The rest of the output should give 1.7992 for the three-run recipe, a 0.5222-bit lead over the 6-gram, a 0.0959-bit lead over the best BPE run, first place among the three eligible seeds, zero test candidates of 40 characters, and checkpoint differences of +0.0080, -0.0023, and -0.0013. Those signs explain why selecting the best validation checkpoint helped much less on independent text.

## The headline result

GP-Thee-11M scored **1.7958 bits per character** across the three untouched works. Rounded for public reporting: **1.80 bits per character**. The other preregistered headline—the mean across all three eligible runs—was **1.7992**, with a spread of **0.0088**. That second number describes the selection recipe rather than only its chosen artifact.

Lower is better. The released model had scored 1.7492 on validation, where it was selected.

The per-work results explain much of the difference:

| Test work | Type | Bits per character |
|---|---|---:|
| *King John* | history | **1.6921** |
| *The Tempest* | romance | 1.9044 |
| *A Lover's Complaint* | narrative poem | 1.9240 |
| **All three** |  | **1.7958** |

The easiest and hardest works differ by 0.23 bits per character—larger than any model-to-model difference debated in the project. With only three test works, the type of text matters more than the third decimal place of the overall score.

That is a useful warning for dashboards: a precise aggregate is not the same as a broad sample.

## Is 1.80 good?

The model was compared with simpler predictors on the exact same test text.

| Predictor | Test bits per character |
|---|---:|
| blind guess over 97 characters | 6.60 |
| character frequencies | 4.79 |
| best tested Witten–Bell character n-gram (6-gram) | 2.32 |
| bzip2 | 2.58 |
| xz | 2.91 |
| **GP-Thee-11M** | **1.80** |

The model beats every tested non-neural baseline by at least 0.52 bits per character. Even a compressor allowed to read the training works before compressing the test text reaches only 2.37.

The project's success criterion, written before training, was to beat the n-gram and compression baselines. GP-Thee met it on text that influenced no choice.

## What does “bits per character” mean?

At each position, the model assigns a probability to every possible next character. If the real next character received high probability, the model pays a small penalty. If it was surprised, it pays a large one.

**Bits per character** is the average amount of surprise across the text. Lower means better prediction.

It is not a percentage correct, and 1 bit is not a universal lower bound. One bit is simply the cost of a fair binary choice, a useful reference point. The raw 97-character alphabet costs about 6.6 bits when every character is treated as equally likely.

At 1.80, GP-Thee has learned a great deal about spelling, words, dialogue, punctuation, stage formatting, and local Shakespearean style. The examples below show that local predictive skill is not the same as understanding.

## Four earlier conclusions retested

Before opening the works, four questions were fixed, with both possible outcome branches written down for each. That stopped the final story from being invented after the numbers arrived.

### 1. Does it recite?

Part 7 found long exact copies in training text, but they were mostly cast lists and scene headings. Its validation control was imperfect because those plays had been selected to avoid overlap.

The test works were a cleaner control. The exhaustive scan found **zero candidate runs of even 40 agreed characters** in them. This is a teacher-forced first-choice measurement, not sampled generation. The test works do contain some real editorial overlap with training—26 shared 50-character windows, mostly one accepted scene heading—but the scan found none of it reproduced.

This cleaner control strengthened the evidence against long recitation on unread text. Combined with the training scan, it also preserved the observed pattern: the model's long exact copies came from editorial structure, not long passages of verse.

### 2. Was the released seed really best?

The three eligible seeds scored:

- 1.7958;
- 1.7971;
- 1.8046.

The released seed was first again, but the full spread was only 0.0088. The project's own threshold calls that a tie. The release remains the output of a rule, not proof that seed 1 is intrinsically superior.

### 3. Do characters still beat BPE?

Yes. On the history, romance, and poem in the test set, the character model scored 1.80. The best word-fragment model scored 1.89.

The tokenizer conclusion from Part 6 survived on new kinds of text.

### 4. How much did checkpoint and seed selection help?

On validation, choosing each run's best checkpoint rather than its last checkpoint improved the mean by about 0.0078 bits per character. On test, the mean advantage shrank to 0.0014 and reversed for two of the three seeds.

Choosing the best of the three runs produced a 0.0054 advantage over their mean on validation and a 0.0034 advantage on test.

The direction is exactly what we expected: selection makes the validation number a little optimistic, and part of that advantage disappears on independent text.

## What the model actually behaves like

The model is an **in-character autocomplete**.

Give it a speaker label and it writes speeches, alternates names, inserts blank lines, and creates stage directions. Give it a scene heading and it constructs the visual shape of a printed play.

The structure is convincing. The meaning is not.

In one sample it writes:

```text
Enter Caesar, Caesar, Agrippa and a Capitol
```

The model entered the same person twice and treated a building as a character. Caesar then asked where Caesar was.

Give it an instruction such as `Write me a poem about a cat.` and it does not obey. It continues the sentence as if it were a line in a play. The model was trained on plays, sonnets, and poems—not examples of users asking assistants to perform tasks.

Give it a modern sentence about a meeting and a budget, and within a few words it returns to nobles, battles, and courtly speech. Shakespeare is the only register it has.

### Temperature changes risk, not intelligence

**Temperature** controls how much randomness is used when selecting the next character.

- At 0, the model always takes its first choice. It is safer but often loops.
- Around 0.5, it is cautious and repetitive.
- At the published default of 0.8, it is more varied but loses track of names and roles.
- Above 1, it takes bigger risks and invents more malformed words.

No temperature makes the model coherent over a long scene. It only changes the tradeoff between repetition and nonsense.

## Is it replaying the training plays?

The generated text often *looks* authentic, so this is the right challenge.

For each published sample, we measured the longest exact run that appears anywhere in the 4.8 million training characters. Samples of 200 to 330 characters had longest matches of only 16 to 34 characters. Those matches were common phrases or formatting such as scene headings—not complete poetic lines.

The model also creates forms absent from training:

- `jollities`, extending the seen word `jollity`;
- `dismalled`, turning `dismal` into a verb;
- `adoxing`, an English-shaped invention.

It also produced new phrases such as `naked battle gates` and `a cold rotten for some monarch`.

That is evidence of composition, even when the composition is bad.

The stronger comparison is numerical. A 6-gram is a real fragment lookup system with a fallback for unseen contexts. Among the Witten–Bell character n-grams tested at orders 1 through 8, the 6-gram was best at 2.32. GP-Thee scored 1.80. The difference shows that this neural model learned more than that tested family of fragment tables. It is not a claim about every lookup method anyone could build.

## Famous lines are not stored as reliable responses

Three illustrative probes asked whether famous lines were easy for the model to retrieve:

- At the normal sampling temperature of 0.8, a prompt beginning Hamlet's soliloquy continued `that is the` with `tender of it`, not `question`.
- After `KING HENRY.\nOnce more unto the b`, a next-character probability check ranked the correct `r` fourth at 14.6%. At temperature 1, the full continuation `reach` has about a 1.10% chance—roughly one draw in 91.
- Given the exact corpus cue before `Et tu, Brute?`, a greedy continuation produced `I am sorry for you.` and repeated it.

These were different kinds of probe, not three identical greedy trials. None retrieved the famous continuation in that attempt.

Three examples cannot prove that a line is absent. The systematic evidence comes from Part 7's scan of every training position, where the longest uninterrupted stretch of Shakespeare's own words reproduced by the model was 25 characters.

The model learned the language the famous lines are made from, not a dependable database of the lines themselves.

### Inspect and reproduce these examples

The committed [structured demonstration record](../docs/demonstrations.json) contains every prompt, seed, temperature, continuation, and longest copied run used in this post. The [readable record](../docs/demonstrations.txt) contains the complete samples, including text shortened on this page. This query gives a compact inventory without running the model:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

record = json.loads(Path("docs/demonstrations.json").read_text())
for row in record["what_it_writes"]:
    first = row["continuation"].strip().splitlines()[0]
    print(f"{row['label']}: temperature={row['settings']['temperature']}, seed={row['settings']['seed']}, "
          f"longest copy={row['longest_verbatim']['characters']}, first line={first!r}")
famous = record["the_famous_line"]
print("\nCaesar cue:", repr(famous["run_up"]))
print("really follows:", repr(famous["what_really_follows"].splitlines()[0]))
print("model wrote:", repr(famous["what_the_model_wrote"].splitlines()[0]))
PY
```

To reproduce the three famous-line probes live, run this against the CPU release. It uses Hugging Face's normal download cache, no held-out text, and does not alter any evidence file:

```bash
uv run python -B - <<'PY'
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, "release/gp-thee-11m")
from load_release import load
from gp_thee.sampling import generate
from gp_thee.tokenizer import CharTokenizer

revision = "352de5f38ddf55cedbe3ac7712dd5321ecedd4cb"
folder = Path(snapshot_download("shivamtiwari93/gp-thee-11m", revision=revision))
model, document = load("cpu", folder)
tokenizer = CharTokenizer(document["chars"])

hamlet = generate(model, tokenizer, "\n\nHAMLET.\nTo be, or not to be, that is the",
                  characters=80, temperature=0.8, seed=0)
print("HAMLET:", hamlet["continuation"].splitlines()[0])

henry = "\n\nKING HENRY.\nOnce more unto the b"
ids = tokenizer.encode(henry)
with torch.no_grad():
    probabilities = torch.softmax(model(torch.tensor([ids]))[0][0, -1].float(), dim=-1)
ranked = torch.topk(probabilities, 4)
print("HENRY:", [(tokenizer.decode([int(i)]), round(float(p), 4))
                 for p, i in zip(ranked.values, ranked.indices)])

probability = 1.0
for character in "reach":
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.tensor([ids]))[0][0, -1].float(), dim=-1)
    token = tokenizer.id_of[character]
    probability *= float(probabilities[token])
    ids.append(token)
print("P(reach):", f"{probability:.6%}", "one in", f"{1/probability:.1f}")

caesar = "and at last by Marcus\nBrutus._]\n\nCAESAR.\n"
greedy = generate(model, tokenizer, caesar, characters=40, temperature=0)
print("CAESAR:", greedy["continuation"].splitlines()[0])
PY
```

On the released CPU model, this prints `tender of it`, ranks Henry's correct `r` fourth at 0.1461, gives `P(reach)` as 1.095970% (about one in 91.2), and makes Caesar say `I am sorry for you.` Fixed seeds make the CPU command repeatable on the same software stack; stochastic continuations can still differ across devices or library versions.

Contributors who possess `runs/sweep-char-seed-1/best.pt` can also run `uv run python scripts/demonstrate.py` to regenerate every demonstration at once. A fresh clone cannot: `.pt` checkpoints are intentionally not committed. The script rewrites `docs/demonstrations.json` and `.txt`, so run it in a disposable copy when comparing its output with the committed record.

## What GP-Thee cannot do

The limitations are as important as the score:

1. **It cannot reliably follow instructions.** It has never trained on that interaction pattern.
2. **It cannot reliably answer questions.** It usually continues the prompt as text instead.
3. **It does not track who is on stage.** Names drift and characters answer themselves.
4. **It loses meaning across sentences.** Its context is 256 characters—about six median non-blank lines—and it has no persistent plot, goal, or state.
5. **It received no outside training data.** Do not rely on it for facts, arithmetic, or knowledge of the modern world.
6. **It supports only 97 characters.** Unsupported characters are refused rather than silently changed.
7. **It loops when made deterministic.** Randomness reduces repetition but adds mistakes.

These are not surprising defects in a general assistant. They are the natural limits of an 11-million-parameter model trained from scratch on five megabytes from one literary universe.

## Lessons for product and engineering teams

**Protect the final metric from product iteration.** Once a test result changes a decision, it is no longer independent.

**Rehearse the full output path, not only the expensive computation.** The bug that nearly ruined the run lived in headline assembly, not model scoring.

**Keep raw evidence.** Summary metrics can be recomputed; an irreplaceable forward pass cannot.

**Show behaviour beside the aggregate.** A score of 1.80 sounds impressive. Samples reveal that the model has excellent form and almost no sustained meaning.

**Describe selection honestly.** The chosen checkpoint and seed received a small validation advantage that mostly faded on test.

**State the boundary of each comparison.** GP-Thee beat the baselines tested here. That does not establish a universal ceiling for other methods.

## Takeaways

- GP-Thee-11M scored **1.7958 bits per character** on three untouched works.
- It beat every tested n-gram and compressor baseline by at least 0.52 bits per character.
- The character-tokenizer result survived independent text, while the clean test control strengthened the evidence against long recitation without changing Part 7's formal “between” verdict.
- Seed and checkpoint selection made validation look slightly better than test, as expected.
- The model is a convincing Shakespearean autocomplete, not an assistant or a source of coherent stories.
- The locked test set, rehearsal, raw evidence, and refusal to rerun are as important as the final number.

GP-Thee-11M is available [on Hugging Face](https://huggingface.co/shivamtiwari93/gp-thee-11m). The complete evidence is in [the final evaluation record](../docs/final-evaluation.json), with the full chronology in [the build log](../docs/BUILD_LOG.md).

[← Part 7: does it recite? — simplified](07-does-it-recite-simplified.md)
