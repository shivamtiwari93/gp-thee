# Building GP-Thee, part 3: teaching a computer to read — simplified

*Before a language model can learn Shakespeare, the text must become numbers. This part builds that translation layer and shows why a seemingly small design choice changes what the model can see and learn.*

> This is the simplified edition. For the implementation details, full measurements, and audit trail, read [the original Part 3](03-the-tokenizer.md).

## What you will learn

- What a tokenizer does, and why vocabulary size is a product decision as much as a technical one.
- Why we tested both characters and learned word fragments.
- What we gained and gave up by preventing fragments from crossing word boundaries.
- How data leakage and weak tests can make a tokenizer look trustworthy when it is not.
- How we planned a fair comparison before training any model.

## From text to numbers

A neural network cannot work directly with the letter `T`. It works with numbers. A **tokenizer** supplies the agreement between the two: a piece of text maps to an integer, and the integer maps back to the same piece of text.

The list of pieces is the **vocabulary**. Turning text into integers is **encoding**; reversing the process is **decoding**. A vocabulary entry is a **piece**, while each occurrence of that piece in a text is a **token**. For example, the piece ` the` appears 22,724 times as a token under each of the BPE vocabularies built here: 22,693 times as a standalone word and 31 times inside longer words.

This project has a strict data rule. Anything learned from text must be learned from the 39 training works, not from the two validation works or the three test works. That includes the tokenizer. Borrowing GPT-2's 50,257-piece vocabulary would import knowledge learned from millions of web pages, so GP-Thee builds its own.

## Start with the simplest option

The character tokenizer sorts the 97 distinct characters in the training works and gives each one a number:

```
"To be"  →  [41, 65, 1, 52, 55]
```

This is simple, reversible, and hard to misunderstand. It also makes the model do more work. GP-Thee can look back over 256 tokens. With one character per token, that means 256 characters, or roughly six median-length non-blank lines. The model must also spell every word one character at a time.

We made two related policy choices.

First, there is no “unknown” token. The project has a closed alphabet, so valid text can always be reconstructed exactly. A character outside that alphabet is rejected instead of silently replaced. This is stricter than a byte-level tokenizer, but it avoids reserving 157 untrained byte rows and avoids letting the model emit incomplete multi-byte characters. It also means user-facing code must normalize prompts: Shakespeare's source contains the curly apostrophe `’`, not the straight keyboard apostrophe `'`.

Second, every work begins with one special piece, `<|start|>`. It teaches the model that a title follows the beginning of a work and gives sampling code a way to request a new work. It is deliberately impossible to type with the corpus alphabet, so ordinary text cannot be confused with it. The first safety check for this was wrong: it rejected an alphabet containing *any* character from the marker, including common letters such as `s` and `t`. The real requirement is only that at least one marker character be absent. The corrected check enforces that.

There are therefore 98 character pieces: 97 text characters plus START. With 384 numbers per vocabulary row, START adds 384 parameters, bringing the character model to 10,757,760 parameters.

START is useful but poorly trained. It appears before 39 works, yet the stream contains only 38 examples of one work ending before another begins. The sampler will not trust the model to stop when it produces START.

## Let the text discover useful fragments

The alternative is **byte-pair encoding**, or BPE. Despite the name, this project uses the text version introduced for machine translation: it starts from characters and repeatedly joins the most frequent neighbouring pair.

For a small example, begin with:

```
a a a b d a a a b a c
```

The most frequent pair is `a a`, so it becomes one piece. Recount, merge the next winning pair, and repeat. The learned merge list is the tokenizer. Encoding new text starts from characters and replays that list in order; it does not learn anything new.

We built vocabularies of 1,024, 1,536, 2,048, and 4,096 pieces. All begin with the same 98 base pieces, so a 4,096-piece tokenizer learns 3,998 merges. The smaller vocabularies are prefixes of that same merge list.

The results look recognizably Shakespearean. Early merges produce pieces such as ` the`, ` and`, ` thou`, ` lord`, ` love`, and eventually `HAMLET`. Nobody supplied a dictionary or a cast list. Frequency did the work.

That gives the model more reach. An 84-character passage beginning “To be, or not to be” takes 84 character tokens, but only 28 tokens with the 2,048-piece vocabulary. A 256-token window then covers about 727 characters, around eighteen lines instead of six.

## The boundary rule: less compression, more consistency

Pure BPE is free to merge across spaces and line breaks. We chose not to let it.

Before learning merges, the text is divided into **chunks**: a word with its leading space, a punctuation run, a line-break run, or indentation. Merges may happen inside a chunk but never across chunks. This step is called **pre-tokenization**.

The space goes before a word because that is the more stable form in this corpus: 84% of words have a leading space, while 73% have a trailing one. The apostrophe counts as a letter because Shakespeare frequently uses forms such as `’tis`, `o’er`, `ne’er`, `lov’d`, and `th’`. Treating it as punctuation would split those forms unnaturally. Closing quotation marks use the same character, but only about 75 to 150 of the 23,019 training apostrophes are closing quotes attached to a word—at most about half a percent.

The boundary rule has a real cost. At 2,048 pieces, a tokenizer without it compresses both training and validation text better:

| Measure | No boundary rule | With the rule |
|---|---:|---:|
| Characters per token, training | 3.03 | 2.84 |
| Characters per token, validation | 3.00 | 2.79 |
| Merged pieces seen at least 100 times | 96.6% | 91.2% |
| Pieces crossing two words | 237 | 0 |
| Pieces mixing line breaks with text | 204 | 0 |
| Average cuts for 150 common words | 12.4 | 2.0 |
| Different cuts for `come` | 45 | 2 |

Without the rule, a 256-token window would cover about 775 characters rather than 727. Its vocabulary is also used more often. On compression alone, it wins.

We kept the rule because consistency may matter more with only about 5 MB of training text. Without boundaries, `come` can reach the model in 45 different fragmentations depending on neighbouring words. With boundaries, it is normally ` come`, or `come` at the start of a line. This is a bet, not a proven result: the four commonest unconstrained forms cover three quarters of the occurrences, and only training both versions could settle the question. The useful lesson is to state the tradeoff honestly. We sacrificed about 7% of context coverage for a much more stable representation of words.

The first explanation in the code did not meet that standard. It claimed, without measuring, that unconstrained BPE would merge a particular cross-word pair. An auditor ran the experiment and showed that example never occurred. The broader concern was real, but the example was invented. The measured comparison above replaced it.

## Keeping held-out text held out

The alphabet and all 3,998 merges are learned from the 39 training works. Validation works are encoded only after the tokenizer exists. Test works are not opened at all.

That separation is visible in the output. With 2,048 pieces:

```
HAMLET.  →  HAMLET | .
ROMEO.   →  RO | M | E | O | .
```

*Hamlet* is a training work, so its title appears often enough to become a piece. *Romeo and Juliet* is held out for validation, so `ROMEO` must be assembled from fragments learned elsewhere. That is not a defect. It is what honest evaluation looks like.

The repository has a sensitivity check: relearning on the training works must reproduce the saved merge list exactly. Adding only the validation works changes merge number 33 and changes 3,848 of the 3,998 positions, so the check is capable of detecting leakage.

The project still leaked test information three times in prose and analysis. Part 2 quoted test-derived statistics. A draft of this part fitted a tokenizer on all 44 works for a sensitivity example, then published numbers derived from the test works. It also made the test share of apostrophes recoverable by subtraction. None of those figures changed a model, but repeated slips showed that intention was not enough.

The fix was operational. Downstream code loads works through one function that refuses test access unless the caller explicitly declares final evaluation. The two scripts that must operate before that boundary are documented exceptions. Token files are not saved for test works because even their file sizes would reveal information. Validation data now stands in for deliberate leakage tests.

## Choosing a vocabulary size

Larger vocabularies provide longer context but cost parameters and divide a small dataset among more rows:

| Tokenizer | Vocabulary | Characters per token, train / validation | Approximate reach of 256 tokens | Merged pieces seen 100+ times | Parameters |
|---|---:|---:|---:|---:|---:|
| Characters | 98 | 1.00 / 1.00 | 256 characters | — | 10,757,760 |
| BPE | 1,024 | 2.48 / 2.49 | 636 characters | 98.3% | 11,113,344 |
| BPE | 1,536 | 2.69 / 2.66 | 689 characters | 95.0% | 11,309,952 |
| BPE | 2,048 | 2.84 / 2.79 | 727 characters | 91.2% | 11,506,560 |
| BPE | 4,096 | 3.19 / 3.07 | 817 characters | 50.6% | 12,292,992 |

At 4,096 pieces, half of the merged pieces occur fewer than 100 times, and 125 never appear in the final training encoding because longer pieces swallow them. Their embedding rows have little chance to learn a useful meaning. A research rule of thumb points near 1,536 pieces after adapting it to exclude rare base characters and START, but that rule came from machine translation. It tells us where to look, not what to choose.

Speaker names are another complication. All-capital pieces make up 9% of the 1,024-piece vocabulary, 13% at 2,048, and 16% at 4,096. They improve compression on the training cast but not on unseen validation casts. Once speaker-label lines are removed, the train-versus-validation compression gap at 2,048 disappears.

So the table will not choose the winner. Each tokenizer gets the same model architecture and is trained until its own best validation checkpoint. The comparison uses **bits per character**: total surprise for the same text divided by its character count. Per-token loss would be unfair because larger pieces create fewer, harder predictions. The deciding result is validation bits per character averaged over three random starts. If two choices are closer than the run-to-run spread, the smaller vocabulary wins. Scores for speaker labels, ordinary lines, and each play will explain the result but will not select it.

## Fast enough to experiment, checked enough to trust

A direct implementation would recount every adjacent pair in 4.8 million characters before each of 3,998 merges. In plain Python that takes most of an hour.

Two optimizations reduce it to under six seconds. First, work on the 38,398 distinct chunks instead of all 1.2 million occurrences, while retaining each chunk's frequency. Second, after a merge, recount only chunks that could have changed.

Those shortcuts create room for subtle bugs, so a slower reference implementation remains in the tests. The fast and slow versions agree on 200 random strings and the first 120 Shakespeare merges; the audit also ran all 3,998 merges independently and got the identical ordered list.

The audit found that the original 28 tests were still too weak. An encoder that ignored every merge after number 1,500 passed. So did one that inserted stray START tokens, because the decoder hid START before the round-trip comparison. A round trip proves that no characters were lost; it does not prove that the intended pieces were used.

The suite grew to 51 tests. It now checks exact token sequences, published token counts, START visibility, saved files, and agreement with the slow implementation. The operational lesson is broad: test the representation itself, not just whether encoding and decoding happen to cancel each other's mistakes.

## Try it: build and inspect the tokenizers

Build all five tokenizers from the 39 training works:

```bash
uv run python scripts/build_tokenizers.py
```

The merge learning took about six seconds on the project machine. The command replaces the generated definitions under `data/tokenizers/` and the train/validation arrays under `data/tokens/`; it does not write a test array. It should say that the alphabet has **97 characters**, that it learned **3,998 merges**, and that BPE-2048 turns the training corpus into about **1.69 million tokens** at **2.84 characters per token**. No test work is opened.

Now inspect one sentence through both representations:

```bash
uv run python - <<'PY'
from pathlib import Path
from gp_thee.tokenizer import load

text = "To be, or not to be"
for name in ("char", "bpe-2048"):
    tokenizer = load(Path(f"data/tokenizers/{name}.json"))
    ids = tokenizer.encode(text)
    pieces = [tokenizer.vocab[i].replace("\n", "\\n") for i in ids]
    print(f"{name}: {len(ids)} tokens")
    print(" | ".join(pieces))
    print("round trip:", tokenizer.decode(ids) == text)
PY
```

This inspection is read-only, operates only on the supplied sentence and saved tokenizer definitions, and takes well under a second. Expected output:

```text
char: 19 tokens
T | o |   | b | e | , |   | o | r |   | n | o | t |   | t | o |   | b | e
round trip: True
bpe-2048: 7 tokens
To |  be | , |  or |  not |  to |  be
round trip: True
```

The leading spaces on pieces such as ` be` are intentional: they demonstrate the boundary rule rather than a display error. Finally, run the tokenizer-specific checks:

```bash
uv run pytest tests/test_tokenizer.py -q
```

The suite took about 20 seconds on the project machine and does not open the test works. Its current observable result is `51 passed`. Together these exercises test three different claims: the tokenizers can be rebuilt from training data, their representation is visible and reversible, and the implementation agrees with its pinned examples and slower reference.

## Takeaways

- A tokenizer determines both what the model can express and how much text fits in its context window.
- Character tokenization is simple and data-efficient; BPE provides roughly 2.5 to 3.2 characters per token but creates rarer, more specialized rows.
- The chunk boundary rule trades about 7% of context coverage for much more consistent word forms. It remains an explicit hypothesis.
- Validation and test isolation must be enforced in code. Repeated accidental leaks showed that written policy was not enough.
- Vocabulary size cannot be chosen fairly from compression alone. The real comparison must use validation bits per character, multiple runs, and rules written before results are known.
- Round-trip tests are necessary but not sufficient. Independent implementations, exact expected pieces, and deliberately broken code found failures that ordinary tests missed.

[← Part 2: the data — simplified](02-the-data-simplified.md) · [Next: Part 4, the model — simplified →](04-the-model-simplified.md)
