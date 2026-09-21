"""Evaluation: what the reported numbers are made of, and three baselines that give them a meaning.

train.evaluate() produces one number per token: the model's surprise at it, in nats. Every figure we report
is made from those numbers the same way: add up the surprise over some part of the text, turn nats into bits,
divide by the number of CHARACTERS in that part. Characters, not tokens: a tokenizer with bigger pieces makes
fewer, harder predictions, and only the per-character figure can be compared between tokenizers.

A score on its own says little. "1.9 bits per character" is good or bad only next to something, so this
file also holds three predictors that contain no neural network at all, fitted on the same training works
and scored on the same validation works:

  * character counts       knows how common each character is, and nothing else
  * character n-grams      knows which character tends to follow the last few (the classic pre-neural model)
  * file compressors       bzip2 and xz: a compressed size in bits IS a prediction score, by another road
"""

import bz2
import lzma
import math
import re

import numpy as np

# A speaker label is a line of its own, in capitals, ending in a full stop, after a blank line and before the
# speech:  "\n\nHAMLET.\nTo be, or not to be"  (also "FIRST LORD.", "2 KEEPER.", "PRINCE & POINS.")
LABEL_SHAPED = r"[^a-z\n]*[A-Z][^a-z\n]*\.\n"
LABEL = re.compile(r"(?<=\n\n)" + LABEL_SHAPED + r"(?=[^\n])")
ANOTHER = re.compile(LABEL_SHAPED)


# ---------------------------------------------------------------------------------------------- the model's score, taken apart
def piece_lengths(tokenizer) -> np.ndarray:
    """How many characters each piece of the vocabulary stands for. START stands for none."""
    return np.array([0 if piece == tokenizer.vocab[tokenizer.start_id] else len(piece) for piece in tokenizer.vocab])


def speaker_label_characters(work: str) -> np.ndarray:
    """True for every character of `work` that sits on a speaker-label line (the line's own newline included).

    A cast list is a run of label-shaped lines one after another. Those are not labels, so a label-shaped line
    that is followed by another one is left out.

    The rule is simple and not perfect. On the 41 training and validation works it marks 30,538 lines, and an
    auditor's looser search found about 30 mistakes, all in training works and none in the validation works:
    joint labels with a small word ("ROSENCRANTZ and GUILDENSTERN.", 23 of them), a label straight under a stage
    direction with no blank line between, a speech whose whole first line is "I.", and three headings taken for
    labels ("       THE SONG.", and "KING HENRY V." at the top of a cast list). It is frozen as it is: it cannot be
    tuned on the test works, which nobody has looked at, and tests/test_evaluation.py pins these cases so that
    a later edit cannot move them unnoticed.
    """
    on_label = np.zeros(len(work), dtype=bool)
    for found in LABEL.finditer(work):
        if not ANOTHER.match(work, found.end()):
            on_label[found.start():found.end()] = True
    return on_label


def score(nats: float, characters: int, tokens: int) -> dict:
    return {"characters": int(characters), "tokens": int(tokens), "bits": float(nats / math.log(2)),
            "bits_per_character": float(nats / math.log(2) / characters) if characters else None}


def breakdown(surprise: np.ndarray, stream: np.ndarray, tokenizer, works: list[str]) -> dict:
    """Split the total score by work, and into speaker labels against everything else.

    `surprise[i]` is the surprise at token i + 1 of `stream` (the first token is given, not predicted).
    Every part is scored like the whole: its nats, over ln 2, over its characters. The parts add up to the
    whole exactly, and this function checks that they do.

    Conventions, all tiny in effect. A START token stands for no characters; its surprise is counted with the work
    it opens, and with "everything else". (So the first work never pays for one, and nobody pays for knowing that
    the last work has ended.) The blank line BEFORE a label, which is the decision "this speech is over", counts as
    "everything else".

    A piece is either all line breaks or has none: the tokenizer's chunk rule keeps a run of line breaks apart from
    text. A run can be one piece ("\n\n" is a single token in the word-fragment vocabularies), but a label line is
    always followed at once by the speech, so the line break that ends a label is a piece of its own. That is
    what lets us say which pieces belong to a label. That, the places of the START tokens, and the parts adding up
    to the whole are all checked below rather than trusted.
    """
    stream = np.asarray(stream).astype(np.int64)
    text = "".join(works)
    if tokenizer.decode(stream.tolist()) != text:
        raise ValueError("this token stream does not spell these works")
    lengths = piece_lengths(tokenizer)[stream]
    begins = np.concatenate([[0], np.cumsum(lengths)[:-1]])          # where each token begins in `text`
    which_work = np.cumsum(stream == tokenizer.start_id) - 1          # START opens a work
    on_label_character = np.concatenate([speaker_label_characters(work) for work in works])
    on_label = np.zeros(len(stream), dtype=bool)
    real = lengths > 0
    on_label[real] = on_label_character[begins[real]]
    ends_on_label = np.zeros(len(stream), dtype=bool)
    ends_on_label[real] = on_label_character[begins[real] + lengths[real] - 1]
    if (on_label != ends_on_label).any():
        raise AssertionError("a piece straddles the edge of a speaker-label line")

    nats = np.concatenate([[0.0], surprise])                         # the first token costs nothing: it was given
    scored = np.arange(len(stream)) > 0
    part = lambda chosen: score(nats[chosen].sum(), lengths[chosen].sum(), (chosen & scored).sum())
    result = {
        "all": part(np.ones(len(stream), dtype=bool)),
        "works": [{"title": work.split("\n", 1)[0], **part(which_work == i)} for i, work in enumerate(works)],
        "speaker_labels": part(on_label),
        "everything_else": part(~on_label),
    }
    if stream[0] != tokenizer.start_id or [work["characters"] for work in result["works"]] != [len(work) for work in works]:
        raise AssertionError("the START tokens are not where the works begin")
    for parts in (result["works"], [result["speaker_labels"], result["everything_else"]]):
        if sum(p["characters"] for p in parts) != result["all"]["characters"] or abs(sum(p["bits"] for p in parts) - result["all"]["bits"]) > 1e-6 * result["all"]["bits"]:
            raise AssertionError("the parts do not add up to the whole")
    return result


# ---------------------------------------------------------------------------------------------- baselines
class NGram:
    """Predict the next character from the `order - 1` before it, by counting. No network, no training loop.

    With order 1 it knows only how common each character is. With order 5 it looks back four characters.
    The trouble with counting is the unseen: after "Romeo" the training works offer nothing, because they never
    contain the word. So each order's counts are blended with the prediction of the next shorter order, and the
    blend leans on the shorter order more when the context has been seen rarely or has been followed by many
    different characters. This rule is called Witten-Bell smoothing. It has no settings to tune, which matters:
    a baseline tuned on the validation works would have had a look at them.

        P(c | context) = (count(context, c) + kinds(context) x P(c | shorter context)) / (count(context) + kinds(context))

    where kinds(context) is how many different characters ever followed that context. Below order 1 sits the
    blind guess: every character of the alphabet equally likely.
    """

    def __init__(self, order: int, alphabet: list[str]):
        if order < 1 or (len(alphabet) + 1) ** order >= 2 ** 63:
            raise ValueError("an order this long, with this many characters, does not fit in one 64-bit integer")
        self.order, self.ids = order, {ch: i for i, ch in enumerate(alphabet)}
        self.base = len(alphabet) + 1                                # one extra symbol: "before the work began"
        self.tables = []                                             # one per context length 0 .. order - 1

    def _codes(self, work: str, context: int) -> np.ndarray:
        """For every character of `work`: itself and the `context` characters before it, packed into one integer."""
        ids = np.array([self.base - 1] * context + [self.ids[ch] for ch in work], dtype=np.int64)
        return sum(ids[context - back:len(ids) - back] * self.base ** back for back in range(context + 1))

    def fit(self, works: list[str]) -> "NGram":
        self.tables = []
        for context in range(self.order):
            grams, counts = np.unique(np.concatenate([self._codes(work, context) for work in works]), return_counts=True)
            contexts, first = np.unique(grams // self.base, return_index=True)   # grams are sorted, so contexts are too
            self.tables.append({"grams": grams, "counts": counts, "contexts": contexts,
                                "context_counts": np.add.reduceat(counts, first), "kinds": np.diff(np.append(first, len(grams)))})
        return self

    def bits(self, works: list[str]) -> np.ndarray:
        """The surprise, in bits, at every character of `works`."""
        out = []
        for work in works:
            probability = np.full(len(work), 1.0 / (self.base - 1))  # the blind guess
            for context, table in enumerate(self.tables):
                codes = self._codes(work, context)
                count = _look_up(table["grams"], table["counts"], codes)
                context_count = _look_up(table["contexts"], table["context_counts"], codes // self.base)
                kinds = _look_up(table["contexts"], table["kinds"], codes // self.base)
                seen = context_count > 0
                probability = np.where(seen, (count + kinds * probability) / np.where(seen, context_count + kinds, 1), probability)
            out.append(-np.log2(probability))
        return np.concatenate(out)


def _look_up(keys: np.ndarray, values: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """values[i] where keys[i] == wanted, and 0 where `wanted` is not among the (sorted) keys."""
    at = np.minimum(np.searchsorted(keys, wanted), len(keys) - 1)
    return np.where(keys[at] == wanted, values[at], 0)


def compressed_bits_per_character(text: str, after_reading: str = "") -> dict:
    """Bits per character when a file compressor squeezes `text`.

    A compressor is a predictor in disguise: it spends few bits on what it expected and many on what it did not.
    With `after_reading`, the compressor gets that text first and we charge it only for the extra bytes, which is
    the fair comparison with a model that has read the training works. xz at this setting can look back over
    all of them. bzip2 works in blocks of 900,000 bytes and forgets each one, so it only ever sees the tail end
    of the training works next to the validation text. Both still gain from it (measured: 2.80 to 2.39, and 2.48 to 2.30).
    bzip2's figure is partly luck: it depends on where the block edge happens to fall. Shortening the training text
    by 0 to 800,000 bytes moves it between 2.27 and 2.47. xz's hardly moves (2.390 to 2.396).
    """
    before, both = after_reading.encode("utf-8"), (after_reading + text).encode("utf-8")
    squeezers = {"bzip2 -9": lambda data: bz2.compress(data, 9), "xz -9e": lambda data: lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)}
    return {name: 8 * (len(squeeze(both)) - (len(squeeze(before)) if before else 0)) / len(text) for name, squeeze in squeezers.items()}
