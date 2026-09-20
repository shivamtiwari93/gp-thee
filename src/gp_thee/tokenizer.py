"""Tokenizers: turning text into the integers a model reads, and back.

Two kinds, both learned from Shakespeare's text and nothing else:

  CharTokenizer   one token per character. Nothing to learn except which characters exist.
  BPETokenizer    byte-pair encoding: start from characters, then repeatedly glue together the pair of
                  neighbouring tokens that occurs most often. After enough merges, common words are single
                  tokens and rare words are a few pieces. Written from scratch here; no library, and no
                  vocabulary borrowed from the internet.

Both share one special token, START, which marks the beginning of a work. It is not text and can never
come out of `encode`. It is placed before every work, the first included, when the corpus is turned into
one long stream, so the model learns "after START comes a title". When the model produces START itself,
the work has ended.

Both are exact: decode(encode(text)) == text for any text made of known characters. There is no
"unknown" token. A character the tokenizer has never seen is an error, not a guess, and so is an id that
is not in the vocabulary.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

START = "<|start|>"

# ---------------------------------------------------------------------------------------------- chunks
# BPE merges are learned, and applied, inside chunks and never across them. We measured what happens
# without this rule, on the training works at 2,048 entries: the very first merge is "e" + " " (a letter
# glued to the space AFTER it), about one piece in eight straddles two words (", and "), and the same
# word ends up tokenized many different ways depending on its neighbours. Without the rule the text
# compresses slightly better. With it, a word nearly always looks the same to the model, which is what we
# want it to learn from. The chunks are:
#
#    a word, with the space before it      " the"  " lov’d"  " ’tis"  "HAMLET"
#    a number, with the space before it    " 2"
#    a run of punctuation, likewise        ","  "._]"  " [_"
#    a run of newlines                     "\n"  "\n\n"      (a blank line: before a speech, four times in five)
#    spaces of indentation                 "   "   (all but the last, which belongs to the next word)
#
# The apostrophe counts as a letter, because in this text it nearly always is one: ’tis, o’er, lov’d, th’.
# `[^\W\d_]` is how to say "a letter, in any alphabet" with Python's standard `re` module. (Strictly:
# anything alphanumeric that is not a decimal digit or an underscore. In our alphabet that means letters.)
CHUNK = re.compile(r" ?(?:[^\W\d_]|’)+| ?\d+| ?(?:[^\w\s’]|_)+|\n+| +(?= )| ")


def chunks(text: str) -> list[str]:
    """Split text into chunks. Joining them gives back the text exactly."""
    pieces = CHUNK.findall(text)
    if sum(map(len, pieces)) != len(text):  # cannot happen with the text we have; a tab would do it
        raise ValueError("text contains a character the chunk rule does not cover (a tab, for example)")
    return pieces


# ---------------------------------------------------------------------------------------------- tokenizers
class Tokenizer:
    """What both tokenizers share: a vocabulary, START, exact decoding, saving and loading."""

    kind = "base"

    def __init__(self, chars: list[str], merges: list[tuple[str, str]] = ()):
        if chars != sorted(set(chars)) or any(len(ch) != 1 for ch in chars):
            raise ValueError("chars must be a sorted list of distinct single characters")
        if set(START) <= set(chars):  # "<", "|" and ">" are not in Shakespeare's alphabet, so START cannot be typed
            raise ValueError("START can be spelled with the alphabet, so text could be mistaken for it")
        self.chars = list(chars)
        self.merges = [tuple(pair) for pair in merges]
        # The vocabulary, in id order: every character, then every merged piece in the order it was learned,
        # then START. Character ids are therefore the same in every tokenizer.
        self.vocab = self.chars + [left + right for left, right in self.merges] + [START]
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("two vocabulary entries are the same string")
        self.id_of = {token: i for i, token in enumerate(self.vocab)}
        self.start_id = self.id_of[START]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int], show_start: bool = False) -> str:
        """Ids back to text. START is not text, so it is left out unless show_start is set."""
        ids = [int(i) for i in ids]
        bad = [i for i in ids if not 0 <= i < len(self.vocab)]  # without this, Python would read -1 as "the last entry"
        if bad:
            raise ValueError(f"ids outside the vocabulary of {len(self.vocab)}: {bad[:5]}")
        return "".join(self.vocab[i] for i in ids if show_start or i != self.start_id)

    def encode_works(self, works: list[str]) -> list[int]:
        """Several works as one stream of ids, with START before each."""
        stream = []
        for work in works:
            stream.append(self.start_id)
            stream.extend(self.encode(work))
        return stream

    def _check_characters(self, text: str) -> None:
        unknown = sorted(set(text) - set(self.chars))
        if unknown:
            raise ValueError(f"characters not in the vocabulary: {unknown!r}. Nothing outside the corpus's alphabet can be encoded.")

    def save(self, path: Path, fitted_on: dict) -> None:
        document = {"kind": self.kind, "vocab_size": self.vocab_size, "start_token": START, "start_id": self.start_id, "fitted_on": fitted_on,
                    "chars": self.chars, "merges": [list(pair) for pair in self.merges]}
        Path(path).write_bytes((json.dumps(document, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


class CharTokenizer(Tokenizer):
    kind = "char"

    def __init__(self, chars: list[str]):
        super().__init__(chars)

    def encode(self, text: str) -> list[int]:
        self._check_characters(text)
        return [self.id_of[ch] for ch in text]


class BPETokenizer(Tokenizer):
    kind = "bpe"

    def __init__(self, chars: list[str], merges: list[tuple[str, str]]):
        super().__init__(chars, merges)
        self.rank = {pair: i for i, pair in enumerate(self.merges)}  # lower rank = learned earlier = applied first
        self._cache: dict[str, list[int]] = {}

    def encode(self, text: str) -> list[int]:
        self._check_characters(text)
        ids = []
        for chunk in chunks(text):
            if chunk not in self._cache:
                self._cache[chunk] = [self.id_of[piece] for piece in self._merge(chunk)]
            ids.extend(self._cache[chunk])
        return ids

    def _merge(self, chunk: str) -> list[str]:
        """Replay the learned merges on one chunk, earliest-learned first, exactly as training applied them."""
        pieces = list(chunk)
        while len(pieces) > 1:
            rank, i = min((self.rank.get(pair, len(self.rank)), i) for i, pair in enumerate(zip(pieces, pieces[1:])))
            if rank == len(self.rank):  # no neighbouring pair is one we ever merged
                break
            pieces[i:i + 2] = [pieces[i] + pieces[i + 1]]
        return pieces


def load(path: Path) -> Tokenizer:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document["kind"] not in ("char", "bpe") or (document["kind"] == "char" and document["merges"]):
        raise ValueError(f"{path}: not a tokenizer file this code understands")
    merges = [tuple(pair) for pair in document["merges"]]
    tokenizer = CharTokenizer(document["chars"]) if document["kind"] == "char" else BPETokenizer(document["chars"], merges)
    # The file says what it should build. If what we built differs, the file and this code disagree.
    if (document["start_token"], document["start_id"], document["vocab_size"]) != (START, tokenizer.start_id, tokenizer.vocab_size):
        raise ValueError(f"{path}: the file's START or vocabulary size does not match what it builds")
    return tokenizer


# ---------------------------------------------------------------------------------------------- learning BPE
def learn_merges(texts: list[str], how_many: int) -> list[tuple[str, str]]:
    """Learn up to `how_many` merges from a list of texts (one per work).

    The loop is: count every neighbouring pair of tokens, merge the most frequent pair everywhere, repeat.
    Done naively that recounts five million characters for every merge. Two standard shortcuts make it quick:

      * Work on distinct chunks, not the running text. " the" occurs tens of thousands of times but is
        merged once; its count is simply weighted by how often it occurs.
      * After a merge, only the chunks that contained that pair can have changed, so only their pairs are
        recounted. `found_in` remembers which chunks have contained which pair. Entries are never removed,
        so a chunk may be listed under a pair it no longer holds. Visiting it is harmless: its pairs are
        taken out of the counts and the same pairs are put back.

    Pairs are counted wherever they occur, overlaps included: "aaa" counts (a, a) twice although it can
    only be merged once. That is how the reference implementations count, and it only matters for runs of
    spaces. Learning stops when the best count falls below 2.

    Ties are broken by taking the pair that sorts last, so the result never depends on dictionary order.
    Ties are not a corner case. On this corpus they decide about three merges in four, because inside a
    name such as SYRACUSE every neighbouring pair occurs exactly as often as the name does. The last
    entries of any vocabulary size are therefore an arbitrary pick among equally frequent pairs.
    """
    chunk_counts: Counter = Counter()
    for text in texts:
        chunk_counts.update(chunks(text))
    words = [list(chunk) for chunk in chunk_counts]  # each distinct chunk, as a list of its current tokens
    times = list(chunk_counts.values())               # how often each occurs in the text

    pair_count: Counter = Counter()
    found_in: dict[tuple[str, str], set[int]] = defaultdict(set)
    for w, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_count[pair] += times[w]
            found_in[pair].add(w)

    merges = []
    while len(merges) < how_many and pair_count:
        best = max(pair_count, key=lambda pair: (pair_count[pair], pair))
        if pair_count[best] < 2:
            break
        merges.append(best)
        for w in found_in.pop(best):
            word = words[w]
            for pair in zip(word, word[1:]):  # take this chunk's pairs out of the counts...
                pair_count[pair] -= times[w]
                if pair_count[pair] <= 0:
                    del pair_count[pair]
            merged, i = [], 0
            while i < len(word):              # ...merge `best` wherever it occurs, left to right...
                if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                    merged.append(word[i] + word[i + 1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            words[w] = merged
            for pair in zip(merged, merged[1:]):  # ...and put its new pairs back in.
                pair_count[pair] += times[w]
                found_in[pair].add(w)
    return merges
