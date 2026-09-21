"""Does the model write its training text out word for word?

The question sounds simple and is not, because Shakespeare repeats himself and his editors repeat themselves
far more. So this file holds three things: a way to look a passage up in the training works, a way to tell the
poet's words from the editor's furniture, and the instrument that actually answers the question.

FINDING A COPY. Every run of W characters in the 39 training works is hashed into a sorted table, and any
passage can then be looked up in it. Every hit is confirmed character by character, so a hash collision cannot
invent a copy that is not there.

THE POET AND THE EDITOR. Most of what a Shakespeare file contains was not written by Shakespeare: cast lists,
"ACT I", "SCENE II. A room in the palace.", "[_Exeunt._]", and the speaker labels before every line. That text
is short, formulaic and repeated across works, so it is the easiest thing in the corpus to reproduce and the
least interesting. Every figure here is therefore reported twice, once over everything and once over the poet's
words alone. Neither is the footnote.

THE INSTRUMENT is not sampling. Asking the model for text and looking for copies tells you what it happens to
say; it can miss a passage it knows perfectly well. So the model is walked along the true training text instead
and asked, at every one of 4.8 million positions, whether the character that really comes next is the one it
would have written. A run of such positions is a CANDIDATE. Candidates are then CONFIRMED by making the model
write: from the real 256 tokens before the run, generate with no dice at all and count how many characters it
gets right. Only confirmed lengths are ever reported. (An earlier version of this skipped the confirmation and
was wrong by a third: a screened position has between 128 and 255 tokens of run-up where a generating model has
the full 256, and that is enough to change its mind.)
"""

import math
import re

import numpy as np
import torch

from gp_thee.evaluation import LABEL, ANOTHER
from gp_thee.train import plan_windows

# The editor's furniture, as opposed to Shakespeare's words. Published, pinned by tests, and deliberately generous:
# anything it wrongly marks as the editor's is taken away from the model's poet-only score, never added to it.
DIRECTION = re.compile(r"\[_.*?_\.?\]", re.S)                       # [_Exit._] and [_One knocks_.]: the full stop can fall inside or outside
HEADING = re.compile(r"^(ACT |SCENE |PROLOGUE|EPILOGUE|INDUCTION|Dramatis Person|THE PROLOGUE|THE EPILOGUE)", re.M)
ENTRANCE = re.compile(r"^(Enter |Exeunt|Exit |Re-enter |Alarum|Flourish|Sennet|Manet|They fight|Music)", re.M)
LETTERS = re.compile(r"[^\W\d_]")


def apparatus(work: str) -> np.ndarray:
    """True for every character of `work` that the editor wrote rather than the poet.

    Speaker labels (the frozen rule of gp_thee.evaluation), bracketed stage directions, act and scene headings,
    entrances and exits, and everything before the first ACT line, which is the title and the cast list.
    """
    mask = np.zeros(len(work), dtype=bool)
    for found in LABEL.finditer(work):
        if not ANOTHER.match(work, found.end()):
            mask[found.start():found.end()] = True
    for pattern in (DIRECTION, HEADING, ENTRANCE):
        for found in pattern.finditer(work):
            end = found.end() if pattern is DIRECTION else work.find("\n", found.start()) + 1 or len(work)
            mask[found.start():end] = True
    first_act = HEADING.search(work)
    mask[:first_act.start() if first_act else 0] = True             # the title page and the cast list
    return mask


def enough_letters(text: str, least: int = 25) -> bool:
    """A passage counts only if it holds this many letters. Without it, runs of indentation score as copies."""
    return len(LETTERS.findall(text)) >= least


class Corpus:
    """The training works, with every passage of every length lookup-able."""

    def __init__(self, works: list[str]):
        self.works = works
        self.text = "\n".join(works)                                 # one line break between works, which belongs to neither
        pieces = [apparatus(work) for work in works]
        self.editor = np.concatenate([p for piece in pieces[:-1] for p in (piece, np.ones(1, dtype=bool))] + [pieces[-1]])
        assert len(self.editor) == len(self.text)
        self.tables: dict[int, np.ndarray] = {}

    def table(self, width: int) -> np.ndarray:
        if width not in self.tables:
            self.tables[width] = np.unique(np.array([hash(self.text[i:i + width]) for i in range(len(self.text) - width + 1)], dtype=np.int64))
        return self.tables[width]

    def holds(self, passage: str) -> bool:
        """Does this exact passage occur in the training works? (Hash first, then read the text: no collisions.)"""
        return bool(len(passage) and np.isin(hash(passage), self.table(len(passage))) and passage in self.text)

    def matching(self, text: str, width: int) -> np.ndarray:
        """For every window of `width` characters in `text`: is it in the training works?"""
        if len(text) < width:
            return np.zeros(0, dtype=bool)
        found = np.isin(np.array([hash(text[i:i + width]) for i in range(len(text) - width + 1)], dtype=np.int64), self.table(width))
        for i in np.flatnonzero(found):                             # confirm each hit against the text itself
            found[i] = text[i:i + width] in self.text
        return found

    def longest_in(self, text: str) -> tuple[int, str]:
        """The longest passage of `text` that occurs in the training works, by bisection on the length."""
        low, high, best = 1, min(len(text), 2000), ""
        while low <= high:
            middle = (low + high) // 2
            found = self.matching(text, middle)
            where = np.flatnonzero(found)
            if len(where):
                best, low = text[where[0]:where[0] + middle], middle + 1
            else:
                high = middle - 1
        return len(best), best


def curve(corpus: Corpus, text: str, widths=(20, 25, 30, 40, 50, 60, 80, 100), poet_only: bool = False) -> dict:
    """How much of `text` is copied, at every length, three ways: windows, characters, and the longest run."""
    out = {}
    for width in widths:
        found = corpus.matching(text, width)
        if poet_only and len(found):                                  # a window must carry real words, not indentation
            letters = np.concatenate([[0], np.cumsum([bool(LETTERS.match(c)) for c in text])])
            found = found & (letters[width:width + len(found)] - letters[:len(found)] >= 25)
        inside = np.zeros(len(text), dtype=bool)
        for i in np.flatnonzero(found):
            inside[i:i + width] = True
        out[width] = {"windows": int(found.sum()), "of": int(max(0, len(text) - width + 1)),
                      "share_of_windows": float(found.mean()) if len(found) else 0.0,
                      "share_of_characters": float(inside.mean()) if len(text) else 0.0}
    length, passage = corpus.longest_in(text)
    out["longest"] = {"characters": length, "passage": passage}
    return out


# ---------------------------------------------------------------------------------------------- the scan
@torch.no_grad()
def agreement(model, stream: np.ndarray, device: str, batch: int = 64) -> np.ndarray:
    """At every token of `stream` after the first: is the true next token the one the model would have written?

    The same windows train.evaluate uses, so every token is judged exactly once with at least 128 tokens behind it.
    """
    was_training = model.training
    model.eval()
    try:
        tokens = torch.from_numpy(np.asarray(stream).astype(np.int64))
        targets = len(tokens) - 1
        length = min(model.config.context, targets)
        plan = plan_windows(targets, model.config.context, model.config.context // 2)
        agreed = np.zeros(targets, dtype=bool)
        for i in range(0, len(plan), batch):
            chunk = plan[i:i + batch]
            windows = torch.stack([tokens[b:b + length] for b, _ in chunk]).to(device)
            answers = torch.stack([tokens[b + 1:b + length + 1] for b, _ in chunk]).to(device)
            guessed = model(windows)[0].argmax(dim=-1) == answers
            for row, (begin, first) in enumerate(chunk):
                agreed[begin + first:begin + length] = guessed[row, first:].cpu().numpy()
    finally:
        model.train(was_training)
    return agreed


def candidates(agreed: np.ndarray, least: int = 40) -> list[tuple[int, int]]:
    """Runs of consecutive agreement, as (first token predicted, length), longest first."""
    edges = np.diff(np.concatenate([[0], agreed.view(np.int8), [0]]))
    runs = [(int(b), int(e - b)) for b, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
    return sorted([run for run in runs if run[1] >= least], key=lambda run: -run[1])


@torch.no_grad()
def confirm(model, tokenizer, stream: np.ndarray, begin: int, length: int, device: str) -> dict:
    """Make the model write it. From the real context before `begin`, generate with no dice and count what it gets right.

    `begin` is the index of the first token the model is asked to predict; the context is everything before it.
    """
    context = model.config.context
    was_training = model.training
    model.eval()
    try:
        ids = list(np.asarray(stream[max(0, begin + 1 - context):begin + 1]).astype(int))
        truth = list(np.asarray(stream[begin + 1:begin + 1 + length]).astype(int))
        written = []
        for true in truth:
            window = torch.tensor([ids[-context:]], device=device)
            chosen = int(model(window)[0][0, -1].argmax())
            if chosen != true:
                break
            written.append(chosen)
            ids.append(chosen)
    finally:
        model.train(was_training)
    text = tokenizer.decode(written)
    return {"confirmed_tokens": len(written), "characters": len(text), "text": text,
            "letters": len(LETTERS.findall(text)), "screened_tokens": length}
