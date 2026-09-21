"""Asking the model for text: making a prompt spellable, drawing the next character, and holding a conversation.

The model's universe has 97 characters in it and nothing else. A prompt typed on a modern keyboard usually does
not: the apostrophe in "I'll" is a straight one, the dash is a hyphen, the quotes are straight, and there may be
a tab in there. So a prompt passes through a normaliser first, outside the tokenizer, which does one of three
things to every character:

  * leaves it alone, if the corpus already uses it;
  * puts in the thing this corpus uses in its place (the straight apostrophe becomes ’, the tab becomes a space);
  * refuses the whole prompt and names the character, with its codepoint and the nearest plain letter we can offer.

Refusing is the point. A model that has never seen a `ñ` cannot be asked about one, and quietly dropping it would
leave the reader wondering what the model was really given. Every change is reported beside the sample.

Two rules that are not about spelling. Trailing spaces are stripped, because in this project's tokenizers a space
belongs to the token AFTER it (part 3): a prompt ending in a space asks the model to continue a word that has not
started. And START is never prepended. START means "a new work begins here"; an ordinary prompt is the middle of
one. The only way to feed it is to ask for it.

The sampler itself is the one from training (train.generate) with the things a real sampler needs: a temperature
of zero (take the likeliest character, no dice at all), top-k and top-p, and a reason for stopping. It re-reads
the whole window for every character, which costs about a hundred times what a cache would; for a few hundred
characters on this machine that is under two seconds, so the simple code stays.
"""

import dataclasses
import json
import math
import unicodedata

import torch

from gp_thee.data import corpus_alphabet

# What a modern keyboard offers, and what this corpus uses in its place.
INSTEAD = {
    "'": "’", "ʼ": "’", "′": "’", "´": "’", "`": "’",          # apostrophes and things that look like one
    "\t": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "\r": "\n", "": "\n", "": "\n", "": "\n", " ": "\n", " ": "\n",
    "–": "—", "‒": "—", "―": "—", "−": "-",               # dashes: the corpus has an em dash and a hyphen
    "​": "", "‌": "", "‍": "", "﻿": "", "­": "",     # zero-width things, simply dropped
}
OPENS_AFTER = set(" \n([{-—")   # a double quote here is an opening one


@dataclasses.dataclass(frozen=True)
class Refused(Exception):
    """The prompt cannot be spelled in this universe. `why` says what to do about it."""
    why: str

    def __str__(self) -> str:
        return self.why


def normalise(prompt: str) -> tuple[str, list[str]]:
    """The prompt as the model can be given it, and a list of everything that was changed.

    Raises Refused if a character has no place in the corpus and no sensible stand-in.
    """
    alphabet = set(corpus_alphabet())
    text, notes = [], []
    for character in unicodedata.normalize("NFC", prompt.replace("\r\n", "\n")):
        if character in alphabet:
            text.append(character)
            continue
        if character in INSTEAD:
            instead = INSTEAD[character]
            notes.append(f"{name_of(character)} became {name_of(instead) if instead else 'nothing'}")
            text.append(instead)
            continue
        if character == '"':
            instead = "“" if not text or text[-1] in OPENS_AFTER else "”"
            notes.append(f"a straight double quote became {instead}")
            text.append(instead)
            continue
        plain = "".join(c for c in unicodedata.normalize("NFD", character) if c in alphabet)
        nearest = f" The nearest letter this universe has is {plain!r}." if plain else ""
        raise Refused(f"{name_of(character)} is not one of the 97 characters in the complete works, so the model has never "
                      f"seen it and cannot be asked about it.{nearest}")
    text = "".join(text)
    if text.rstrip(" ") != text:
        notes.append(f"{len(text) - len(text.rstrip(' '))} trailing space(s) removed: a space belongs to the character after it")
        text = text.rstrip(" ")
    if not text:
        raise Refused("the prompt is empty. Try a speaker's name on a line of its own, such as 'HAMLET.' and a line break.")
    return text, notes


def name_of(character: str) -> str:
    if character == "":
        return "nothing"
    if character == " ":
        return "a space"
    if character == "\n":
        return "a line break"
    if character == "\t":
        return "a tab"
    try:
        return f"{character!r} ({unicodedata.name(character).lower()}, U+{ord(character):04X})"
    except ValueError:
        return f"U+{ord(character):04X}"


@torch.no_grad()
def generate(model, tokenizer, prompt: str, characters: int = 400, temperature: float = 0.8, seed: int = 0,
             top_k: int | None = None, top_p: float | None = None, on_start: str = "stop", stop: str | None = None) -> dict:
    """Continue `prompt` until it has written `characters` characters, or the work ends, or `stop` appears.

    temperature   below 1 the model plays safe, above 1 it gambles. Zero means no dice at all: always the likeliest.
    top_k         consider only the k likeliest characters. top_p: only the likeliest whose chances add up to p.
    on_start      what to do when the model writes START, which means "this work has ended": "stop", or "mask" to
                  forbid it, as the sampler inside a training run does.

    Returns the continuation (not the prompt), why it stopped, and what the normaliser changed.
    """
    if temperature < 0 or (top_p is not None and not 0 < top_p <= 1) or (top_k is not None and top_k < 1):
        raise ValueError("temperature cannot be negative, top_p must be in (0, 1], top_k at least 1")
    text, notes = normalise(prompt)
    ids = tokenizer.encode(text)
    given = len(ids)
    context = model.config.context
    dropped = max(0, len(ids) - context)
    was_training = model.training
    model.eval()
    dice = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    written, why = "", f"wrote {characters} characters"
    try:
        while len(written) < characters:
            window = torch.tensor([ids[-context:]], device=device)
            scores = model(window)[0][0, -1].float().cpu()
            if on_start == "mask":
                scores[tokenizer.start_id] = -math.inf
            if temperature == 0:
                chosen = int(scores.argmax())
            else:
                scores = scores / temperature
                if top_k is not None:
                    scores[scores < torch.topk(scores, min(top_k, len(scores))).values[-1]] = -math.inf
                if top_p is not None:
                    ordered, where = torch.sort(scores, descending=True)
                    beyond = torch.softmax(ordered, dim=-1).cumsum(dim=-1) - torch.softmax(ordered, dim=-1) >= top_p
                    scores[where[beyond]] = -math.inf
                chosen = int(torch.multinomial(torch.softmax(scores, dim=-1), 1, generator=dice))
            if chosen == tokenizer.start_id:
                why = "the model ended the work (it wrote START)"
                break
            ids.append(chosen)
            written = tokenizer.decode(ids[given:])
            if stop and stop in written:
                written, why = written[:written.index(stop)], f"reached {stop!r}"
                break
    finally:
        model.train(was_training)
    return {"prompt": text, "continuation": written[:characters] if why.startswith("wrote") else written, "why": why,
            "notes": notes, "dropped_from_the_window": dropped,
            "settings": {"temperature": temperature, "seed": seed, "top_k": top_k, "top_p": top_p, "on_start": on_start}}


# ---------------------------------------------------------------------------------------------- speak as a character
def as_a_character(model, tokenizer, turns: list[tuple[str, str]], answerer: str | None = None, characters: int = 600,
                   temperature: float = 0.8, seed: int = 0) -> dict:
    """One reply, written as a page of a play: the only kind of conversation this model can hold.

    `turns` is [(speaker, line), ...] so far. The transcript is built exactly as the corpus lays a scene out, and
    the model is asked to continue after the answerer's label. With no answerer, the model casts one itself.

    A reply ends at the first blank line, because that is where the next speaker begins. What comes after it is
    thrown away: it is the model writing the rest of the scene, which nobody asked for.
    """
    page = "".join(f"\n\n{speaker.strip().upper().rstrip('.')}.\n{normalise(line)[0]}" for speaker, line in turns)
    if answerer:
        opening = f"{page}\n\n{answerer.strip().upper().rstrip('.')}.\n"
        cast_by = None
    else:                                                      # let the model choose who answers: it writes the label itself
        cast = generate(model, tokenizer, f"{page}\n\n", characters=40, temperature=temperature, seed=seed, stop="\n")
        cast_by = cast["continuation"].strip().rstrip(".") or "A VOICE"
        opening = f"{page}\n\n{cast_by.upper()}.\n"
    said = generate(model, tokenizer, opening, characters=characters, temperature=temperature, seed=seed + 1, stop="\n\n")
    reply = said["continuation"]
    if said["why"].startswith("wrote") and "\n" in reply:      # ran to the cap: cut back to a whole line
        reply, cut = reply[:reply.rindex("\n")], True
    else:
        cut = False
    return {"answerer": answerer or cast_by, "cast_by_the_model": answerer is None, "reply": reply.strip("\n"),
            "was_cut": cut, "why": said["why"], "dropped_from_the_window": said["dropped_from_the_window"],
            "transcript": opening + reply}


# ---------------------------------------------------------------------------------------------- the ten-prompt suite
SUITE = [
    ("speaker label", "\n\nHAMLET.\n"),
    ("a name the training works never contain", "\n\nROMEO.\n"),
    ("stage direction", "\n\nEnter "),
    ("a sonnet that does not exist", "\n\n\n                    155\n\nWhen first I looked upon thy "),
    ("prose, typed with a straight apostrophe", "\n\nFALSTAFF.\nI'll be sworn upon all the books in England, "),
    ("a famous line, cut off", "To be, or not to be, that is the "),
    ("cut off inside a word", "Good morrow, gentle cousin. Whith"),
    ("a modern sentence", "The meeting is at nine o'clock, and the budget "),
    ("an instruction, which this model cannot follow", "Write me a poem about a cat."),
    ("nothing but a blank line", "\n\n"),
]
SCENE = [("ROMEO", "But soft, what light through yonder window breaks?"), ("JULIET", "I have a headache and no money.")]
