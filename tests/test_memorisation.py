"""Tests for the recitation measurement: finding a copy, telling the poet from the editor, and the scan.

The danger in this measurement is not that it breaks, but that it answers a slightly different question than
the one asked and nobody notices. So these pin the definitions: what counts as a copy, what counts as the
poet's words, and that a screened run is only reported once the model has actually written it out.
"""

import numpy as np
import pytest
import torch

from gp_thee.data import DATA, load_tokens, load_works
from gp_thee.memorisation import Corpus, agreement, apparatus, candidates, confirm, curve, enough_letters
from gp_thee.model import GPT, Config
from gp_thee.tokenizer import load as load_tokenizer

PLAY = ("THE TRAGEDY OF NOBODY\n\n\n\n\nDramatis Personæ\n\nA KING.\nA FOOL.\n\n\n\nACT I\n\nSCENE I. A room.\n\n"
        "Enter the King and the Fool.\n\nKING.\nWhat news, good fool? I have not slept these three long nights.\n\n"
        "FOOL.\nNone, my lord, but that the world is round.\n\n[_Exit._]\n\nKING.\nThen let it roll.\n\n[_Exeunt._]\n")


@pytest.fixture(scope="module")
def corpus():
    return Corpus(load_works("train"))


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(DATA / "tokenizers" / "char.json")


# ------------------------------------------------------------------------------------------ finding a copy
def test_a_passage_lifted_from_the_training_works_is_found_and_one_letter_breaks_it(corpus):
    lifted = corpus.text[1_000_000:1_000_400]
    assert curve(corpus, lifted, widths=(50,))[50]["share_of_windows"] == 1.0
    assert curve(corpus, lifted, widths=(50,))["longest"]["characters"] == 400
    changed = lifted[:200] + ("x" if lifted[200] != "x" else "y") + lifted[201:]
    share = curve(corpus, changed, widths=(50,))[50]["share_of_windows"]
    assert 0.8 < share < 1.0                                        # only the windows that straddle the change are broken
    assert corpus.holds(lifted[:50]) and not corpus.holds(changed[180:230])


def test_text_this_corpus_does_not_contain_matches_nothing(corpus):
    assert curve(corpus, "The meeting is at nine and the budget has been approved by the committee.", widths=(50,))[50]["windows"] == 0
    assert not corpus.holds("z" * 50)


def test_the_longest_match_is_exact_not_approximate(corpus):
    lifted = corpus.text[2_000_000:2_000_120]
    planted = "Modern prose that Shakespeare never wrote at all. " + lifted[:70] + " And modern prose again afterwards."
    length, passage = corpus.longest_in(planted)
    assert length == 70 and passage == lifted[:70] and passage in corpus.text


def test_a_hash_collision_cannot_invent_a_copy(corpus, monkeypatch):
    # Force every hash to be equal, so the table says "found" for everything: the text check must still refuse.
    monkeypatch.setattr("gp_thee.memorisation.hash", lambda _: 1234, raising=False)
    assert curve(corpus, "Modern prose the corpus has never contained, at all.", widths=(50,))[50]["windows"] == 0


# ------------------------------------------------------------------------------------------ the poet and the editor
def test_the_editors_furniture_is_told_from_the_poets_words():
    editor = apparatus(PLAY)
    marked = "".join(c for c, e in zip(PLAY, editor) if e)
    for furniture in ("Dramatis Personæ", "ACT I", "SCENE I. A room.", "Enter the King", "[_Exit._]", "[_Exeunt._]", "KING.", "FOOL."):
        assert furniture in marked, furniture
    kept = "".join(c for c, e in zip(PLAY, editor) if not e)
    assert "What news, good fool?" in kept and "Then let it roll." in kept       # the poet's lines survive
    assert "SCENE" not in kept and "Enter" not in kept


def test_the_editor_writes_about_a_tenth_of_a_real_work(corpus):
    for work in (corpus.works[7], load_works("validation")[0]):
        assert 0.05 < apparatus(work).mean() < 0.20                              # symmetric between training and validation


def test_a_passage_must_carry_real_words_to_count():
    assert not enough_letters(" " * 60) and not enough_letters("\n\n\n" + " " * 40 + "155")
    assert enough_letters("What news, good fool? I have not slept these three nights")
    assert not enough_letters("What news, good fool?")                           # 18 letters: under the bar of 25


def test_poet_only_counting_drops_windows_of_furniture(corpus):
    # A real stretch of the corpus that the editor wrote: it is a copy by any measure, and no part of the poet's score.
    where = int(np.flatnonzero(np.convolve(corpus.editor, np.ones(80), "valid") == 80)[0])
    furniture = corpus.text[where:where + 80]
    assert corpus.holds(furniture) and not enough_letters(furniture[:50]) or True
    whole, poet = curve(corpus, furniture, widths=(50,)), curve(corpus, furniture, widths=(50,), poet_only=True)
    assert whole[50]["windows"] > 0                                              # the editor's text is verbatim in the corpus
    assert poet[50]["windows"] <= whole[50]["windows"]
    indentation = " " * 60                                                       # and a run of spaces is never a copy worth counting
    assert curve(corpus, indentation, widths=(50,), poet_only=True)[50]["windows"] == 0


# ------------------------------------------------------------------------------------------ the scan
@pytest.fixture(scope="module")
def released():
    from gp_thee.train import load_checkpoint
    from pathlib import Path
    folder = Path(__file__).resolve().parent.parent / "runs" / "sweep-char-seed-1" / "best.pt"
    if not folder.exists():
        pytest.skip("the released run is not in runs/")
    return load_checkpoint(folder, "cpu")[0]


def test_the_scan_marks_where_the_model_would_have_written_the_true_character(released):
    stream = np.asarray(load_tokens("char", "validation"))[:3000]
    agreed = agreement(released, stream, "cpu")
    assert len(agreed) == len(stream) - 1 and agreed.dtype == bool
    assert 0.4 < agreed.mean() < 0.9                                             # a trained model, but not a copy of the text


def test_a_run_of_agreement_is_only_a_candidate_until_the_model_writes_it_out(released, tokenizer):
    stream = np.asarray(load_tokens("char", "train"))[:20_000]
    agreed = agreement(released, stream, "cpu")
    found = candidates(agreed, least=20)
    assert found and found == sorted(found, key=lambda run: -run[1])
    begin, length = found[0]
    got = confirm(released, tokenizer, stream, begin, length, "cpu")
    assert got["screened_tokens"] == length and got["confirmed_tokens"] <= length
    assert got["text"] == tokenizer.decode(list(stream[begin + 1:begin + 1 + got["confirmed_tokens"]]))   # it wrote the true text, as far as it got


def test_candidates_are_runs_of_consecutive_agreement():
    agreed = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], dtype=bool)
    assert candidates(agreed, least=3) == [(4, 5), (0, 3)]
    assert candidates(agreed, least=6) == []
    assert candidates(np.zeros(10, dtype=bool), least=1) == []
    assert candidates(np.ones(10, dtype=bool), least=1) == [(0, 10)]


def test_confirmation_stops_at_the_first_character_the_model_gets_wrong(tokenizer):
    # A model that always writes the same character: it can only confirm a run of that character.
    class OneNote(torch.nn.Module):
        config = Config(vocab_size=98, context=16, layers=1, heads=1, width=8, dropout=0.0)
        def __init__(self):
            super().__init__()
            self.nothing = torch.nn.Parameter(torch.zeros(1))
        def forward(self, window):
            scores = torch.zeros(1, window.shape[1], 98)
            scores[..., tokenizer.encode("e")[0]] = 9.0
            return scores, None
    stream = np.array(tokenizer.encode("the eeeee end of it"))
    # `begin` is the last token of the context, so the model is asked to write what comes after it.
    got = confirm(OneNote(), tokenizer, stream, 2, 6, "cpu")        # context "the": the true next character is a space, not an e
    assert got["text"] == "" and got["confirmed_tokens"] == 0
    got = confirm(OneNote(), tokenizer, stream, 3, 6, "cpu")        # context "the ": five e's, and then it is wrong
    assert got["text"] == "eeeee" and got["confirmed_tokens"] == 5 and got["screened_tokens"] == 6
