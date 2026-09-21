"""Tests for asking the model for text: the prompt normaliser, the sampler, and the conversation wrapper.

The normaliser is the only part of this project a stranger types into, so its rules are pinned character by
character. The sampler's promises are: the same seed gives the same text, temperature 0 uses no dice at all,
and the model is left exactly as it was found.
"""

import dataclasses
import math

import pytest
import torch

from gp_thee.data import DATA, corpus_alphabet
from gp_thee.model import GPT, Config
from gp_thee.sampling import SCENE, SUITE, Refused, as_a_character, generate, normalise
from gp_thee.tokenizer import load as load_tokenizer

TINY = Config(vocab_size=98, context=64, layers=2, heads=2, width=32, dropout=0.0)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return GPT(TINY)


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(DATA / "tokenizers" / "char.json")


# ------------------------------------------------------------------------------------------ the normaliser
@pytest.mark.parametrize("typed, fed", [
    ("I'll", "I’ll"),                                   # the straight apostrophe, which is in almost every prompt
    ("I`ll", "I’ll"), ("Iʼll", "I’ll"), ("I′ll", "I’ll"),
    ('"yes"', "“yes”"), ('say "yes"', "say “yes”"), ('("yes")', "(“yes”)"),   # opening after a space or a bracket, closing elsewhere
    ("a\tb", "a b"), ("a\r\nb", "a\nb"), ("a\rb", "a\nb"), ("a b", "a\nb"),
    ("a b", "a b"), ("a–b", "a—b"), ("a−b", "a-b"), ("a​b", "ab"),
    ("é", "é"),                                   # NFC: e + combining acute is the é the corpus has
    ("HAMLET.\n", "HAMLET.\n"), ("a-b", "a-b"), ("a—b", "a—b"), ("...", "..."), ("…", "…"), ("&", "&"), ("[_Exit._]", "[_Exit._]"),
])
def test_a_prompt_is_spelled_the_way_this_corpus_spells_it(typed, fed):
    assert normalise(typed)[0] == fed


def test_every_character_of_the_corpus_passes_through_untouched():
    alphabet = "".join(corpus_alphabet())
    assert normalise(alphabet)[0] == alphabet.rstrip(" ") and normalise(alphabet)[1] == []


@pytest.mark.parametrize("typed, named", [("naïve", "ï"), ("cat 🐈", "🐈"), ("café ñ", "ñ"), ("<|start|>", "<"), ("5 × 3", "×")])
def test_a_character_this_universe_does_not_contain_is_refused_by_name(typed, named):
    with pytest.raises(Refused) as refusal:
        normalise(typed)
    assert named in str(refusal.value) and "97 characters" in str(refusal.value)


def test_the_nearest_letter_is_offered_when_there_is_one():
    assert "'n'" in str(pytest.raises(Refused, lambda: normalise("mañana")).value)       # ñ is n with a tilde
    assert "nearest" not in str(pytest.raises(Refused, lambda: normalise("🐈")).value)   # a cat is not a letter at all


def test_trailing_spaces_go_because_a_space_belongs_to_the_character_after_it():
    fed, notes = normalise("To be, or not to be, that is the   ")
    assert fed.endswith("the") and "3 trailing space" in notes[0]
    assert normalise("a b")[0] == "a b" and normalise("\n\n")[0] == "\n\n"               # spaces inside, and line breaks, stay


@pytest.mark.parametrize("typed", ["", "   ", "​", "\t\t"])
def test_a_prompt_with_nothing_left_in_it_is_refused_with_advice(typed):
    assert "HAMLET." in str(pytest.raises(Refused, lambda: normalise(typed)).value)


def test_every_change_is_reported():
    fed, notes = normalise('He said "I\'ll go" – slowly. ')
    assert fed == "He said “I’ll go” — slowly."
    assert len(notes) == 5 and all(isinstance(n, str) for n in notes)                     # two quotes, an apostrophe, a dash, a trailing space


# ------------------------------------------------------------------------------------------ the sampler
def test_the_same_seed_gives_the_same_text_and_another_seed_does_not(model, tokenizer):
    once = generate(model, tokenizer, "HAMLET.\n", characters=60, seed=3, on_start="mask")
    assert once["continuation"] == generate(model, tokenizer, "HAMLET.\n", characters=60, seed=3, on_start="mask")["continuation"]
    assert once["continuation"] != generate(model, tokenizer, "HAMLET.\n", characters=60, seed=4, on_start="mask")["continuation"]
    assert len(once["continuation"]) == 60 and once["prompt"] == "HAMLET.\n" and once["why"].startswith("wrote")


def test_temperature_zero_rolls_no_dice_at_all(model, tokenizer):
    greedy = [generate(model, tokenizer, "HAMLET.\n", characters=40, temperature=0, seed=seed, on_start="mask")["continuation"] for seed in (0, 1, 99)]
    assert len(set(greedy)) == 1                                                          # the seed cannot matter
    assert greedy[0] == generate(model, tokenizer, "HAMLET.\n", characters=40, temperature=1.0, top_k=1, seed=7, on_start="mask")["continuation"]


def test_top_k_and_top_p_narrow_the_choice(model, tokenizer):
    with torch.no_grad():                                                                  # the three the model itself thinks likeliest
        scores = model(torch.tensor([tokenizer.encode("HAMLET.\n")]))[0][0, -1]
    scores[tokenizer.start_id] = -math.inf
    likeliest = {tokenizer.decode([int(i)]) for i in torch.topk(scores, 3).indices}
    ordered = torch.softmax(scores / 2.0, dim=-1).sort(descending=True)                    # the smallest set of characters worth 30% between them
    inside = {tokenizer.decode([int(i)]) for i in ordered.indices[:int((ordered.values.cumsum(0) < 0.3).sum()) + 1]}
    draw = lambda **more: {generate(model, tokenizer, "HAMLET.\n", characters=1, temperature=2.0, seed=s, on_start="mask", **more)["continuation"] for s in range(60)}
    wide, narrow, nucleus = draw(), draw(top_k=3), draw(top_p=0.3)
    assert narrow <= likeliest and len(wide) > 3                                           # top-k draws only from the likeliest few
    assert nucleus <= inside and len(nucleus) < len(wide)                                  # top-p only from the ones that carry the mass


@pytest.mark.parametrize("settings", [{"temperature": -1}, {"top_p": 0}, {"top_p": 1.5}, {"top_k": 0}])
def test_settings_that_make_no_sense_are_refused(model, tokenizer, settings):
    with pytest.raises(ValueError):
        generate(model, tokenizer, "HAMLET.\n", characters=5, **settings)


def test_the_model_can_end_the_work_or_be_forbidden_to(model, tokenizer):
    # A model that wants nothing but START: with "stop" it writes nothing, with "mask" it must write something else.
    class Ends(torch.nn.Module):
        config = TINY
        def __init__(self):
            super().__init__()
            self.nothing = torch.nn.Parameter(torch.zeros(1))
        def forward(self, window):
            scores = torch.zeros(1, window.shape[1], TINY.vocab_size)
            scores[..., tokenizer.start_id] = 50.0
            return scores, None
    ends = Ends()
    assert generate(ends, tokenizer, "HAMLET.\n", characters=20)["continuation"] == ""
    assert "ended the work" in generate(ends, tokenizer, "HAMLET.\n", characters=20)["why"]
    assert len(generate(ends, tokenizer, "HAMLET.\n", characters=20, on_start="mask")["continuation"]) == 20


def test_generation_stops_at_a_stop_string_and_never_includes_it(model, tokenizer):
    block = generate(model, tokenizer, "HAMLET.\n", characters=400, seed=1, stop="e", on_start="mask")
    assert "e" not in block["continuation"] and block["why"] == "reached 'e'"


def test_a_prompt_longer_than_the_window_keeps_its_end_and_says_how_much_fell_out(model, tokenizer):
    long = "ALL’S WELL.\n" + "a b c d e f g h " * 20                                       # more than 64 tokens
    block = generate(model, tokenizer, long, characters=10, seed=0, on_start="mask")
    assert block["dropped_from_the_window"] == len(tokenizer.encode(block["prompt"])) - TINY.context > 0


def test_the_model_is_left_exactly_as_it_was_found(model, tokenizer):
    model.train()
    generate(model, tokenizer, "HAMLET.\n", characters=5, on_start="mask")
    assert model.training
    with pytest.raises(Refused):
        generate(model.eval(), tokenizer, "ñ", characters=5)
    assert not model.training


# ------------------------------------------------------------------------------------------ speak as a character
def test_a_conversation_is_laid_out_as_a_page_of_a_play(model, tokenizer):
    scene = as_a_character(model, tokenizer, [("romeo", "Good morrow."), ("JULIET.", "And to you.")], answerer="friar", characters=50, seed=2)
    assert scene["transcript"].startswith("\n\nROMEO.\nGood morrow.\n\nJULIET.\nAnd to you.\n\nFRIAR.\n")   # names are shouted, the full stop is not doubled
    assert scene["answerer"] == "friar" and not scene["cast_by_the_model"]
    assert "\n\n" not in scene["reply"]                                                    # a reply ends where the next speaker begins


def test_the_model_can_cast_the_answerer_itself(model, tokenizer):
    scene = as_a_character(model, tokenizer, SCENE, characters=40, seed=5)
    assert scene["cast_by_the_model"] and scene["answerer"] and "\n" not in scene["answerer"]
    assert f"\n\n{scene['answerer'].upper()}.\n" in scene["transcript"]


def test_a_reply_that_runs_to_the_cap_is_cut_back_to_a_whole_line(model, tokenizer):
    scene = as_a_character(model, tokenizer, SCENE, answerer="X", characters=120, seed=1)
    assert len(scene["reply"]) <= 120 and (not scene["was_cut"] or not scene["reply"].endswith("\n"))


def test_a_line_in_a_conversation_is_normalised_too(model, tokenizer):
    scene = as_a_character(model, tokenizer, [("ROMEO", "I'll go")], answerer="JULIET", characters=20, seed=0)
    assert "I’ll go" in scene["transcript"]
    with pytest.raises(Refused):
        as_a_character(model, tokenizer, [("ROMEO", "naïve")], answerer="JULIET", characters=20)


# ------------------------------------------------------------------------------------------ the suite
def test_the_suite_is_ten_fixed_prompts_that_this_universe_can_spell():
    assert len(SUITE) == 10 and len({p for _, p in SUITE}) == 10
    for kind, typed in SUITE:
        assert normalise(typed)[0]                                                          # none of them is refused
    assert [kind for kind, _ in SUITE][:2] == ["speaker label", "a name the training works never contain"]
    assert "cat" in dict((k, p) for k, p in SUITE)["an instruction, which this model cannot follow"]
    assert len(SCENE) == 2 and SCENE[0][0] == "ROMEO"
