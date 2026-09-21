"""Tests for the evaluation arithmetic and the baselines. CPU only, a few seconds. The test works are never opened."""

import bz2
import collections
import lzma
import math

import numpy as np
import pytest

from gp_thee.data import DATA, corpus_alphabet, load_tokens, load_works
from gp_thee.evaluation import NGram, breakdown, compressed_bits_per_character, piece_lengths, speaker_label_characters
from gp_thee.tokenizer import BPETokenizer, load as load_tokenizer
from gp_thee.train import bits_per_character, characters_in

TOKENIZERS = ["char", "bpe-1024", "bpe-1536", "bpe-2048", "bpe-4096"]


# ---------------------------------------------------------------------------------------------- n-grams
def slow_witten_bell(train: list[str], text: str, order: int, alphabet: list[str]) -> list[float]:
    """The same model written the obvious way, with dictionaries, one character at a time."""
    follows = [collections.defaultdict(collections.Counter) for _ in range(order)]
    for work in train:
        padded = "\x00" * (order - 1) + work
        for i in range(order - 1, len(padded)):
            for context in range(order):
                follows[context][padded[i - context:i]][padded[i]] += 1
    bits, padded = [], "\x00" * (order - 1) + text
    for i in range(order - 1, len(padded)):
        probability = 1 / len(alphabet)
        for context in range(order):
            seen = follows[context].get(padded[i - context:i])
            if seen:
                probability = (seen[padded[i]] + len(seen) * probability) / (sum(seen.values()) + len(seen))
        bits.append(-math.log2(probability))
    return bits


def test_character_counts_by_hand():
    # "aab": a twice, b once, 2 kinds. P(a) = (2 + 2 x 1/3) / (3 + 2), with an alphabet of three.
    bits = NGram(1, ["a", "b", "c"]).fit(["aab"]).bits(["ac"])
    assert bits == pytest.approx([-math.log2((2 + 2 / 3) / 5), -math.log2((0 + 2 / 3) / 5)])


@pytest.mark.parametrize("order", [1, 2, 3, 5, 8])
def test_ngram_matches_the_obvious_way_on_real_text(order):
    alphabet = corpus_alphabet()
    train = [work[:30_000] for work in load_works("train")[:3]]
    text = load_works("validation")[0][5_000:7_000]
    fast = NGram(order, alphabet).fit(train).bits([text])
    assert fast == pytest.approx(slow_witten_bell(train, text, order, alphabet), abs=1e-9)


@pytest.mark.parametrize("context", ["", "t", "th", "zq", "\n\nHAM", "the ", "Qx’z"])
def test_ngram_probabilities_add_up_to_one(context):
    # Whatever came before, seen or never seen, the chances of all possible next characters must total exactly 1.
    # A smoothing rule that leaks or invents probability makes every bits-per-character figure meaningless.
    alphabet = corpus_alphabet()
    model = NGram(5, alphabet).fit([work[:50_000] for work in load_works("train")[:2]])
    total = sum(2 ** -model.bits([context + ch])[-1] for ch in alphabet)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_ngram_starts_each_work_afresh():
    # The first character of a work is predicted from "nothing came before", not from the end of the previous work.
    model = NGram(3, ["a", "b"]).fit(["ab", "ab"])
    assert model.bits(["ab", "ab"])[:2] == pytest.approx(model.bits(["ab"]))
    assert model.bits(["ab", "ab"])[2:] == pytest.approx(model.bits(["ab"]))


def test_a_long_memory_learns_its_own_training_text_by_heart():
    work = load_works("train")[0][:20_000]
    alphabet = corpus_alphabet()
    assert NGram(8, alphabet).fit([work]).bits([work]).mean() < 0.5 < NGram(1, alphabet).fit([work]).bits([work]).mean()


def test_ngram_refuses_an_order_that_would_overflow():
    NGram(9, corpus_alphabet())                                   # 98 ** 9 still fits in 64 bits
    for order, alphabet in ((10, corpus_alphabet()), (9, [chr(i) for i in range(200)]), (0, ["a"])):
        with pytest.raises(ValueError):
            NGram(order, alphabet)


def test_fitting_twice_is_fitting_afresh():
    model = NGram(3, ["a", "b"]).fit(["abab"])
    once = model.bits(["aabb"])
    assert np.array_equal(model.fit(["abab"]).bits(["aabb"]), once)


# ---------------------------------------------------------------------------------------------- compressors
def test_compressor_figures_are_bits_per_character_at_the_settings_their_names_say():
    text = load_works("validation")[0]
    data = text.encode("utf-8")
    assert len(data) > len(text) > 100_000                        # bytes are not characters here, and the text is more than one small block
    found = compressed_bits_per_character(text)
    assert found["bzip2 -9"] == 8 * len(bz2.compress(data, 9)) / len(text)
    assert found["xz -9e"] == 8 * len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)) / len(text)
    assert found["xz -9e"] != 8 * len(lzma.compress(data, preset=9)) / len(text)


def test_compressors():
    text = load_works("validation")[0][:50_000]
    alone = compressed_bits_per_character(text)
    assert set(alone) == {"bzip2 -9", "xz -9e"} and all(1.5 < bits < 4.5 for bits in alone.values())
    # a compressor that has already read this very text needs almost nothing to write it again
    assert compressed_bits_per_character(text, after_reading=text)["xz -9e"] < 0.05


# ---------------------------------------------------------------------------------------------- speaker labels
PLAY = "A PLAY\n\nDramatis Personæ\n\nHAMLET.\nOPHELIA.\nA Ghost.\n\nACT I\n\nSCENE I. A room.\n\nHAMLET.\nTo be, or not to be.\nO.\n\n2 KEEPER.\nAy.\n\n[_Exit._]\n"


def test_speaker_labels_are_found_and_cast_lists_are_not():
    mask = speaker_label_characters(PLAY)
    assert "".join(ch for ch, marked in zip(PLAY, mask) if marked) == "HAMLET.\n2 KEEPER.\n"
    # the cast list's HAMLET is not marked; the one before the speech is
    assert not mask[PLAY.index("HAMLET.")] and mask[PLAY.rindex("HAMLET.")]


def test_the_label_rule_on_its_edge_cases():
    marked = lambda text: "".join(ch for ch, on in zip(text, speaker_label_characters(text)) if on)
    assert marked("T\n\nACT II.\n\nSCENE I. Paris.\n") == ""                      # a heading: a blank line follows it
    assert marked("T\n\nGHOST.\n[_Aside._] Boo.\n") == "GHOST.\n"                 # a stage direction may open the speech
    assert marked("T\n\n_Retreat._\nANTONY.\nThey do retire.\n") == ""            # KNOWN MISS: no blank line before the label
    assert marked("T\n\nPETER.\nI.\n\nPETRUCHIO.\nWho?\n") == "PETRUCHIO.\n"      # KNOWN MISS: the speech "I." looks like a label
    assert marked("T\n\nHORATIO and MARCELLUS.\nMy lord.\n") == ""                # KNOWN MISS: a small "and"
    assert marked("T\n\n       THE SONG.\n  Come, thou monarch\n") == "       THE SONG.\n"   # KNOWN FALSE ALARM
    assert marked("T\n\nKING HENRY V.\nDUKE OF CLARENCE, brother to the King.\n") == "KING HENRY V.\n"   # KNOWN FALSE ALARM


def test_speaker_labels_in_the_validation_works():
    works = load_works("validation")
    marked = sum(int(speaker_label_characters(work).sum()) for work in works)
    assert 0.04 < marked / sum(map(len, works)) < 0.08
    labels = {line for work in works for line in "".join(ch if m else "\x00" for ch, m in zip(work, speaker_label_characters(work))).replace("\x00", "").split("\n")}
    assert {"HELENA.", "PAROLLES.", "ROMEO.", "JULIET.", "NURSE."} <= labels
    assert all(line == line.upper() and (line.endswith(".") or not line) for line in labels)


# ---------------------------------------------------------------------------------------------- the breakdown
@pytest.mark.parametrize("name", TOKENIZERS)
def test_breakdown_adds_up_and_agrees_with_the_headline_number(name):
    tokenizer, stream, works = load_tokenizer(DATA / "tokenizers" / f"{name}.json"), load_tokens(name, "validation"), load_works("validation")
    surprise = np.random.default_rng(0).uniform(0.1, 5.0, size=len(stream) - 1)
    parts = breakdown(surprise, stream, tokenizer, works)      # also checks that no piece straddles a label line's edge
    assert parts["all"]["characters"] == characters_in(stream, tokenizer) == sum(map(len, works)) == 274_727
    assert parts["all"]["bits_per_character"] == pytest.approx(bits_per_character(surprise, characters_in(stream, tokenizer)), rel=1e-12)
    assert [p["title"] for p in parts["works"]] == ["ALL’S WELL THAT ENDS WELL", "THE TRAGEDY OF ROMEO AND JULIET"]
    assert [p["characters"] for p in parts["works"]] == [len(work) for work in works]
    assert parts["speaker_labels"]["characters"] == sum(int(speaker_label_characters(work).sum()) for work in works)
    assert parts["all"]["tokens"] == len(stream) - 1 == sum(p["tokens"] for p in parts["works"])


def test_breakdown_puts_each_token_in_the_right_place():
    tokenizer = load_tokenizer(DATA / "tokenizers" / "char.json")
    works = ["T\n\nABE.\nHi\n", "U\n\nBO.\nYo\n"]
    stream = np.array(tokenizer.encode_works(works))
    surprise = np.arange(1, len(stream), dtype=float)             # token i costs i nats, so every sum is known
    parts = breakdown(surprise, stream, tokenizer, works)
    second_start = len(works[0]) + 1                              # START, the first work, then the second START
    first, second = parts["works"]
    assert first["characters"] == len(works[0]) and first["tokens"] == len(works[0])
    assert first["bits"] == pytest.approx(sum(range(1, second_start)) / math.log(2))
    assert second["tokens"] == len(works[1]) + 1                  # its START is scored, and stands for no characters
    assert second["bits"] == pytest.approx(sum(range(second_start, len(stream))) / math.log(2))
    assert parts["speaker_labels"]["characters"] == len("ABE.\n") + len("BO.\n")
    labels_at = [1 + works[0].index("ABE."), second_start + 1 + works[1].index("BO.")]
    expected = sum(range(labels_at[0], labels_at[0] + 5)) + sum(range(labels_at[1], labels_at[1] + 4))
    assert parts["speaker_labels"]["bits"] == pytest.approx(expected / math.log(2))


def test_breakdown_refuses_a_stream_that_does_not_spell_the_works():
    tokenizer = load_tokenizer(DATA / "tokenizers" / "char.json")
    stream = np.array(tokenizer.encode_works(["abc\n"]))
    with pytest.raises(ValueError):
        breakdown(np.ones(len(stream) - 1), stream, tokenizer, ["abd\n"])


def test_piece_lengths_count_characters_not_bytes():
    tokenizer = load_tokenizer(DATA / "tokenizers" / "bpe-2048.json")
    lengths = piece_lengths(tokenizer)
    assert lengths[tokenizer.start_id] == 0 and lengths[tokenizer.encode("’")[0]] == 1 and lengths.max() > 5


def test_breakdown_refuses_a_piece_that_straddles_the_edge_of_a_label_line():
    work = "T\n\nABE.\nHi\n"
    tokenizer = BPETokenizer(sorted(set(work)), [("\n", "H")])     # a piece none of our tokenizers has: a line break glued to a letter
    pieces = ["T", "\n", "\n", "A", "B", "E", ".", "\nH", "i", "\n"]
    stream = np.array([tokenizer.start_id] + [tokenizer.id_of[piece] for piece in pieces])
    with pytest.raises(AssertionError, match="straddles"):
        breakdown(np.ones(len(stream) - 1), stream, tokenizer, [work])


def test_breakdown_refuses_a_stream_whose_starts_are_not_where_the_works_begin():
    tokenizer = load_tokenizer(DATA / "tokenizers" / "char.json")
    works = ["T\n\nABE.\nHi\n", "U\n\nBO.\nYo\n"]
    right = tokenizer.encode_works(works)
    late = [t for t in right if t != tokenizer.start_id]
    late = [tokenizer.start_id] + late[:14] + [tokenizer.start_id] + late[14:]       # the second START three characters late
    missing = [tokenizer.start_id] + [t for t in right if t != tokenizer.start_id]   # no second START at all
    no_first = right[1:]
    for stream in (late, missing, no_first):
        with pytest.raises(AssertionError):
            breakdown(np.ones(len(stream) - 1), np.array(stream), tokenizer, works)


def test_a_part_with_no_characters_has_no_score_rather_than_a_zero():
    tokenizer = load_tokenizer(DATA / "tokenizers" / "char.json")
    works = ["A POEM\n\nNo one speaks here.\n"]
    stream = np.array(tokenizer.encode_works(works))
    parts = breakdown(np.ones(len(stream) - 1), stream, tokenizer, works)
    assert parts["speaker_labels"] == {"characters": 0, "tokens": 0, "bits": 0.0, "bits_per_character": None}
