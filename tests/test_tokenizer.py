"""Tests for the tokenizers.  Run:  uv run pytest

The first group needs no data: small examples worked out by hand. The second group checks the tokenizers
saved in data/tokenizers/ against the corpus, including the two promises that matter most: nothing is
lost in a round trip, and nothing was learned from a held-out work.
"""

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from gp_thee.data import corpus_alphabet, load_works
from gp_thee.tokenizer import START, BPETokenizer, CharTokenizer, chunks, learn_merges, load

ROOT = Path(__file__).resolve().parent.parent
TOKENIZERS = ROOT / "data" / "tokenizers"
NAMES = ["char", "bpe-1024", "bpe-1536", "bpe-2048", "bpe-4096"]


# ------------------------------------------------------------------------------------------ by hand
def test_chunks_are_words_with_their_leading_space():
    assert chunks("HAMLET.\nTo be, or not") == ["HAMLET", ".", "\n", "To", " be", ",", " or", " not"]


def test_the_apostrophe_is_a_letter():
    assert chunks("’Tis lov’d o’er th’ sea") == ["’Tis", " lov’d", " o’er", " th’", " sea"]


def test_stage_direction_punctuation_stays_together():
    assert chunks("[_Exit._]\n\nKING.") == ["[_", "Exit", "._]", "\n\n", "KING", "."]


def test_indentation_keeps_its_last_space_for_the_word():
    assert chunks("\n    And yet") == ["\n", "   ", " And", " yet"]


def test_learning_merges_on_an_example_small_enough_to_do_by_hand():
    # One chunk: a a a b d a a a b a c
    #   most frequent pair is (a, a), counted 4 times
    #   (overlaps count: "aaa" holds it twice)             -> aa a b d aa a b a c
    #   then (aa, a) and (a, b) tie at 2; the tie goes to
    #   the pair that sorts last, (aa, a)                  -> aaa b d aaa b a c
    #   then (aaa, b), twice                               -> aaab d aaab a c
    #   every remaining pair occurs once, so learning stops, however many merges were asked for.
    assert learn_merges(["aaabdaaabac"], 10) == [("a", "a"), ("aa", "a"), ("aaa", "b")]


def test_merges_are_replayed_in_the_order_they_were_learned():
    tokenizer = BPETokenizer(sorted("abcd"), [("a", "a"), ("aa", "a"), ("aaa", "b")])
    assert [tokenizer.vocab[i] for i in tokenizer.encode("aaabdaaabac")] == ["aaab", "d", "aaab", "a", "c"]
    assert tokenizer.decode(tokenizer.encode("aaabdaaabac")) == "aaabdaaabac"


def test_merges_never_cross_a_chunk_boundary():
    # "ab ab ab ab": the chunks are "ab", " ab", " ab", " ab". The pair (b, " ") never sits inside one chunk,
    # so it can never be merged, however often it occurs in the running text.
    merges = learn_merges(["ab ab ab ab"], 10)
    assert ("b", " ") not in merges and all(" " not in (left + right)[1:] for left, right in merges)


def test_a_character_outside_the_alphabet_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="not in the vocabulary"):
        CharTokenizer(sorted("abc")).encode("abz")


def test_start_cannot_be_typed():
    # Shakespeare's alphabet has no "<", "|" or ">", so no text can spell START and encode() can never
    # produce its id. An alphabet that COULD spell it is refused when the tokenizer is built.
    tokenizer = CharTokenizer(sorted("arst"))  # START's letters, but none of its brackets
    assert tokenizer.start_id not in tokenizer.encode("start")
    with pytest.raises(ValueError, match="not in the vocabulary"):
        tokenizer.encode(START)
    with pytest.raises(ValueError, match="can be spelled"):
        CharTokenizer(sorted(set(START)))


def test_works_are_joined_with_start_and_decode_drops_it():
    tokenizer = CharTokenizer(sorted("abc"))
    stream = tokenizer.encode_works(["ab", "ca"])
    assert stream == [tokenizer.start_id, 0, 1, tokenizer.start_id, 2, 0]
    assert tokenizer.decode(stream) == "abca"
    assert tokenizer.decode(stream, show_start=True) == START + "ab" + START + "ca"


def naive_learn_merges(texts, how_many):
    """The slow, obvious way: recount every pair in every chunk before every merge. No shortcuts to get wrong."""
    counts = Counter(chunk for text in texts for chunk in chunks(text))
    words = {chunk: list(chunk) for chunk in counts}
    merges = []
    while len(merges) < how_many:
        pairs = Counter()
        for chunk, word in words.items():
            for pair in zip(word, word[1:]):
                pairs[pair] += counts[chunk]
        if not pairs or max(pairs.values()) < 2:
            break
        best = max(pairs, key=lambda pair: (pairs[pair], pair))
        merges.append(best)
        for chunk, word in words.items():
            merged, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                    merged.append(word[i] + word[i + 1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            words[chunk] = merged
    return merges


def test_the_fast_learner_agrees_with_the_obvious_one_on_random_text():
    rng = random.Random(1)
    for _ in range(200):
        text = "".join(rng.choice("aab c’\n_.") for _ in range(rng.randint(1, 60)))
        assert learn_merges([text], 25) == naive_learn_merges([text], 25), repr(text)


def test_ids_outside_the_vocabulary_are_errors_too():
    tokenizer = CharTokenizer(sorted("abc"))
    for bad in ([-1], [4], [0, 99]):  # -1 would otherwise mean "the last entry", which is START
        with pytest.raises(ValueError, match="outside the vocabulary"):
            tokenizer.decode(bad)


def test_bad_vocabularies_are_refused():
    with pytest.raises(ValueError):
        CharTokenizer(["b", "a"])                                    # not sorted
    with pytest.raises(ValueError):
        BPETokenizer(sorted("ab"), [("a", "b"), ("a", "b")])        # two entries spelling the same piece
    with pytest.raises(ValueError, match="not in the vocabulary"):
        BPETokenizer(sorted("ab"), [("a", "b")]).encode("abz")
    with pytest.raises(ValueError, match="chunk rule"):
        chunks("a\tb")


def test_saving_and_loading_loses_nothing(tmp_path):
    awkward = BPETokenizer(sorted('ab\n"\\ ’'), [("\n", "\n"), (" ", "a"), ('"', "\\")])
    awkward.save(tmp_path / "t.json", {"set": "none"})
    again = load(tmp_path / "t.json")
    assert (again.vocab, again.merges, again.start_id) == (awkward.vocab, awkward.merges, awkward.start_id)
    document = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    for field, wrong in (("kind", "banana"), ("vocab_size", 5), ("start_token", "<s>"), ("start_id", 0)):
        (tmp_path / "bad.json").write_text(json.dumps({**document, field: wrong}), encoding="utf-8")
        with pytest.raises(ValueError):
            load(tmp_path / "bad.json")


# ------------------------------------------------------------------------------------------ against the corpus
@pytest.fixture(scope="module")
def works():
    return {name: load_works(name) for name in ("train", "validation")}  # never the test works


def test_the_test_works_are_locked():
    with pytest.raises(PermissionError, match="final evaluation"):
        load_works("test")


def test_any_work_can_be_encoded_without_opening_it(tokenizers):
    # The alphabet of all 44 works was recorded at cleaning time. If every tokenizer knows all of it, and any
    # string over the alphabet survives the round trip (tested below), then the test works do too, unread.
    for tokenizer in tokenizers.values():
        assert set(corpus_alphabet()) <= set(tokenizer.chars)


@pytest.fixture(scope="module")
def tokenizers():
    return {name: load(TOKENIZERS / f"{name}.json") for name in NAMES}


@pytest.mark.parametrize("name", NAMES)
def test_every_work_survives_the_round_trip(name, works, tokenizers):
    tokenizer = tokenizers[name]
    for texts in works.values():
        for text in texts:
            ids = tokenizer.encode(text)
            assert tokenizer.start_id not in ids
            assert tokenizer.decode(ids, show_start=True) == text  # START shown, so a stray one cannot hide


@pytest.mark.parametrize("name", NAMES)
def test_any_string_of_known_characters_survives_the_round_trip(name, tokenizers):
    # Prompts will not look like the corpus. Four kinds of nasty string, 500 of each.
    tokenizer, rng = tokenizers[name], random.Random(0)
    letters = [ch for ch in tokenizer.chars if ch.isalpha()]
    kinds = [
        tokenizer.chars,                                          # anything at all
        [" ", " ", " ", "\n", "\n", "a", "’", "_", "1"],          # mostly whitespace: long runs of spaces and newlines
        letters + ["’", "’", " ", " ", " "],                      # word-like, heavy on apostrophes and accents
        list("[_]._,;:—!?“”‘’…&-()") + [" ", "\n", "7"],          # punctuation and digits
    ]
    for kind in kinds:
        for _ in range(500):
            text = "".join(rng.choice(kind) for _ in range(rng.randint(0, 120)))
            ids = tokenizer.encode(text)
            assert "".join(chunks(text)) == text
            assert tokenizer.start_id not in ids and tokenizer.decode(ids, show_start=True) == text


def test_the_segmentation_itself_is_pinned(tokenizers):
    # A round trip accepts ANY way of cutting the text up. This pins the actual cuts, so an encoder that
    # ignored half its merges would be caught.
    tokenizer = tokenizers["bpe-2048"]
    pieces = [tokenizer.vocab[i] for i in tokenizer.encode("HAMLET.\nTo be, or not to be, that is the question:")]
    assert pieces == ["HAMLET", ".", "\n", "To", " be", ",", " or", " not", " to", " be", ",", " that", " is", " the", " qu", "estion", ":"]


@pytest.mark.parametrize("name", NAMES)
def test_token_counts_match_the_published_report(name, works, tokenizers):
    report = json.loads((TOKENIZERS / "report.json").read_text(encoding="utf-8"))
    assert '"test"' not in (TOKENIZERS / "report.json").read_text(encoding="utf-8")  # nothing about the test works is stored
    for set_name in ("train", "validation"):
        stream = tokenizers[name].encode_works(works[set_name])
        assert len(stream) == report["tokenizers"][name]["tokens"][set_name]
        assert stream[0] == tokenizers[name].start_id and stream.count(tokenizers[name].start_id) == len(works[set_name])


@pytest.mark.parametrize("name", NAMES)
def test_saved_files_are_exactly_what_the_code_writes(name, tokenizers, tmp_path):
    fitted_on = json.loads((TOKENIZERS / f"{name}.json").read_text(encoding="utf-8"))["fitted_on"]
    tokenizers[name].save(tmp_path / "again.json", fitted_on)
    assert (tmp_path / "again.json").read_bytes() == (TOKENIZERS / f"{name}.json").read_bytes()


def test_the_alphabet_is_the_training_alphabet(works, tokenizers):
    # This cannot catch a leak on its own: the training works happen to contain all 97 characters of the
    # corpus. The test that WOULD catch one is the next, on the merges.
    alphabet = sorted(set("".join(works["train"])))
    for tokenizer in tokenizers.values():
        assert tokenizer.chars == alphabet


def test_the_merges_come_from_the_training_works_only(works, tokenizers):
    # Learn again from the training works alone: the saved merges must come out, exactly and in order.
    # This is sensitive: see test_a_leak_would_be_noticed below.
    relearned = learn_merges(works["train"], len(tokenizers["bpe-4096"].merges))
    assert relearned == tokenizers["bpe-4096"].merges


def test_a_leak_would_be_noticed(works, tokenizers):
    # Let the validation works stand in for a leak (the test works stay closed, even for this).
    # Fitting on 41 works instead of 39 changes the list early, so the test above would fail.
    leaked = learn_merges(works["train"] + works["validation"], 200)
    assert leaked != tokenizers["bpe-4096"].merges[:200]


def test_smaller_vocabularies_are_prefixes_of_larger_ones(works, tokenizers):
    big = tokenizers["bpe-4096"].merges
    for name in ("bpe-1024", "bpe-2048"):
        small = tokenizers[name].merges
        assert small == big[:len(small)]
    assert learn_merges(works["train"], 926) == big[:926]  # and asking for fewer gives the same first ones


def test_the_fast_learner_agrees_with_the_obvious_one_on_shakespeare(works, tokenizers):
    # The slow version takes about two seconds per hundred merges on the training works.
    assert naive_learn_merges(works["train"], 120) == tokenizers["bpe-4096"].merges[:120]


@pytest.mark.parametrize("name", NAMES[1:])
def test_no_piece_straddles_two_words_or_two_lines(name, tokenizers):
    for piece in tokenizers[name].vocab[:-1]:
        assert " " not in piece[1:] or set(piece) == {" "}, f"{piece!r} has a space inside it"
        assert "\n" not in piece or set(piece) == {"\n"}, f"{piece!r} mixes a newline with text"


@pytest.mark.parametrize("name", NAMES)
def test_vocabulary_sizes_and_start(name, tokenizers):
    tokenizer = tokenizers[name]
    assert tokenizer.vocab_size == (98 if name == "char" else int(name.split("-")[1]))  # 97 characters + START
    assert tokenizer.vocab[-1] == START and tokenizer.vocab[:97] == tokenizer.chars
