"""Fit the tokenizers on the TRAINING works, check them, and turn the corpus into token ids.

Run:  uv run python scripts/build_tokenizers.py

Reads   the training and validation works (never the test works)
Writes  data/tokenizers/char.json, bpe-1024.json, bpe-1536.json, bpe-2048.json, bpe-4096.json   (committed)
        data/tokenizers/report.json                                                             (committed)
        data/tokens/<tokenizer>/train.npy, validation.npy                        (not committed: rebuilt by this script)

Everything a tokenizer learns, it learns from the 39 training works. The alphabet is the set of characters in
those works. The merges are counted in those works. The validation works are only encoded with the result.

The test works are not opened at all, not even to check that they can be encoded. They do not need to be:
the alphabet of the whole corpus was recorded when it was cleaned, before the split existed, and this script
checks that the training works contain all of it. A tokenizer that can encode any string over that alphabet
(which the tests establish) can encode any work.

If that check ever fails, the remedy is decided now, before it can be tempting: a cleaning rule for the whole
corpus in prepare_data.py, or a new split drawn before any model is trained. The alphabet is never widened
from held-out text.

The BPE sizes are one tokenizer cut at several lengths. Merges are learned in order, most frequent first,
so the first 926 merges of the 4096 tokenizer ARE the 1024 tokenizer. We learn the longest list once.
(The candidate sizes date from the research stage, before the split existed, and came from counts over the
whole corpus. The choice AMONG them will be made on validation text only.)

No token stream is saved for the test works and no number about them is printed or stored: a file's size
would give away its token count, and a careless `*.npy` would sweep it into training. The final evaluation
encodes them itself.
"""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from gp_thee.data import PROCESSED, corpus_alphabet, load_works
from gp_thee.tokenizer import START, BPETokenizer, CharTokenizer, learn_merges, load

ROOT = Path(__file__).resolve().parent.parent
TOKENIZERS = ROOT / "data" / "tokenizers"
TOKENS = ROOT / "data" / "tokens"
BPE_SIZES = [1024, 1536, 2048, 4096]  # total vocabulary: characters + merges + START
SETS = ("train", "validation")  # never "test": see the docstring

# The model these vocabularies are for (blog part 1): 6 layers, width 384, context 256 tokens.
LAYERS, WIDTH, CONTEXT = 6, 384, 256


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"CHECK FAILED: {message}")


# ---------------------------------------------------------------- 1. the works, as frozen in split.json (checksums verified on load)
split_bytes = (PROCESSED / "split.json").read_bytes()
works = {name: load_works(name) for name in SETS}
train = works["train"]
check([len(w) for w in works.values()] == [39, 2], "expected 39 training and 2 validation works")

# ---------------------------------------------------------------- 2. fit: alphabet and merges, from the training works only
chars = sorted(set("".join(train)))
missing = sorted(set(corpus_alphabet()) - set(chars))
check(not missing, f"the corpus uses characters the training works lack, so some work could not be encoded: {missing!r}")

started = time.perf_counter()
merges = learn_merges(train, max(BPE_SIZES) - len(chars) - 1)
learning_seconds = time.perf_counter() - started
check(len(merges) == max(BPE_SIZES) - len(chars) - 1, f"ran out of pairs to merge after {len(merges)}")

tokenizers = {"char": CharTokenizer(chars)}
for size in BPE_SIZES:
    tokenizers[f"bpe-{size}"] = BPETokenizer(chars, merges[:size - len(chars) - 1])
    check(tokenizers[f"bpe-{size}"].vocab_size == size, f"bpe-{size} has the wrong vocabulary size")

# ---------------------------------------------------------------- 3. save, reload, and check that nothing is lost
fitted_on = {"split_sha256": hashlib.sha256(split_bytes).hexdigest(), "set": "train", "works": len(train),
             "chars": sum(map(len, train)), "text_sha256": hashlib.sha256("".join(train).encode("utf-8")).hexdigest(),
             "note": "alphabet and merges computed from the 39 training works only; the validation works were only "
                     "encoded; the test works were not opened"}
TOKENIZERS.mkdir(parents=True, exist_ok=True)
everything = [text for name in SETS for text in works[name]]
report = {"fitted_on": fitted_on, "alphabet": len(chars), "chars": {s: sum(map(len, works[s])) for s in SETS},
          "note": "token counts include one START per work; chars_per_token leaves START out. Nothing here describes the test works.",
          "tokenizers": {}}

for name, tokenizer in tokenizers.items():
    tokenizer.save(TOKENIZERS / f"{name}.json", fitted_on)
    reloaded = load(TOKENIZERS / f"{name}.json")
    check(reloaded.vocab == tokenizer.vocab, f"{name}: vocabulary changed when saved and loaded")

    # The promise: every work survives the round trip, character for character, and encoding text never
    # produces START.
    for text in everything:
        ids = reloaded.encode(text)
        check(reloaded.start_id not in ids and reloaded.decode(ids, show_start=True) == text, f"{name}: a work did not survive encode then decode")

    # One stream of ids per saved set, START before each work. uint16 holds ids up to 65,535.
    check(reloaded.vocab_size <= 2 ** 16, f"{name}: vocabulary too large for uint16")
    streams = {set_name: np.array(reloaded.encode_works(works[set_name]), dtype=np.uint16) for set_name in SETS}
    (TOKENS / name).mkdir(parents=True, exist_ok=True)
    for set_name, stream in streams.items():
        expected = "".join(START + text for text in works[set_name])  # START shown, so a stray or missing one is caught
        check(reloaded.decode(stream.tolist(), show_start=True) == expected, f"{name}: the saved {set_name} stream does not decode to its works")
        np.save(TOKENS / name / f"{set_name}.npy", stream)

    # How the vocabulary is used. Only training and validation are looked at.
    use_train = Counter(streams["train"].tolist())
    use_val = Counter(streams["validation"].tolist())
    V = reloaded.vocab_size
    chars_per_token = {s: sum(map(len, works[s])) / (len(streams[s]) - len(works[s])) for s in ("train", "validation")}
    report["tokenizers"][name] = {
        "vocab_size": V,
        "merges": len(reloaded.merges),
        "tokens": {s: int(len(streams[s])) for s in ("train", "validation")},
        "chars_per_token": {s: round(v, 3) for s, v in chars_per_token.items()},
        "context_in_chars": round(CONTEXT * chars_per_token["train"]),
        "vocab_seen_100_times_in_training": round(sum(1 for i in range(V) if use_train[i] >= 100) / V, 4),
        # 25 rare characters and START can never reach 100, whatever the size. Merged pieces alone:
        "merges_seen_100_times_in_training": round(sum(1 for i in range(len(chars), V - 1) if use_train[i] >= 100) / max(1, V - 1 - len(chars)), 4),
        "vocab_seen_under_10_times_in_training": sum(1 for i in range(V) if use_train[i] < 10),
        # Pieces swallowed whole by a later merge ("YRACUS" inside "YRACUSE") stay in the vocabulary, unused.
        "vocab_never_used_in_training": sum(1 for i in range(V) if use_train[i] == 0),
        "vocab_never_used_in_validation": sum(1 for i in range(V) if use_val[i] == 0),
        "model_parameters": 12 * WIDTH * WIDTH * LAYERS + (2 * LAYERS + 1) * WIDTH + V * WIDTH + CONTEXT * WIDTH,
    }

(TOKENIZERS / "report.json").write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

# ---------------------------------------------------------------- 4. report
print(f"alphabet: {len(chars)} characters, all from the training works. Learned {len(merges):,} merges in {learning_seconds:.1f} s.\n")
print(f"{'tokenizer':<10}{'vocab':>7}{'train tokens':>14}{'chars/token':>13}{'  on validation':>15}{'context':>10}{'merges seen 100+':>18}{'never used':>12}{'parameters':>13}")
for name, r in report["tokenizers"].items():
    print(f"{name:<10}{r['vocab_size']:>7,}{r['tokens']['train']:>14,}{r['chars_per_token']['train']:>13.2f}{r['chars_per_token']['validation']:>15.2f}"
          f"{r['context_in_chars']:>7,} ch{r['merges_seen_100_times_in_training']:>18.1%}{r['vocab_never_used_in_training']:>12,}{r['model_parameters']:>13,}")

bpe = tokenizers["bpe-2048"]
print("\nfirst 12 merges:", "  ".join(repr(a + b) for a, b in merges[:12]))
print("merges 1000-1011:", "  ".join(repr(a + b) for a, b in merges[1000:1012]))
for sample in ["HAMLET.\nTo be, or not to be, that is the question:", "ROMEO.\nBut soft, what light through yonder window breaks?"]:
    print(f"\nbpe-2048 on {sample!r}:\n  " + " | ".join(bpe.vocab[i].replace("\n", "\\n") for i in bpe.encode(sample)))
print(f"\nwrote {len(tokenizers)} tokenizers to {TOKENIZERS.relative_to(ROOT)}/ and train/validation token streams to {TOKENS.relative_to(ROOT)}/")
