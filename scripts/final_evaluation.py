"""The final evaluation: the three test works, opened once, and then never again.

Run:  uv run python scripts/final_evaluation.py        (no arguments, ever; about half an hour)

*King John*, *The Tempest* and *A Lover's Complaint* have been held back since part 2. No model has read a
character of them and no number in seven parts has come from them. This script is the only thing in the project
permitted to open them, it may do so once, and what it fails to capture in that one pass is lost: a second run
would be a second look, which is the one thing seven parts of pre-registration were for.

The measurement protocol was fixed in docs/BUILD_LOG.md entry 19 BEFORE this file existed. The spent result names
the exact commit and script hash that ran. A later audit moved each primary arm's array save ahead of its summary
arithmetic and pinned that order with a unit test; that hardening is recorded in the build log and cannot authorise
another look at the test works.

THE SHAPE is three phases with a one-way door in the middle.

  Phase A, door shut, nothing at stake. The gate, then a full dress rehearsal: every code path phase B will run,
      executed against the VALIDATION works, with every arm's score checked against what its own result.json
      recorded during training. Anything may fail here at no cost, and nothing is written.

  The door. docs/final-evaluation.json is claimed with open(..., "x"), holding a stub that says the run is in
      progress, BEFORE load_works("test", ...) is called. A second invocation then refuses however the first
      ended. A crash after this point leaves the stub, which is the honest record that the measurement is spent.

  Phase B, door open. Everything that needs the test works, each in a caught block. For every primary model arm,
      the current hardened code saves the raw per-token array immediately after its forward pass and before its
      summary arithmetic. Auxiliary analyses capture their reusable arrays where applicable. A failure records
      itself and the next block runs.

  Phase C, door shut again. Assemble and persist the final JSON and headline from what phase B captured, with no
      new model pass or access to the test works.

THE GATE refuses cleanly, in this order: the output file must not exist; docs/release.json must exist, be
committed and be unchanged; src/, scripts/ and data/ must be clean; every named checkpoint must exist and the
released one must match its recorded sha256; and the rehearsal must reproduce every HEADLINE arm's validation
score EXACTLY, to all digits (entry 19 measured that the forward pass is bit-exact on this machine, so a
tolerance would only hide drift). A non-headline arm that fails is dropped and recorded as dropped.

Every invocation, refusals included, appends one line to docs/final-evaluation-attempts.jsonl before anything
else happens.

Writes docs/final-evaluation.json and docs/final-evaluation/*.npy.
"""

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gp_thee.data import DATA, corpus_alphabet, load_works
from gp_thee.evaluation import NGram, breakdown, compressed_bits_per_character, speaker_label_characters
from gp_thee.memorisation import Corpus, apparatus, curve, plainly
from gp_thee.tokenizer import load as load_tokenizer
from gp_thee.train import bits_per_character, evaluate, load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memorisation import scan_one  # noqa: E402  the same instrument part 7 used, not a copy of it

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "final-evaluation.json"
ARRAYS = ROOT / "docs" / "final-evaluation"
ATTEMPTS = ROOT / "docs" / "final-evaluation-attempts.jsonl"
UNLOCK = "this is the final evaluation"

# Entry 19's table, by name. Nothing outside this list is ever scored.
HEADLINE = [("sweep-char-seed-1", "best"), ("sweep-char-seed-2", "best"), ("sweep-char-seed-3", "best")]
RELEASED = ("sweep-char-seed-1", "best")
CHARACTER_ARMS = HEADLINE + [("sweep-char-seed-1", "last"), ("sweep-char-seed-2", "last"), ("sweep-char-seed-3", "last"),
                             ("pilot-char-34", "best"), ("pilot-char-17", "best"), ("pilot-char-68", "best")]
FRAGMENT_ARMS = [(f"sweep-bpe-{size}-seed-{seed}", "best") for size in (1024, 1536, 2048, 4096) for seed in (1, 2, 3)]
ARMS = CHARACTER_ARMS + FRAGMENT_ARMS
STRIDES = (64, 256)                      # disclosed sensitivities, deciding nothing
WIDTHS = (20, 25, 30, 40, 50, 60, 80, 100)


# ------------------------------------------------------------------------------------ bookkeeping and the gate
def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def attempt(outcome: str, detail: str = "") -> None:
    """Every invocation leaves a line here, including every refusal. Append-only, written before anything else."""
    with ATTEMPTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": now(), "outcome": outcome, "detail": detail,
                                 "commit": git("rev-parse", "HEAD", allow_failure=True),
                                 "by": __file__}, ensure_ascii=False) + "\n")


def refuse(why: str) -> None:
    attempt("refused", why)
    raise SystemExit(f"REFUSED: {why}\nThe test works were not opened. Nothing was written.")


def git(*args: str, allow_failure: bool = False) -> str:
    """git, with its return code respected: a gate built on a command that quietly failed would fail OPEN."""
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if done.returncode and not allow_failure:
        raise RuntimeError(f"git {' '.join(args)} failed ({done.returncode}): {done.stderr.strip()}")
    return done.stdout.strip()


def shown(path: Path) -> str:
    """A path for a message. A refusal must never itself raise, not even while formatting its reason."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate() -> dict:
    """Refuse cleanly, in entry 19's order, while the door is still shut."""
    if OUT.exists():
        refuse(f"{shown(OUT)} already exists. The test works have been opened once and that is the "
               "allowance. If that run failed, part 8 reports what it captured and says what was lost.")
    release = ROOT / "docs" / "release.json"
    if not release.exists():
        refuse("docs/release.json does not exist: run scripts/choose_release.py first (entry 17)")
    if not git("ls-files", "--error-unmatch", "docs/release.json", allow_failure=True):
        refuse("docs/release.json is not committed, so the released run was not fixed before this measurement")
    if git("status", "--porcelain", "docs/release.json"):
        refuse("docs/release.json has uncommitted changes: the release must be fixed before the test works open")
    dirty = [line for line in git("status", "--porcelain", "src", "scripts", "data").splitlines() if line.strip()]
    if dirty:
        refuse("src/, scripts/ or data/ is not clean, so what produced the numbers could not be pinned:\n  "
               + "\n  ".join(dirty))
    chosen = json.loads(release.read_text())
    for name, which in ARMS:
        if not (ROOT / "runs" / name / f"{which}.pt").exists():
            refuse(f"runs/{name}/{which}.pt is missing, and entry 19 fixed the arms by name")
    got = sha256_of(ROOT / "runs" / RELEASED[0] / f"{RELEASED[1]}.pt")
    if got != chosen["checkpoint_sha256"]:
        refuse(f"the released checkpoint does not match docs/release.json: {got[:12]} against "
               f"{chosen['checkpoint_sha256'][:12]}")
    return chosen


# ------------------------------------------------------------------------------------ scoring, one way only
def stream_of(works: list[str], tokenizer) -> np.ndarray:
    """A token stream built exactly as scripts/build_tokenizers.py builds train.npy and validation.npy."""
    return np.array(tokenizer.encode_works(works), dtype=np.uint16)


def characters_of(works: list[str]) -> int:
    return sum(len(work) for work in works)


def score(model, stream: np.ndarray, characters: int, device: str, stride: int | None = None) -> tuple[np.ndarray, float]:
    surprise = evaluate(model, stream, device, stride=stride)
    return surprise, bits_per_character(surprise, characters)


def recorded(name: str, which: str) -> float:
    """What this arm's own result.json wrote down during training."""
    result = json.loads((ROOT / "runs" / name / "result.json").read_text())
    return result["best"]["validation_bpc"] if which == "best" else result["final_validation_bpc"]


def tokenizer_of(name: str):
    kind = "char" if "char" in name else name.split("-")[1] + "-" + name.split("-")[2]
    return kind, load_tokenizer(DATA / "tokenizers" / f"{kind}.json")


# ------------------------------------------------------------------------------------ phase A: the rehearsal
def rehearse(device: str) -> dict:
    """Every code path phase B will run, against the validation works, with nothing at stake.

    The scores are checked against what training recorded. Entry 19 measured that this reproduces bit-exactly on
    this machine, so the headline arms are held to exact equality: a tolerance here would only hide drift.
    """
    works = load_works("validation")
    characters = characters_of(works)
    streams = {}
    checked, dropped = {}, []
    for name, which in ARMS:
        kind, tokenizer = tokenizer_of(name)
        if kind not in streams:
            streams[kind] = stream_of(works, tokenizer)
        model, _ = load_checkpoint(ROOT / "runs" / name / f"{which}.pt", device)
        _, bpc = score(model, streams[kind], characters, device)
        want = recorded(name, which)
        same = bpc == want
        checked[f"{name}/{which}"] = {"recorded": want, "rescored": bpc, "delta": bpc - want, "exact": same}
        if not same:
            if (name, which) in HEADLINE:
                refuse(f"{name}/{which} re-scores {bpc!r} on validation where training recorded {want!r}. "
                       "A headline arm that does not reproduce exactly means the pipeline has drifted since it ran.")
            dropped.append(f"{name}/{which}")
        print(f"  {name}/{which:<4} {bpc:.15f} against {want:.15f}  {'exact' if same else 'DROPPED'}", flush=True)

    # The rest of the machinery, exercised on text that is allowed to be looked at.
    training = load_works("train")
    tokenizer = load_tokenizer(DATA / "tokenizers" / "char.json")
    model, _ = load_checkpoint(ROOT / "runs" / RELEASED[0] / f"{RELEASED[1]}.pt", device)
    surprise, _ = score(model, streams["char"], characters, device)
    taken_apart = breakdown(surprise, streams["char"], tokenizer, works)
    corpus = Corpus(load_works("train"))
    plain = Corpus([plainly(work) for work in training])
    curve(corpus, works[0][:3000], widths=WIDTHS)
    curve(plain, plainly(works[0][:3000]), widths=WIDTHS)
    curve(corpus, works[0][:3000], widths=WIDTHS, poet_only=True)

    # The bar, recomputed on validation and checked against what part 5 published.
    bar = baseline_table(works, training)
    published = json.loads((ROOT / "docs" / "baselines.json").read_text())["results"]
    bar_check = {}
    for name, row in bar.items():
        if name == "per_work_compressors" or name not in published:
            continue
        want, got = published[name].get("bits_per_character"), row.get("bits_per_character")
        bar_check[name] = {"published": want, "recomputed": got,
                           "close": want is not None and abs(got - want) < 1e-9}
    wrong = [n for n, r in bar_check.items() if not r["close"]]
    if wrong:
        refuse(f"the baseline table does not reproduce docs/baselines.json on the validation works: {wrong}. "
               "The bar the test score will be set against cannot be trusted, so the works stay shut.")
    print(f"  the bar reproduces docs/baselines.json on all {len(bar_check)} rows", flush=True)

    # Phase B's OWN functions, on validation, with their arrays sent somewhere harmless. The fatal bug this
    # rehearsal was extended to catch lived in _headline, which nothing before phase B had ever called.
    global ARRAYS
    real_arrays, ARRAYS = ARRAYS, ROOT / "docs" / "final-evaluation-rehearsal"
    dry_failures: dict = {}
    try:
        ARRAYS.mkdir(parents=True, exist_ok=True)
        dry = {"works": [{"title": w.split("\n", 1)[0], "characters": len(w)} for w in works],
               "characters": characters, "arms": {}}
        for name, which in (RELEASED, ("sweep-char-seed-2", "best"), ("sweep-bpe-1024-seed-1", "best")):
            kind, tok = tokenizer_of(name)
            dry["arms"][f"{name}/{which}"] = _one_arm(name, which, stream_of(works, tok), tok, works,
                                                      characters, device)
        dry["masks"] = _save_arrays(dry["arms"], {"char": streams["char"]}, works,
                                    [w.split("\n", 1)[0] for w in works])
        dry["memorisation"] = _memorisation(model, tokenizer, corpus, works,
                                            [w.split("\n", 1)[0] for w in works], streams["char"], device)
        dry["stride_sensitivity"] = strides_on_validation = _strides(model, streams["char"], characters, device)
        dry["cross_device"] = _cross_device(streams["char"], characters, dry["arms"][f"{RELEASED[0]}/{RELEASED[1]}"],
                                            tokenizer, dry["memorisation"]["scan"])
        dry["headline"] = _headline({**dry, "about": {"released": {"step": 0}}})
        json.dumps({"rehearsal": checked, "bar": bar_check, "dry": dry}, ensure_ascii=False, default=float)
        _report({**dry, "about": {"minutes": 0.0}, "failures": {}}, spent=False)
    finally:
        for stale in ARRAYS.glob("*.npy"):
            stale.unlink()
        ARRAYS.rmdir() if ARRAYS.exists() else None
        ARRAYS = real_arrays
    print("  every phase B function ran on the validation works and serialised", flush=True)

    return {"checked": checked, "dropped": dropped, "arms_scored": len(ARMS) - len(dropped),
            "the_bar_reproduces_part_5": bar_check,
            "stride_sensitivity_on_validation": strides_on_validation,
            "the_scan_on_validation": {k: v for k, v in dry["memorisation"]["scan"].items() if k != "confirmed"},
            "cross_device_on_validation": dry["cross_device"]}


# ------------------------------------------------------------------------------------ phase B: the measurement
def measure(device: str, dropped: list[str], failures: dict) -> dict:
    """Everything that needs the three works. Every block is caught; a failure costs its block and nothing else."""
    def block(name, function, *args, **kwargs):
        started = time.perf_counter()
        try:
            got = function(*args, **kwargs)
            print(f"  {name}: {time.perf_counter() - started:.1f}s", flush=True)
            return got
        except Exception:
            failures[name] = traceback.format_exc()
            print(f"  {name}: FAILED, recorded, continuing\n{failures[name]}", flush=True)
            return None

    ARRAYS.mkdir(parents=True, exist_ok=True)              # before anything tries to save into it
    works = load_works("test", unlock=UNLOCK)               # <<< the door is now open
    characters = characters_of(works)
    titles = [work.split("\n", 1)[0] for work in works]
    out = {"works": [{"title": t, "characters": len(w)} for t, w in zip(titles, works)],
           "characters": characters}

    tokenizers, streams, arms = {}, {}, {}
    for name, which in ARMS:
        if f"{name}/{which}" in dropped:
            continue
        kind = block(f"{name}/{which} stream", _stream_for, name, works, tokenizers, streams)
        if kind is None:
            continue                                        # its tokenizer or stream failed; recorded, move on
        got = block(f"{name}/{which}", _one_arm, name, which, streams[kind], tokenizers[kind], works, characters, device)
        if got is not None:
            arms[f"{name}/{which}"] = got
    out["arms"] = arms

    # These masks make the already-scored arrays independently re-sliceable by work and kind of text.
    block("arrays", _save_arrays, arms, streams, works, titles)

    released = f"{RELEASED[0]}/{RELEASED[1]}"
    model, _ = load_checkpoint(ROOT / "runs" / RELEASED[0] / f"{RELEASED[1]}.pt", device)
    corpus = Corpus(load_works("train"))
    # The character stream and tokenizer are needed by every block below; the released model is a character arm,
    # so they were built in the loop above, but build them here too rather than subscript a dict that a failed
    # arm might have left empty — a KeyError in an argument list would escape block() and reach the door.
    char_tokenizer = tokenizers.get("char") or load_tokenizer(DATA / "tokenizers" / "char.json")
    char_stream = streams.get("char")
    if char_stream is None:
        char_stream = stream_of(works, char_tokenizer)
    # Model-dependent first: those are the ones that would cost a forward pass to redo, and redoing is forbidden.
    out["memorisation"] = block("memorisation", _memorisation, model, char_tokenizer, corpus, works,
                                titles, char_stream, device)
    out["stride_sensitivity"] = block("strides", _strides, model, char_stream, characters, device)
    out["cross_device"] = block("cross-device", _cross_device, char_stream, characters, arms.get(released),
                                char_tokenizer, (out.get("memorisation") or {}).get("scan"))
    out["the_bar_on_this_text"] = block("baselines", baseline_table, works, load_works("train"))
    out["arrays"] = block("arrays", _files_written)
    return out


def _files_written() -> dict:
    """Taken at the end, so that every array written during phase B is in the list rather than only the early ones."""
    return {"files": sorted(p.name for p in ARRAYS.glob("*.npy"))}


def _stream_for(name, works, tokenizers, streams) -> str:
    """The tokenizer and stream this arm needs, built once per kind. Inside block(), like everything after the door."""
    kind, tokenizer = tokenizer_of(name)
    tokenizers.setdefault(kind, tokenizer)
    if kind not in streams:
        streams[kind] = stream_of(works, tokenizer)
    return kind


def _one_arm(name, which, stream, tokenizer, works, characters, device) -> dict:
    kind_of = lambda n: "char" if "char" in n else n.split("-")[1] + "-" + n.split("-")[2]
    model, saved = load_checkpoint(ROOT / "runs" / name / f"{which}.pt", device)
    surprise = evaluate(model, stream, device)
    surprise_path = ARRAYS / f"{name}--{which}.npy"
    np.save(surprise_path, surprise)                            # save the irreplaceable evidence before reducing it
    bpc = bits_per_character(surprise, characters)
    row = {"test_bpc": bpc, "validation_bpc": recorded(name, which), "step": saved.get("step"),
           "tokenizer": kind_of(name), "nats": float(surprise.sum()),
           "surprise_file": f"final-evaluation/{name}--{which}.npy"}
    if kind_of(name) == "char":                                 # entry 19: breakdown for the character arms only
        row["taken_apart"] = breakdown(surprise, stream, tokenizer, works)
    return row


def _save_arrays(arms, streams, works, titles) -> dict:
    """The masks that make every saved surprise array re-sliceable forever, without a second look."""
    text = "".join(works)
    work_index = np.concatenate([np.full(len(w), i, dtype=np.int16) for i, w in enumerate(works)])
    labels = np.concatenate([speaker_label_characters(w) for w in works])
    editor = np.concatenate([apparatus(w) for w in works])
    for name, array in (("work-index", work_index), ("speaker-label-mask", labels), ("editor-mask", editor),
                        ("token-stream-char", streams["char"])):
        np.save(ARRAYS / f"{name}.npy", array)
    return {"characters": len(text), "files": sorted(p.name for p in ARRAYS.glob("*.npy")),
            "editor_share": float(editor.mean()), "speaker_label_share": float(labels.mean()),
            "note": "the three works have been committed in the repository since part 2; these arrays "
                    "publish nothing new and live under docs/ so that no training glob can sweep them up"}


def baseline_table(works: list[str], training: list[str]) -> dict:
    """The bar on a given text: line for line the table scripts/baselines.py builds for the validation works.

    Not imported from that script, which has no main() guard and would refit eight n-grams and overwrite
    docs/baselines.json on import. The library calls are identical, and the rehearsal checks that this
    reproduces docs/baselines.json on the validation works before the test works are opened.
    """
    alphabet = corpus_alphabet()
    on_label = np.concatenate([speaker_label_characters(work) for work in works])
    edges = np.cumsum([len(work) for work in works])[:-1]
    out = {"blind guess": {"bits_per_character": math.log2(len(alphabet))}}
    for order in range(1, 9):
        bits = NGram(order, alphabet).fit(training).bits(works)
        name = "character counts" if order == 1 else f"{order}-gram (looks back {order - 1})"
        out[name] = {"bits_per_character": float(bits.mean()),
                     "speaker_labels": float(bits[on_label].mean()) if on_label.any() else None,
                     "everything_else": float(bits[~on_label].mean()),
                     "per_work": {work.split("\n", 1)[0]: float(part.mean())
                                  for work, part in zip(works, np.split(bits, edges))}}
    text = "".join(works)
    alone = compressed_bits_per_character(text)
    primed = compressed_bits_per_character(text, after_reading="".join(training))
    for name in alone:
        out[name] = {"bits_per_character": alone[name], "after_reading_the_training_works": primed[name]}
    out["per_work_compressors"] = {work.split("\n", 1)[0]: compressed_bits_per_character(work) for work in works}
    return out


def _memorisation(model, tokenizer, corpus, works, titles, stream, device) -> dict:
    """Part 7's third arm, and the copy curve that makes it readable."""
    text = "\n".join(works)
    plain = Corpus([plainly(w) for w in corpus.works])
    scanned = _scan_on(model, tokenizer, works, stream, device)
    return {"scan": scanned,
            "the_test_works_against_the_39": curve(corpus, text, widths=WIDTHS),
            "normalised": curve(plain, plainly(text), widths=WIDTHS),
            "poet_only": curve(corpus, text, widths=WIDTHS, poet_only=True),
            "editor_share": {t: float(apparatus(w).mean()) for t, w in zip(titles, works)},
            "part_7_for_comparison": {"training works": 44, "validation plays": 0,
                                      "what": "longest confirmed passage of the poet's own words, entry 18"}}


def _scan_on(model, tokenizer, works, stream, device) -> dict:
    """Part 7's third arm: scan_one(), the identical function its two published arms came from."""
    return scan_one(model, tokenizer, works, stream, device, "GP-Thee-11M", "three works it never read",
                    keep_mask=ARRAYS / "agreement-released.npy")


def _strides(model, stream, characters, device) -> dict:
    return {str(s): score(model, stream, characters, device, stride=s)[1] for s in STRIDES}


def _cross_device(stream, characters, released_row, tokenizer, scanned) -> dict:
    """Entry 17 asked for one named device with a fixed slice re-run on the other, because argmax is discontinuous.

    So this re-scores the released model on the CPU, and re-confirms the longest passage the scan found there too:
    bits per character would survive a tiny float difference, and an argmax would not.
    """
    model, _ = load_checkpoint(ROOT / "runs" / RELEASED[0] / f"{RELEASED[1]}.pt", "cpu")
    _, bpc = score(model, stream, characters, "cpu")
    on_gpu = released_row["test_bpc"] if released_row else None
    out = {"bits_per_character": {"cpu": bpc, "gpu": on_gpu,
                                  "difference": (bpc - on_gpu) if on_gpu is not None else None}}
    longest = ((scanned or {}).get("confirmed") or [None])[0]
    if longest:
        from gp_thee.memorisation import confirm
        again = confirm(model, tokenizer, stream, longest["begin"], longest["screened_tokens"], "cpu")
        out["the_longest_confirmed_passage"] = {
            "on the gpu": {"characters": longest["characters"], "text": longest["text"]},
            "on the cpu": {"characters": again["characters"], "text": again["text"]},
            "unchanged": again["text"] == longest["text"]}
    return out


# ------------------------------------------------------------------------------------ phase C, and the whole
def main() -> None:
    started = time.perf_counter()
    ATTEMPTS.parent.mkdir(parents=True, exist_ok=True)
    attempt("started")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("the gate")
    chosen = gate()
    print("\nthe dress rehearsal, on the validation works, with nothing at stake")
    rehearsal = rehearse(device)

    stub = {"status": "in progress", "began_at": now(), "commit": git("rev-parse", "HEAD"),
            "note": "If this is what you are reading, the works were opened and the run did not finish. "
                    "That is the honest record: the measurement is spent, and part 8 reports what was captured."}
    with OUT.open("x", encoding="utf-8") as handle:                 # <<< the door is claimed before it is opened
        json.dump(stub, handle, indent=2, ensure_ascii=False)

    print("\nthe measurement")
    failures: dict = {}
    measured = measure(device, rehearsal["dropped"], failures)

    out = {"about": {"what": "The final evaluation. The three test works, opened once.",
                     "rule": "docs/BUILD_LOG.md entry 19, committed before this script existed",
                     "began_at": stub["began_at"], "finished_at": now(),
                     "minutes": round((time.perf_counter() - started) / 60, 1),
                     "device": device, "torch": torch.__version__, "python": platform.python_version(),
                     "machine": platform.platform(), "commit": git("rev-parse", "HEAD"),
                     "script_sha256": sha256_of(Path(__file__)),
                     "tokenizer_sha256": sha256_of(DATA / "tokenizers" / "char.json"),
                     "split_sha256": sha256_of(ROOT / "data" / "processed" / "split.json"),
                     "released": {"run": RELEASED[0], "checkpoint": RELEASED[1], "step": chosen["step"],
                                  "sha256": chosen["checkpoint_sha256"]}},
           "rehearsal": rehearsal, "failures": failures, **measured}

    # Phase C is guarded the way phase B is. Everything the works cost is already in `out`; the headline is
    # computed inside a net, and the complete record is written before the report. A bug in either records itself as a
    # failure and rewrites the file rather than dying with only the stub on disk. Entry 19 fixed this discipline
    # for the blocks after the door; it simply never extended past them until an audit found a phase-C crash.
    def commit_out() -> None:
        beside = OUT.with_suffix(".writing")               # the stub is only replaced once the new file is whole
        beside.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
        beside.replace(OUT)

    try:
        out["headline"] = _headline(out)
    except Exception:
        failures["headline"] = traceback.format_exc()
    commit_out()
    attempt("completed", f"{len(out.get('arms', {}))} arms, {len(failures)} failures")
    try:
        _report(out)
    except Exception:
        failures["report"] = traceback.format_exc()
        commit_out()                                       # so the recorded failure is on disk, not only on screen
        print(f"the report failed but the measurement is safe on disk:\n{failures['report']}")


def _headline(out: dict) -> dict:
    """Entry 17's two headlines, and nothing else. Every other number in the file is secondary."""
    arms = out.get("arms", {})
    released = arms.get(f"{RELEASED[0]}/{RELEASED[1]}", {})
    present = [n for n, _ in HEADLINE if f"{n}/best" in arms]
    missing = [n for n, _ in HEADLINE if f"{n}/best" not in arms]
    three = [arms[f"{n}/best"]["test_bpc"] for n in present]
    per_work = (released.get("taken_apart") or {}).get("works", [])     # a LIST of dicts, each with a title
    leave_one_out = {}
    for row in per_work:
        rest = [r for r in per_work if r is not row]        # by identity, so two like-titled works cannot collide
        bits = sum(r["bits"] for r in rest)
        chars = sum(r["characters"] for r in rest)
        leave_one_out[f"without {row['title']}"] = bits / chars if chars else None
    return {
        "the_artifact": {"model": "GP-Thee-11M", "run": RELEASED[0], "step": out["about"]["released"]["step"],
                         "test_bits_per_character": released.get("test_bpc"),
                         "its_validation_bits_per_character": released.get("validation_bpc"),
                         "leave_one_work_out": leave_one_out},
        "the_recipe": {"runs": present, "runs_that_failed_and_are_not_in_this_mean": missing, "test": three,
                       "mean": (sum(three) / len(three)) if three else None,
                       "spread": (max(three) - min(three)) if three else None,
                       "released_rank_on_test": (sorted(three).index(released["test_bpc"]) + 1)
                       if three and "test_bpc" in released else None,
                       "note": "Entry 17 pre-registered two headlines: what the artifact scores, and what the "
                               "recipe produces. The three eligible runs were a tie on validation by the sweep's "
                               "own threshold, and the release is the rule's output, not the best model."},
    }


def _report(out: dict, spent: bool = True) -> None:
    """Print the outcome. `spent` is False when the rehearsal calls this on its dry dict, so the closing line does
    not claim the works were opened when the door was still shut."""
    head = out["headline"]
    print("\n" + "=" * 78)
    if out.get("failures"):
        print(f"{len(out['failures'])} block(s) FAILED and were recorded: {', '.join(out['failures'])}\n")
    figure = lambda v: "MISSING (its block failed; see failures)" if v is None else f"{v:.4f}"
    print(f"GP-Thee-11M on the three works it never read: "
          f"{figure(head['the_artifact']['test_bits_per_character'])} bits per character")
    print(f"  (its validation score, which two layers of selection sit on: "
          f"{figure(head['the_artifact']['its_validation_bits_per_character'])})")
    recipe = head["the_recipe"]
    if recipe["test"]:
        n = len(recipe["test"])
        print(f"\nThe recipe, {n} eligible run{'s' if n != 1 else ''} on test: "
              + ", ".join(f"{v:.4f}" for v in recipe["test"])
              + f"\n  mean {recipe['mean']:.4f}, spread {recipe['spread']:.4f}, "
                f"the released run ranks {recipe['released_rank_on_test']} of {n}")
    scan_row = (out.get("memorisation") or {}).get("scan")
    if scan_row:
        longest, poet = scan_row["longest_confirmed"], scan_row["longest_confirmed_poet_only"]
        print(f"\nPart 7's third arm: the longest passage confirmed is {longest} characters; the longest that is "
              f"mostly the poet's own words is {poet}.\n  (On the training works these were 66 and 44; on the two "
              "validation plays, 40 and 0. Entry 18.)")
    print(f"\nwrote {shown(OUT)} in {out['about']['minutes']} minutes")
    if spent:
        print("The test works are spent. They have now taken part in something.")
    print("=" * 78)


if __name__ == "__main__":
    main()
