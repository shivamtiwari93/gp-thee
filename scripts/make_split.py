"""Freeze the train / validation / test split, and prove the held-out works do not leak.

Run:  uv run python scripts/make_split.py

Reads   data/processed/manifest.json and the 44 cleaned works
Writes  data/processed/split.json

Why split at all: a model this size can memorise 5 MB of text. Its score on text it trained on says
nothing about whether it learned how Shakespeare writes. So some works are set aside and never trained
on, and the model is judged on those.

  train       what the model learns from
  validation  looked at during training: when to stop, which settings are better
  test        looked at ONCE, at the very end. Validation stops being a fair judge the moment we
              use it to make choices, because we keep whatever happens to score well on it.

Why WHOLE works, and not random chunks or the last 10% of the file:
  * Random chunks of a play sit next to chunks that were trained on: same scene, same speakers, same
    names. The model would be tested on text it has half seen, and the score would flatter it.
  * The last 10% of this corpus is most of The Two Gentlemen of Verona, all of The Two Noble Kinsmen and
    The Winter's Tale, and all five poems: 53% romance, 33% poetry, and not one tragedy or history
    (measured). It also includes The Passionate Pilgrim, whose poems reappear in the training text.

How the five held-out works were chosen:
  1. One per genre across the two sets: a comedy and a tragedy for validation; a history, a romance and
     a poem for test.
  2. Only works that are Shakespeare's alone. A score on a scene John Fletcher wrote would not measure
     what we care about.
  3. Only works that share no passage with any other work. Measured below, not assumed.
  4. About 5% of the corpus each: enough text for a stable score, little enough to leave 90% for training.

The model we MEASURE never reads Romeo and Juliet. The model we RELEASE is retrained on all 44 works,
for the number of steps the measured run found best, so it knows every play.

Changing this file's lists after any model has been trained would make earlier results incomparable.
"""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"

VALIDATION = ["ALL’S WELL THAT ENDS WELL", "THE TRAGEDY OF ROMEO AND JULIET"]
TEST = ["THE LIFE AND DEATH OF KING JOHN", "THE TEMPEST", "A LOVER’S COMPLAINT"]

# Works that must stay in training because they share text with another work (measured in BUILD_LOG entry 9):
# The Passionate Pilgrim reprints Sonnets 138 and 144 and three poems from Love's Labour's Lost;
# 2 Henry IV quotes Richard II; Lucrece and Venus share their dedication header.
SHARES_TEXT = ["THE SONNETS", "LOVE’S LABOUR’S LOST", "THE PASSIONATE PILGRIM", "THE SECOND PART OF KING HENRY THE FOURTH",
               "KING RICHARD THE SECOND", "THE RAPE OF LUCRECE", "VENUS AND ADONIS"]
# Widely attributed in part to other dramatists (general scholarship, not derivable from the text).
COLLABORATIONS = ["THE TWO NOBLE KINSMEN", "KING HENRY THE EIGHTH", "PERICLES, PRINCE OF TYRE", "THE LIFE OF TIMON OF ATHENS",
                  "THE TRAGEDY OF TITUS ANDRONICUS", "THE FIRST PART OF HENRY THE SIXTH", "THE PASSIONATE PILGRIM"]

LEAK_WINDOW = 50  # characters, after normalising
# The one overlap we accept: a scene heading and the word "Enter", in King John and The Winter's Tale.
# It is an editor's heading, not a line of the play.
KNOWN_OVERLAPS = {("THE LIFE AND DEATH OF KING JOHN", "THE WINTER’S TALE"): "e exeunt scene ii the same a room of state in the palace enter "}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"CHECK FAILED: {message}\nNothing was written.")


def normalise(text: str) -> str:
    """Lowercase letters and single spaces only, so that layout and punctuation cannot hide a copy."""
    text = text.lower().replace("’", "").replace("‘", "")
    return re.sub(r"\s+", " ", "".join(ch if ch.isalpha() else " " for ch in text)).strip()


manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
works = {w["title"]: w for w in manifest["works"]}
held_out = VALIDATION + TEST

# ---------------------------------------------------------------- 1. the lists make sense
check(all(t in works for t in held_out + SHARES_TEXT + COLLABORATIONS), "a title is not in the manifest")
check(len(set(held_out)) == len(held_out), "a work is in both validation and test")
check(not set(held_out) & set(SHARES_TEXT), "a held-out work shares text with another work")
check(not set(held_out) & set(COLLABORATIONS), "a held-out work is a collaboration")
check({works[t]["genre"] for t in held_out} == {"comedy", "tragedy", "history", "romance", "poetry"}, "held-out works do not cover all five genres")
train = [t for t in works if t not in held_out]

# ---------------------------------------------------------------- 2. the files are the ones the manifest describes
bodies = {}
for title, w in works.items():
    body = (OUT / w["file"]).read_bytes()
    check(hashlib.sha256(body).hexdigest() == w["sha256"], f"{w['file']} does not match the manifest; re-run prepare_data.py")
    bodies[title] = normalise(body.decode("utf-8"))

# ---------------------------------------------------------------- 3. leak check, measured
# For each held-out work, look for ANY 50-character run that also appears in any other work.
overlaps = {}
for held in held_out:
    h = bodies[held]
    windows = {}
    for i in range(len(h) - LEAK_WINDOW + 1):
        windows.setdefault(h[i:i + LEAK_WINDOW], i)
    for other, o in bodies.items():
        if other == held:
            continue
        positions = sorted({windows[o[j:j + LEAK_WINDOW]] for j in range(len(o) - LEAK_WINDOW + 1) if o[j:j + LEAK_WINDOW] in windows})
        if positions:
            check(positions == list(range(positions[0], positions[-1] + 1)), f"{held} / {other}: more than one shared passage")
            overlaps[(held, other)] = h[positions[0]:positions[-1] + LEAK_WINDOW]
check(overlaps == KNOWN_OVERLAPS, f"held-out text also appears elsewhere: {overlaps}")

# ---------------------------------------------------------------- 4. sizes
total = sum(w["chars"] for w in works.values())
share = {name: sum(works[t]["chars"] for t in titles) / total for name, titles in [("train", train), ("validation", VALIDATION), ("test", TEST)]}
check(0.04 <= share["validation"] <= 0.06 and 0.04 <= share["test"] <= 0.06, f"held-out sets should each be about 5%: {share}")

split = {
    "note": "Frozen before any training. Whole works only. See scripts/make_split.py for the reasoning.",
    "leak_check": {"window_chars": LEAK_WINDOW, "normalisation": "lowercase, apostrophes deleted, non-letters to spaces, whitespace collapsed",
                   "accepted_overlaps": [{"held_out": a, "other": b, "text": text, "why": "an editor's scene heading plus 'Enter', not a line of the play"}
                                         for (a, b), text in overlaps.items()]},
    "sets": {name: {"works": len(titles), "chars": sum(works[t]["chars"] for t in titles), "share_of_chars": round(share[name], 4),
                    "files": [works[t]["file"] for t in titles]}
             for name, titles in [("train", train), ("validation", VALIDATION), ("test", TEST)]},
}
(OUT / "split.json").write_bytes((json.dumps(split, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

for name, s in split["sets"].items():
    print(f"{name:<11}{s['works']:>3} works {s['chars']:>10,} chars {s['share_of_chars']:>7.1%}")
    if name != "train":
        for f in s["files"]:
            title = next(t for t, w in works.items() if w["file"] == f)
            print(f"{'':<14}{works[title]['genre']:<8} {works[title]['chars']:>8,}  {title}")
print(f"\nleak check: {len(held_out)} held-out works x {len(works) - 1} others, window {LEAK_WINDOW} chars: "
      f"{len(overlaps)} overlap, the accepted scene heading")
print(f"wrote {(OUT / 'split.json').relative_to(ROOT)}")
