"""Does GP-Thee recite its training works word for word?

Run:  uv run python scripts/memorisation.py                        (the released model; about forty minutes)
      uv run python scripts/memorisation.py --runs pilot-char-17 pilot-char-34 pilot-char-68 --scan-only
      uv run python scripts/memorisation.py --baseline-only        (no model at all: what an innocent writer scores)

Everything here was fixed in docs/BUILD_LOG.md entry 17, before it was run. In short:

  * A COPY is a run of characters occurring verbatim in the 39 training works. Raw characters are the headline,
    because "word for word" is a claim about what the model emitted; the normalised figure (lowercase letters and
    single spaces, the rule make_split.py used) is reported beside it as the sensitivity check.
  * The unit is a 50-character window, but the whole run-length curve is reported -- 20, 25, 30, 40, 50, 60, 80,
    100 characters and the longest match -- because 50 on its own is a near-certain zero for the model and for an
    innocent writer alike, and so says nothing about how close either came.
  * Every figure appears twice: over everything, and over the poet's words alone (no speaker labels, stage
    directions, scene headings or cast lists, and at least 25 letters in the passage).
  * THE INNOCENT BASELINE is leave-one-out over the 39 TRAINING works, not the validation plays. The validation
    plays were CHOSEN for sharing no 50-character window with the rest (scripts/make_split.py), so their zero is
    guaranteed and means nothing.
  * The instrument is the exhaustive scan, not sampling: walk the model along the true text, mark where its most
    likely next character is the right one, and then CONFIRM every long run by making the model write it out.

THE VERDICT, fixed in advance: the model recites if the longest confirmed passage of Shakespeare's own words it
reproduces from the training works reaches 50 characters AND is at least twice the longest such passage it
reproduces from the two validation plays it never read. It does not recite if that figure is under 50 characters
and exceeds the validation figure by no more than 10. Anything between is reported as it stands.

Writes docs/memorisation.json, and everything the model wrote to runs/<name>/sampled-grid.txt, so that the
curve can be recomputed at any width later without generating it all again.

    uv run python scripts/memorisation.py --sampled-only          (re-do the grid, keep the scan already recorded)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gp_thee.data import DATA, load_tokens, load_works
from gp_thee.memorisation import Corpus, agreement, apparatus, candidates, confirm, curve, enough_letters, plainly
from gp_thee.sampling import SUITE, generate
from gp_thee.tokenizer import load as load_tokenizer
from gp_thee.train import load_checkpoint

ROOT = Path(__file__).resolve().parent.parent
RELEASED = "sweep-char-seed-1"          # entry 17's rule, applied
TEMPERATURES = (0.0, 0.5, 0.8, 1.0)
PER_CELL = 20_000                       # characters of sampled text per (prompt kind, temperature)


def innocent(corpus: Corpus, works: list[str], width: int = 50) -> dict:
    """What a writer of Shakespeare who has memorised nothing scores: each work against the other 38.

    The hashes of every window of every work are taken once and then shared, which is the difference between
    a minute and an hour.
    """
    each = [np.array([hash(work[i:i + width]) for i in range(len(work) - width + 1)], dtype=np.int64) for work in works]
    out, shared, total = {}, 0, 0
    for i, work in enumerate(works):
        others = np.unique(np.concatenate([each[j] for j in range(len(works)) if j != i]))
        found = np.isin(each[i], others)
        where = [j for j in np.flatnonzero(found) if any(work[j:j + width] in other for k, other in enumerate(works) if k != i)]
        out[work.split("\n", 1)[0]] = {"windows": len(where), "of": int(len(found)),
                                       "example": work[where[0]:where[0] + width] if where else None}
        shared += len(where); total += len(found)
    reprints = ("THE PASSIONATE PILGRIM", "THE SONNETS", "LOVE’S LABOUR’S LOST", "VENUS AND ADONIS", "THE RAPE OF LUCRECE")
    without = [(t, r) for t, r in out.items() if t not in reprints]
    return {"all_39_works": {"windows": shared, "of": total, "share": shared / total},
            "without_the_five_that_reprint_one_another": {"windows": sum(r["windows"] for _, r in without), "of": sum(r["of"] for _, r in without),
                                                          "share": sum(r["windows"] for _, r in without) / sum(r["of"] for _, r in without)},
            "per_work": out}


def where_the_editor_wrote(works: list[str], stream: np.ndarray) -> np.ndarray:
    """The editor mask, one entry per TOKEN of the stream, so a passage found by the scan can be classified where it sits."""
    editor = np.concatenate([p for work in works for p in (apparatus(work), np.zeros(0, dtype=bool))])
    lengths = np.array([0 if int(t) == stream[0] else 1 for t in stream])   # the char tokenizer: one character a token, START none
    at = np.cumsum(lengths) - lengths                                        # where each token's character sits in the joined text
    return editor[np.minimum(at, len(editor) - 1)]


def scan_one(model, tokenizer, works: list[str], stream: np.ndarray, device: str, name: str = "", which: str = "",
             keep_mask: Path | None = None) -> dict:
    """The primary instrument, over one text: screen it all, then confirm every long candidate by making the model write it.

    Split out of scan() so that the final evaluation can point the SAME instrument at the test works. Part 7's two
    arms and part 8's third arm must differ in nothing but which text they are given; a hand-rolled variant that
    differed by one predicate would not be a third arm of the same measurement.
    """
    editor_at = where_the_editor_wrote(works, stream)
    started = time.perf_counter()
    agreed = agreement(model, stream, device)
    if keep_mask is not None:
        np.save(keep_mask, agreed)     # the sufficient statistic for any later question about run lengths
    found = candidates(agreed, least=40)
    confirmed = []
    for begin, length in found[:400]:
        got = confirm(model, tokenizer, stream, begin, length, device)
        if got["characters"] >= 30:
            run = editor_at[begin + 1:begin + 1 + got["confirmed_tokens"]]
            editor = bool(run.mean() > 0.5) or not enough_letters(got["text"])
            confirmed.append({**got, "begin": begin, "the_editors_words": editor,
                              "share_the_editor_wrote": float(run.mean()) if len(run) else 1.0})
    confirmed.sort(key=lambda c: -c["characters"])
    poet = [c for c in confirmed if not c["the_editors_words"]]
    out = {"agreement": float(agreed.mean()), "tokens": int(len(agreed)),
           "candidates_of_40_or_more": len(found), "longest_candidate": found[0][1] if found else 0,
           "confirmed": confirmed[:40],
           "longest_confirmed": confirmed[0]["characters"] if confirmed else 0,
           "longest_confirmed_poet_only": poet[0]["characters"] if poet else 0,
           "confirmed_of_50_or_more": sum(c["characters"] >= 50 for c in confirmed),
           "poet_only_of_50_or_more": sum(c["characters"] >= 50 for c in poet),
           "seconds": round(time.perf_counter() - started, 1)}
    print(f"  {name} on the {which}: {out['agreement']:.2%} of next characters are its own first guess; "
          f"{len(found)} candidates, longest confirmed {out['longest_confirmed']} characters "
          f"({out['longest_confirmed_poet_only']} of the poet's)")
    return out


def scan(model, tokenizer, corpus: Corpus, name: str, device: str) -> dict:
    """Part 7's two arms: the works it trained on, and the two plays it never read."""
    out = {}
    for which, works, split in (("training works", corpus.works, "train"), ("validation plays", load_works("validation"), "validation")):
        out[which] = scan_one(model, tokenizer, works, np.asarray(load_tokens("char", split)), device, name, which)
    return out


def sampled(model, tokenizer, corpus: Corpus, device: str, plain: Corpus, keep: Path | None = None) -> dict:
    """What the model copies when it is simply asked to write, at four temperatures and from four kinds of prompt.

    The reported object is the whole run-length curve entry 17 fixed (20, 25, 30, 40, 50, 60, 80, 100 and the
    maximum), over everything and over the poet's words alone, with the normalised figure beside it as the
    sensitivity check the same entry promised.

    Everything the model writes is kept in `keep`, which is under docs/ and not runs/ because runs/ is not in git
    and this is the evidence behind a published number: the curve can be recomputed at any width, by anyone, without
    generating it all over again.
    """
    training, validation = corpus.text, "\n".join(load_works("validation"))
    rng = np.random.default_rng(0)
    prompts = {"unprompted": ["\n\n"] * 8,
               "from the training works": [training[i:i + 256] for i in rng.integers(0, len(training) - 256, 8)],
               "from the validation plays": [validation[i:i + 256] for i in rng.integers(0, len(validation) - 256, 8)],
               "the ten-prompt suite": [p for _, p in SUITE][:8]}
    out, kept = {}, []
    for kind, these in prompts.items():
        for temperature in TEMPERATURES:
            written = []
            for i, prompt in enumerate(these):
                block = generate(model, tokenizer, prompt, characters=PER_CELL // len(these), temperature=temperature,
                                 seed=i, on_start="mask")
                written.append(block["continuation"])            # only what the model wrote: never the prompt
            text = "\n".join(written)
            kept.append(f"=== {kind} at temperature {temperature} ===\n{text}")
            whole, poet = curve(corpus, text), curve(corpus, text, poet_only=True)
            normalised = curve(plain, plainly(text))
            out[f"{kind} at temperature {temperature}"] = {"characters": len(text), "whole": whole, "poet_only": poet,
                                                           "normalised": normalised}
            print(f"  {kind:<26} t={temperature}: {whole[50]['windows']:>3} copies of 50 characters in {len(text):>6,}"
                  f" ({poet[50]['windows']} of the poet's, {normalised[50]['windows']} normalised);"
                  f" longest {whole['longest']['characters']}")
    if keep is not None:
        keep.write_text(f"Everything GP-Thee wrote for the sampled grid of docs/BUILD_LOG.md entry 18: "
                        f"{len(kept)} cells of about {PER_CELL:,} characters, four kinds of prompt at four "
                        f"temperatures, seeds fixed.\nThe prompts are never included, only the model's own writing.\n\n"
                        + "\n\n".join(kept) + "\n", encoding="utf-8")
        print(f"  wrote {keep}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--runs", nargs="+", default=[RELEASED])
    parser.add_argument("--which", default="best", choices=["best", "last"])
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--sampled-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    corpus = Corpus(load_works("train"))
    plain = Corpus([plainly(work) for work in corpus.works])   # the same corpus with layout and punctuation gone
    record = ROOT / "docs" / "memorisation.json"
    out = json.loads(record.read_text()) if record.exists() else {}
    out.setdefault("about", {})["device"] = args.device
    out["about"]["rule"] = "docs/BUILD_LOG.md entry 17, committed before this was run"

    if "innocent_baseline" not in out:
        print("the innocent baseline: each training work against the other 38")
        out["innocent_baseline"] = innocent(corpus, corpus.works)
        b = out["innocent_baseline"]
        print(f"  all 39 works: {b['all_39_works']['share']:.6f} of 50-character windows; "
              f"without the five that reprint one another: {b['without_the_five_that_reprint_one_another']['share']:.6f}")
    if "20" not in out.get("circular_baseline_the_validation_plays", {}):        # the full curve entry 17 asked for
        validation = "\n".join(load_works("validation"))
        out["circular_baseline_the_validation_plays"] = {**curve(corpus, validation),
                                                         "normalised": curve(plain, plainly(validation)),
                                                         "why_it_is_worthless": "make_split.py chose the held-out works by this very test"}
    if args.baseline_only:
        record.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote docs/memorisation.json")
        return

    tokenizer = load_tokenizer(DATA / "tokenizers" / "char.json")
    for name in args.runs:
        print(f"\n{name} ({args.which}.pt)")
        model, saved = load_checkpoint(ROOT / "runs" / name / f"{args.which}.pt", args.device)
        here = out.setdefault("runs", {}).setdefault(f"{name}/{args.which}", {"step": saved["step"], "validation_bpc": saved["facts"]["validation_bpc"]})
        if not args.sampled_only:
            here["scan"] = scan(model, tokenizer, corpus, name, args.device)
        if not args.scan_only:
            here["sampled"] = sampled(model, tokenizer, corpus, args.device, plain,
                                      keep=ROOT / "docs" / f"sampled-grid-{name}.txt")
        record.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if RELEASED + "/best" in out.get("runs", {}):
        scanned = out["runs"][RELEASED + "/best"]["scan"]
        poet, innocent_ceiling = scanned["training works"]["longest_confirmed_poet_only"], scanned["validation plays"]["longest_confirmed_poet_only"]
        verdict = ("it recites" if poet >= 50 and poet >= 2 * innocent_ceiling else
                   "it does not recite" if poet < 50 and poet - innocent_ceiling <= 10 else "between the two: report it as it stands")
        out["verdict"] = {"longest_poet_only_from_the_training_works": poet, "longest_poet_only_from_the_validation_plays": innocent_ceiling,
                          "verdict": verdict}
        print(f"\nTHE VERDICT: {verdict}. The longest passage of Shakespeare's own words it reproduces from the works it "
              f"trained on is {poet} characters; from the plays it never read, {innocent_ceiling}.")
    record.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote docs/memorisation.json")


if __name__ == "__main__":
    main()
