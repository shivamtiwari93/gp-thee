"""The innocent baseline at every width entry 17 asked for, exactly, in one pass per width.

Run:  uv run python scripts/baseline_curve.py        (*measured*: 52 seconds; writes docs/baseline-curve.json)

What an innocent writer of Shakespeare scores: leave-one-out over the 39 TRAINING works, each against the other
38. Not the validation plays, whose zero `scripts/make_split.py` guaranteed by choosing them for it.

`scripts/memorisation.py:innocent()` answers the same question at 50 characters by hashing every window and then
confirming each hit with a substring search of the other works. That is fine at 50, where hits are rare, and
quadratic at 20, where they are not. So this asks it the other way round: for each width, build a dictionary from
the window's own TEXT to the set of works holding it, and read leave-one-out straight off the index. The keys are
the text, so no hash can invent a copy, and the whole curve costs one pass per width.

The two programs share no code path and agree exactly at 50 characters -- 916 shared windows of 4,809,425, and 202
of 4,425,256 once the five works that reprint one another are set aside -- which is the only reason to trust
either of them.
"""

import json
from collections import defaultdict
from pathlib import Path

from gp_thee.data import load_works
from gp_thee.memorisation import apparatus, enough_letters

ROOT = Path(__file__).resolve().parent.parent
WIDTHS = (20, 25, 30, 40, 50, 60, 80, 100)
# The Passionate Pilgrim reprints sonnets that also appear in The Sonnets and Love's Labour's Lost; Venus and
# Adonis and The Rape of Lucrece share their dedication. Real reprints, not memorisation, so they are reported apart.
REPRINTS = ("THE PASSIONATE PILGRIM", "THE SONNETS", "LOVE’S LABOUR’S LOST", "VENUS AND ADONIS", "THE RAPE OF LUCRECE")


def leave_one_out(works: list[str], editors: list, width: int) -> dict:
    """For each work, how many of its `width`-character windows occur in one of the others, and how many are his."""
    holders = defaultdict(set)
    for i, work in enumerate(works):
        for j in range(len(work) - width + 1):
            holders[work[j:j + width]].add(i)
    rows = {}
    for i, work in enumerate(works):
        shared = poet = 0
        for j in range(len(work) - width + 1):
            window = work[j:j + width]
            if len(holders[window]) > 1:                        # some other work holds this exact text
                shared += 1
                if editors[i][j:j + width].mean() <= 0.5 and enough_letters(window):
                    poet += 1
        rows[work.split("\n", 1)[0]] = {"windows": shared, "poet_only": poet, "of": len(work) - width + 1}
    return rows


def totals(rows: list[dict]) -> dict:
    out = {"windows": sum(r["windows"] for r in rows), "poet_only": sum(r["poet_only"] for r in rows),
           "of": sum(r["of"] for r in rows)}
    return {**out, "share": out["windows"] / out["of"], "poet_only_share": out["poet_only"] / out["of"]}


def main() -> None:
    works = load_works("train")
    editors = [apparatus(work) for work in works]
    out = {"about": "leave-one-out over the 39 training works; docs/BUILD_LOG.md entries 17 and 18"}
    print(f"{'width':>6}{'all 39 works':>22}{'without the five':>22}{'the poet’s only':>22}")
    for width in WIDTHS:
        rows = leave_one_out(works, editors, width)
        every = list(rows.values())
        keep = [r for title, r in rows.items() if title not in REPRINTS]
        out[str(width)] = {"all_39_works": totals(every), "without_the_five_that_reprint_one_another": totals(keep),
                           "per_work": rows}
        a, b = out[str(width)]["all_39_works"], out[str(width)]["without_the_five_that_reprint_one_another"]
        print(f"{width:>6}{a['share']:>12.6f} ({a['windows']:>6,}){b['share']:>12.6f} ({b['windows']:>6,})"
              f"{b['poet_only_share']:>13.7f} ({b['poet_only']:>4,})", flush=True)
    (ROOT / "docs" / "baseline-curve.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nThe poet-only column is blank below 30 characters by construction, not by finding: a window must hold\n"
          "25 letters to count as the poet's, and a window of 25 characters cannot.\n\nwrote docs/baseline-curve.json")


if __name__ == "__main__":
    main()
