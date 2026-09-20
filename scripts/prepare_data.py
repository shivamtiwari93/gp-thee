"""Turn the raw Project Gutenberg file into a clean corpus: one text file per work.

Run:  uv run python scripts/prepare_data.py

Reads   data/raw/100-0.txt                  (never modified)
Writes  data/processed/works/NN-title.txt   44 files, one per work
        data/processed/manifest.json        what each file is, sizes, checksums, rule counts

The guiding rule: remove what an editor or transcriber added, keep what Shakespeare's readers
would see, and never reword a line of verse or prose. The only wording this script changes is in the
editor's apparatus: nine ditto lines in one cast list, and two misspelt speaker labels.

Two things protect the corpus, and they do different jobs. The SHA-256 checksum guarantees the
WORDS: this is byte for byte the file we studied. The assertions guarantee the STRUCTURE: every
rule states exactly how many lines it touches, so a rule that quietly matches too much or too
little stops the script. Nothing is written until every check has passed, so a failed check can never
leave a half-cleaned corpus on disk. This is the only corpus there is.

All line numbers here are 1-indexed lines of the RAW file. Rules never shift them: lines are
marked as dropped, or edited in place, and only assembled into files at the very end.

Standard library only. Output is deterministic: same input, same bytes out, on any platform.
"""

import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "100-0.txt"
OUT = ROOT / "data" / "processed"
RAW_SHA256 = "a023115c2d4e2ee12221bdd780fdf2ac5a864fe225948656f51f8be462c7fffb"
FIRST_WORK_LINE = 63  # lines 1-62 are the Gutenberg marker, a title block and a table of contents

# Genre is knowledge from OUTSIDE the text. It is used only to build a balanced
# train/validation/test split, and is never shown to the model.
GENRES = {
    "poetry": ["THE SONNETS", "A LOVER’S COMPLAINT", "THE PASSIONATE PILGRIM", "THE PHOENIX AND THE TURTLE",
               "THE RAPE OF LUCRECE", "VENUS AND ADONIS"],
    "comedy": ["ALL’S WELL THAT ENDS WELL", "AS YOU LIKE IT", "THE COMEDY OF ERRORS", "LOVE’S LABOUR’S LOST",
               "MEASURE FOR MEASURE", "THE MERCHANT OF VENICE", "THE MERRY WIVES OF WINDSOR",
               "A MIDSUMMER NIGHT’S DREAM", "MUCH ADO ABOUT NOTHING", "THE TAMING OF THE SHREW",
               "TWELFTH NIGHT; OR, WHAT YOU WILL", "THE TWO GENTLEMEN OF VERONA"],
    "history": ["THE FIRST PART OF KING HENRY THE FOURTH", "THE SECOND PART OF KING HENRY THE FOURTH",
                "THE LIFE OF KING HENRY THE FIFTH", "THE FIRST PART OF HENRY THE SIXTH",
                "THE SECOND PART OF KING HENRY THE SIXTH", "THE THIRD PART OF KING HENRY THE SIXTH",
                "KING HENRY THE EIGHTH", "THE LIFE AND DEATH OF KING JOHN", "KING RICHARD THE SECOND",
                "KING RICHARD THE THIRD"],
    "tragedy": ["THE TRAGEDY OF ANTONY AND CLEOPATRA", "THE TRAGEDY OF CORIOLANUS",
                "THE TRAGEDY OF HAMLET, PRINCE OF DENMARK", "THE TRAGEDY OF JULIUS CAESAR",
                "THE TRAGEDY OF KING LEAR", "THE TRAGEDY OF MACBETH", "THE TRAGEDY OF OTHELLO, THE MOOR OF VENICE",
                "THE TRAGEDY OF ROMEO AND JULIET", "THE LIFE OF TIMON OF ATHENS",
                "THE TRAGEDY OF TITUS ANDRONICUS", "TROILUS AND CRESSIDA"],
    "romance": ["CYMBELINE", "PERICLES, PRINCE OF TYRE", "THE TEMPEST", "THE TWO NOBLE KINSMEN", "THE WINTER’S TALE"],
}
GENRE_OF = {title: genre for genre, titles in GENRES.items() for title in titles}
POEMS = set(GENRES["poetry"]) - {"THE SONNETS"}

counts: Counter = Counter()  # how many lines each rule touched (0 if never touched); saved in the manifest


def check(condition: bool, message: str) -> None:
    """Like assert, but cannot be switched off with `python -O`."""
    if not condition:
        raise SystemExit(f"CHECK FAILED: {message}\nNothing was written.")


def slug(index: int, title: str) -> str:
    ascii_title = unicodedata.normalize("NFKD", title.replace("’", "")).encode("ascii", "ignore").decode()
    return f"{index:02d}-" + re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")


def indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


# ---------------------------------------------------------------- 0. load, and check it is the file we think it is
data = RAW.read_bytes()
check(hashlib.sha256(data).hexdigest() == RAW_SHA256, "raw file checksum changed; every count below assumes this exact file")
check(not data.startswith(b"\xef\xbb\xbf") and b"\r" not in data, "unexpected BOM or carriage returns")
text = data.decode("utf-8")
check(unicodedata.normalize("NFC", text) == text, "text is not in Unicode NFC form")
check(sum(len(titles) for titles in GENRES.values()) == 44, "a title is listed under two genres")

lines = text.split("\n")
check(lines[-1] == "", "file should end with a newline")
raw = [None] + lines[:-1]  # raw[1] is line 1, so indices match every line number in the comments
N = len(raw) - 1
check(N == 196_022, f"expected 196,022 lines, got {N:,}")

edited = list(raw)  # the text we will output; edits go here, raw[] stays pristine for the checks
dropped: dict[int, str] = {}  # raw line number -> name of the rule that removed it


def blank(i: int) -> bool:
    return raw[i].strip() == ""  # a few "blank" lines hold a single space


def drop(line_numbers, rule: str) -> None:
    for i in line_numbers:
        check(i not in dropped, f"line {i} removed twice ({dropped.get(i)} and {rule})")
        dropped[i] = rule
        counts[rule] += 1


def edit(i: int, before: str, after: str, rule: str) -> None:
    """Replace one exact line. Fails unless the line still reads exactly as expected."""
    check(edited[i] == before, f"line {i} is not {before!r}")
    edited[i] = after
    counts[rule] += 1


# ---------------------------------------------------------------- 1. Project Gutenberg markers and front matter
check(raw[1] == "*** START OF THE PROJECT GUTENBERG EBOOK 100 ***", "line 1 is not the START marker")
check(raw[N] == "*** END OF THE PROJECT GUTENBERG EBOOK 100 ***", "last line is not the END marker")
drop([1, N], "gutenberg_markers")

check(raw[6] == "The Complete Works of William Shakespeare" and raw[8] == "by William Shakespeare", "title block moved")
check(raw[13].strip() == "Contents" and raw[13].startswith(" "), "the file-level Contents heading (the indented one) moved")
titles = [raw[i].strip() for i in range(15, 59)]
check(all(raw[i].startswith("    ") and raw[i].strip() for i in range(15, 59)) and blank(14) and blank(59), "file-level Contents list moved")
check(len(set(titles)) == 44 and set(titles) == set(GENRE_OF), "the 44 titles do not match the genre table")
drop(range(2, FIRST_WORK_LINE), "front_matter")  # title block + the 44-entry table of contents

# ---------------------------------------------------------------- 2. where each of the 44 works starts and ends
# A heading is the title at column 0, with 3+ blank lines before and one after. The text alone is not
# enough: "KING HENRY THE EIGHTH" and "KING RICHARD THE SECOND" also appear as cast-list entries, and
# Richard II's real heading is "THE LIFE AND DEATH OF KING RICHARD THE SECOND". Requiring EXACTLY one
# match per title means a stray look-alike can never silently move a boundary.
# (3 blank lines is the minimum, not the norm: All's Well has only 3 because "THE END" sits above them.)
def heading_lines(title: str) -> list[int]:
    return [i for i in range(FIRST_WORK_LINE, N)
            if (raw[i] == title or raw[i].endswith(" " + title)) and raw[i] == raw[i].strip()
            and blank(i - 1) and blank(i - 2) and blank(i - 3) and blank(i + 1)]


starts = []
for title in titles:
    hits = heading_lines(title)
    # Venus and Adonis prints its title twice, before and after the dedication; section 4 drops the second.
    expected_hits = 2 if title == "VENUS AND ADONIS" else 1
    check(len(hits) == expected_hits, f"{title!r}: expected {expected_hits} heading line(s), found {hits}")
    starts.append(hits[0])
headings = [raw[i] for i in starts]
check(starts == sorted(starts) and starts[0] == FIRST_WORK_LINE and starts[-1] == 194_588, "work headings are out of order or moved")
renamed = [(t, h) for t, h in zip(titles, headings) if t != h]
check(renamed == [("KING RICHARD THE SECOND", "THE LIFE AND DEATH OF KING RICHARD THE SECOND")], f"unexpected heading/title differences: {renamed}")

END_MARKER = re.compile(r"\s*(THE END|FINIS)\s*")  # only 3 of the 44 works have one, so they teach nothing
end_markers = [i for i in range(FIRST_WORK_LINE, N) if END_MARKER.fullmatch(raw[i])]
check(end_markers == [2836, 186_322, 196_020], f"end-of-work markers moved: {end_markers}")
drop(end_markers, "end_markers")

ends = []
for k, start in enumerate(starts):
    nxt = starts[k + 1] if k + 1 < len(starts) else N  # N is the END marker
    end = nxt - 1
    while blank(end) or end in end_markers:
        end -= 1
    check(start < end, f"empty work: {titles[k]}")
    ends.append(end)
    drop([i for i in range(end + 1, nxt) if i not in end_markers], "blank_lines_between_works")
check(counts["blank_lines_between_works"] == 181, f"blank lines between works: {counts['blank_lines_between_works']}")
followed_by_end_marker = ["THE SONNETS", "THE TWO NOBLE KINSMEN", "VENUS AND ADONIS"]
check([ends[titles.index(t)] for t in followed_by_end_marker] == [2833, 186_320, 196_015], "work endings are not where expected")

# ---------------------------------------------------------------- 3. each play's own table of contents
# Every play opens with an editor's list of acts and scenes, then the cast list ("Dramatis Personæ").
# The list cannot be told from the real headings by how it looks: 85 of its ACT lines are byte-identical
# to the real ones. Position is the only reliable rule: from the line "Contents" to the line before
# "Dramatis Personæ". The cast list itself is kept: it is part of the play as readers meet it.
contents = [i for i in range(FIRST_WORK_LINE, N) if raw[i] == "Contents"]
dramatis = [i for i in range(FIRST_WORK_LINE, N) if raw[i].strip() == "Dramatis Personæ"]
check(len(contents) == 38 and len(dramatis) == 38, "expected 38 plays with a Contents list and a cast list")
plays = [k for k, t in enumerate(titles) if t not in POEMS and t != "THE SONNETS"]
check(len(plays) == 38, "expected 38 plays")
for k, c, d in zip(plays, contents, dramatis):
    check(starts[k] < c < d < ends[k], f"Contents/cast list outside its play: {titles[k]}")
    check(24 <= d - c <= 97, f"Contents list has an implausible length in {titles[k]}: {d - c}")
    check(not any("[" in raw[i] or "]" in raw[i] for i in range(c, d)), f"stage direction inside a Contents list: {titles[k]}")
    drop(range(c, d), "play_contents_lists")
    # The blank lines between the title and "Contents" vary from 1 to 4. Drop them too; section 6 puts
    # back exactly 4, so that all 38 plays open the same way.
    check(all(blank(i) for i in range(starts[k] + 1, c)), f"text between the title and Contents: {titles[k]}")
    drop(range(starts[k] + 1, c), "blank_lines_after_play_title")
check((contents[0], dramatis[0] - 1, contents[-1], dramatis[-1] - 1) == (2844, 2883, 186_332, 186_358), "first/last Contents list moved")
check(counts["play_contents_lists"] == 1488, f"Contents lists: expected 1,488 lines, got {counts['play_contents_lists']}")
check(counts["blank_lines_after_play_title"] == 141, f"blank lines after play titles: {counts['blank_lines_after_play_title']}")
BLANK_LINES_AFTER_PLAY_TITLE = 4  # what 33 of the 38 plays already had

for i in dramatis:  # seven of the 38 cast-list headings carry a stray space
    if raw[i] != "Dramatis Personæ":
        edit(i, raw[i], "Dramatis Personæ", "cast_heading_spaces")
check(counts["cast_heading_spaces"] == 7, "expected 7 cast-list headings with stray spaces")

# ---------------------------------------------------------------- 4. line numbers glued onto Venus and Adonis
# An editor numbered every fourth line in the margin: "...love he laugh’d to scorn;        4".
# That is 68% of all the digits in the corpus. The gap is 1 to 22 spaces, so the rule must allow a single
# space (line 195658). Run on the whole file, a looser pattern would also eat the 154 sonnet numbers, so
# the rule is confined to this poem. Some of the numbers are wrong, so match their shape, never their value.
MARGIN_NUMBER = re.compile(r"(?<=\S) +\d+$")
venus = titles.index("VENUS AND ADONIS")
numbered = [i for i in range(starts[venus], ends[venus] + 1) if MARGIN_NUMBER.search(raw[i])]
for i in numbered:
    edited[i] = MARGIN_NUMBER.sub("", edited[i])
counts["venus_margin_numbers"] = len(numbered)
check(len(numbered) == 294 and numbered[0] == 194_627 and numbered[-1] == 196_013, f"Venus margin numbers: {len(numbered)}")
check(not any(ch.isdigit() for i in range(starts[venus], ends[venus] + 1) for ch in edited[i]), "digits remain in Venus and Adonis")

# The poem's title is printed twice, before and after the dedication. Keep the first. Dropping the second
# alone would fuse the blank lines around it into a run of 7, longer than any in the source, so three of
# them go with it and the usual 4 remain.
check(raw[194_588] == raw[194_622] == "VENUS AND ADONIS", "duplicate Venus heading moved")
check(all(blank(i) for i in (*range(194_616, 194_622), 194_623)), "blank lines around the duplicate Venus heading moved")
drop([194_622], "duplicate_heading")
drop([194_620, 194_621, 194_623], "blank_lines_around_duplicate_heading")

# ---------------------------------------------------------------- 5. editor's marks and transcription slips
# A row of asterisks stands in for a lost line in The Passionate Pilgrim IX. It is the editor's mark,
# not the poet's, and the only line of pure punctuation in the corpus. Dropping it is a choice: the poem
# then reads as complete although a line is missing. Three similar marks sit INSIDE lines of verse
# ("But . . . ." in Titus, "Could … get me." in King John, "_. . . odours" in the Dream). They are kept,
# because removing them would mean rewording a line of verse, which this script never does.
asterisms = [i for i in range(FIRST_WORK_LINE, N) if re.fullmatch(r"\s*\*(\s+\*)+\s*", raw[i])]
check(asterisms == [191_908], f"lacuna marker moved: {asterisms}")
drop(asterisms, "lacuna_marker")

# Square brackets mean "the editor supplied this". Two are not stage directions at all.
edit(7790, "[EPILOGUE]", "EPILOGUE", "editorial_brackets")  # the other 6 epilogue headings have no brackets
edit(8694, " Enter Octavius [Caesar], Lepidus and their train.", " Enter Octavius Caesar, Lepidus and their train.", "editorial_brackets")

# The Julius Caesar cast list uses ditto marks ( ” ” ” ) for "same as the line above". They are typesetting
# shorthand, and they make 27 of the corpus's closing quote marks mean something else. Spell them out.
DITTO = re.compile(r",\s*”\s+”\s+”")
check(raw[80_276].endswith(", Triumvir after his death.") and raw[80_280].endswith(", Conspirator against Caesar."), "Julius Caesar cast list moved")
dittos = [i for i in range(FIRST_WORK_LINE, N) if DITTO.search(raw[i])]
check(dittos == [80_277, 80_278, *range(80_281, 80_288)], f"ditto marks moved: {dittos}")
for i in dittos:
    description = ", Triumvir after his death." if i < 80_280 else ", Conspirator against Caesar."
    edit(i, raw[i], DITTO.sub(description, raw[i]), "ditto_marks_spelled_out")
    check(edited[i].endswith(description) and "”" not in edited[i], f"ditto line {i} not fully replaced")

# Slips made when the text was typed up: two misspelt speaker names that would otherwise become two
# one-off characters, a misplaced italic marker, a stray bracket and a missing one. Each is pinned to its
# exact line and wording. After these, every "[" in the corpus has its "]".
for i, before, after in [
    (166_396, "ANDARUS.", "PANDARUS."),    # 152 other speeches say PANDARUS.
    (128_939, "HELCANUS.", "HELICANUS."),  # 35 other speeches say HELICANUS.
    (72_750, "[A_side to Suffolk_.] This priest has no pride in him?", "[_Aside to Suffolk_.] This priest has no pride in him?"),
    (93_358, "[_Reads_.] [_So sweet a kiss the golden sun gives not", "[_Reads_.] _So sweet a kiss the golden sun gives not"),
    (64_573, "and the rest.", "and the rest.]"),  # closes "[Alarums to the fight, ..." on the line above
]:
    edit(i, before, after, "transcription_slips")

tabs = [i for i in range(1, N + 1) if "\t" in raw[i] and i not in dropped]
check(tabs == [61_045, 88_058, 127_705, 188_411], f"tab characters moved: {tabs}")
for i in tabs:
    edit(i, raw[i], re.sub(r" ?\t ?", " ", raw[i]), "tabs_replaced")

check([i for i in range(1, N + 1) if "'" in raw[i]] == [166_731] and '"' not in text, "expected one straight apostrophe, no straight double quotes")
edit(166_731, raw[166_731], raw[166_731].replace("'", "’"), "straight_apostrophes_curled")  # the other 26,338 are curly

trailing = [i for i in range(1, N + 1) if i not in dropped and edited[i] != edited[i].rstrip()]
check(trailing == [57_239, 87_273], f"lines with trailing whitespace: {trailing}")
for i in trailing:
    edit(i, edited[i], edited[i].rstrip(), "trailing_whitespace")

# A single leading space is typing residue, found in 22 of the works: on 2,464 stage-direction lines in 17
# works (so "[_Exeunt._]" and "Enter" each had two spellings), on 15 scene headings in Hamlet, and on 31
# stray lines of dialogue. Real verse indentation in this file is always 2 or more spaces, and is never
# touched. The exceptions are six lines of song in Much Ado About Nothing, where the single space is one
# step of a stepped layout.
STEPPED_SONG_LINES = {119_048, 119_056, 121_799, 121_801, 121_810, 121_818}
check(all(indent(raw[i]) == 1 for i in STEPPED_SONG_LINES), "stepped song lines moved")
for i in range(FIRST_WORK_LINE, N):
    if i not in dropped and i not in STEPPED_SONG_LINES and indent(edited[i]) == 1:
        edit(i, edited[i], edited[i][1:], "one_space_indents")
check(counts["one_space_indents"] == 2510, f"one-space indents: {counts['one_space_indents']}")

# Doubled spaces inside a line are typing residue ("Enter  Doctor Butts."). Spaces after an opening
# italic marker are kept: that is how this file indents songs ("_    Fear no more the heat o’ th’ sun,").
DOUBLED_SPACE = re.compile(r"(?<=[^\s_]) {2,}(?=\S)")
for i in range(FIRST_WORK_LINE, N):
    if i not in dropped and DOUBLED_SPACE.search(edited[i]):
        edit(i, edited[i], DOUBLED_SPACE.sub(" ", edited[i]), "doubled_spaces")
check(counts["doubled_spaces"] == 47, f"doubled spaces: {counts['doubled_spaces']}")

# ---------------------------------------------------------------- 6. assemble each work in memory
sonnet_numbers = [int(edited[i]) for i in range(starts[0], ends[0] + 1) if re.fullmatch(r" {20}\d{1,3}", edited[i])]
check(sonnet_numbers == list(range(1, 155)), "the 154 sonnet numbers are not intact")

works, bodies, alphabet = [], {}, Counter()
for k, (title, start, end) in enumerate(zip(titles, starts, ends), start=1):
    kept = [edited[i] for i in range(start, end + 1) if i not in dropped]
    is_play = (k - 1) in plays
    if is_play:
        check(kept[1] == "Dramatis Personæ", f"{title}: the cast list should follow the title")
        kept[1:1] = [""] * BLANK_LINES_AFTER_PLAY_TITLE
    check(kept[0] == headings[k - 1] and kept[-1].strip() != "", f"{title}: should start with its heading and end with text")
    body = "\n".join(kept) + "\n"
    name = f"works/{slug(k, title)}.txt"
    bodies[name] = body
    alphabet.update(body)
    works.append({
        "index": k, "title": title, "heading": headings[k - 1], "file": name,
        "kind": "play" if is_play else "sonnets" if title == "THE SONNETS" else "poem",
        "genre": GENRE_OF[title], "raw_first_line": start, "raw_last_line": end,
        "lines": len(kept), "words": len(body.split()), "chars": len(body),
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    })
counts["blank_lines_inserted_after_play_titles"] = BLANK_LINES_AFTER_PLAY_TITLE * len(plays)

# ---------------------------------------------------------------- 7. check the result, not just the steps
corpus = "".join(bodies.values())
out_lines = corpus.split("\n")[:-1]
check("gutenberg" not in corpus.lower() and not set("*\t'\"") & set(corpus), "clutter survived")
check(not any(re.fullmatch(r"contents|the end\.?|finis\.?", line.strip(), re.IGNORECASE) for line in out_lines), "a Contents heading or end marker survived")
check(sum(line == "Dramatis Personæ" for line in out_lines) == 38, "expected 38 cast lists")
check(sum(bool(re.fullmatch(r"ACT [IVX]+\.?", line)) for line in out_lines) == 190, "expected 190 ACT headings (38 plays x 5)")
check(sum(bool(re.match(r"(SCENE|Scene) [IVX]+\.(\s|$)", line)) for line in out_lines) == 776, "expected 776 numbered scene headings")
check(sum(bool(re.search(r"\d$", line)) for line in out_lines) == 154, "only the 154 sonnet numbers should end in a digit")
check(corpus.count("[") == corpus.count("]") == 4252, f"brackets are not balanced: {corpus.count('[')} [ and {corpus.count(']')} ]")
check(corpus.count("_") == 9702 and corpus.count("_") % 2 == 0, "italic markers changed")
check(corpus.count("”") == 1059 - 27, "closing quote marks: the 27 ditto marks should be the only ones gone")
check(corpus.count("“") == 1232, "opening quote marks changed")
check(sum(ch.isdigit() for ch in corpus) == 427, "digits: expected 427 (sonnet numbers, a tavern bill, a few speaker labels)")
inner_space_runs = [line for line in out_lines if re.search(r"\S {2,}\S", line)]
check(len(inner_space_runs) == 23 and all(re.match(r" *_ {2,}", line) for line in inner_space_runs), "a doubled space survived outside the 23 song indents")
check(not any(line != line.rstrip() for line in out_lines), "trailing whitespace survived")
check(sum(indent(line) == 1 for line in out_lines) == len(STEPPED_SONG_LINES), "a one-space indent survived")
longest_blank_run = max(len(run) for run in re.findall(r"\n{2,}", corpus)) - 1
check(longest_blank_run == 4, f"longest run of blank lines is {longest_blank_run}, expected 4")
check(len(alphabet) == 97, f"expected an alphabet of 97 characters, got {len(alphabet)}")

# Every raw line is either in a file or in `dropped`, and the only lines added are the play-opening blanks.
check(len(out_lines) == N - len(dropped) + counts["blank_lines_inserted_after_play_titles"], "line accounting is off")
check(len(out_lines) == sum(w["lines"] for w in works), "per-work line counts do not add up")

# ---------------------------------------------------------------- 8. only now, write
manifest = {
    "source": {"file": "data/raw/100-0.txt", "sha256": RAW_SHA256, "bytes": len(data), "lines": N,
               "words": len(text.split()), "chars": len(text), "distinct_chars": len(set(text))},
    "output": {"works": len(works), "lines": len(out_lines), "words": sum(w["words"] for w in works),
               "chars": len(corpus), "distinct_chars": len(alphabet)},
    "rules": dict(counts),
    "alphabet": [{"char": ch, "codepoint": f"U+{ord(ch):04X}", "name": "LINE FEED" if ch == "\n" else unicodedata.name(ch),
                  "count": n} for ch, n in sorted(alphabet.items(), key=lambda item: (-item[1], item[0]))],
    "works": works,
}
# Write into a temporary folder and swap it in, so an interrupted run cannot leave a mix of old and new.
staging = OUT / "works.tmp"
shutil.rmtree(staging, ignore_errors=True)
staging.mkdir(parents=True)
for name, body in bodies.items():
    (staging / Path(name).name).write_bytes(body.encode("utf-8"))  # bytes, so Windows cannot turn "\n" into "\r\n"
shutil.rmtree(OUT / "works", ignore_errors=True)
staging.rename(OUT / "works")
(OUT / "manifest.json").write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

# ---------------------------------------------------------------- 9. report
src, out = manifest["source"], manifest["output"]
print(f"raw      {src['lines']:>9,} lines {src['words']:>9,} words {src['chars']:>10,} chars {src['distinct_chars']:>4} distinct characters")
print(f"cleaned  {out['lines']:>9,} lines {out['words']:>9,} words {out['chars']:>10,} chars {out['distinct_chars']:>4} distinct characters")
print(f"change   {out['lines'] - src['lines']:>+9,} lines {out['words'] - src['words']:>+9,} words "
      f"{out['chars'] - src['chars']:>+10,} chars ({(src['chars'] - out['chars']) / src['chars']:.2%} of the file removed)\n")
print(f"lines dropped: {len(dropped):,}   lines edited in place: {sum(raw[i] != edited[i] for i in range(1, N + 1) if i not in dropped):,}\n")
for rule, n in counts.items():
    print(f"  {rule:<42}{n:>6,}")
print("\nby genre:")
for genre in GENRES:
    group = [w for w in works if w["genre"] == genre]
    print(f"  {genre:<10}{len(group):>3} works {sum(w['chars'] for w in group):>10,} chars {sum(w['chars'] for w in group) / out['chars']:>7.1%}")
rare = [a for a in manifest["alphabet"] if a["count"] < 50]
print(f"\n{len(rare)} of the {out['distinct_chars']} characters occur fewer than 50 times: " + " ".join(a["char"] for a in rare))
print(f"\nwrote {len(works)} files to {(OUT / 'works').relative_to(ROOT)}/ and {(OUT / 'manifest.json').relative_to(ROOT)}")
