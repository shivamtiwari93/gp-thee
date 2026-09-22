"""Every example blog part 8 shows, generated and measured here so that none of them is invented.

Run:  uv run python scripts/demonstrate.py        (about three minutes on the GPU)

Part 8 has to answer a reader who says "this is just replaying the plays verbatim". That cannot be answered with
prose, only with measurements, so every demonstration in the write-up comes out of this script:

  * WHAT IT PREDICTS: the model's actual probability distribution over the next character, which is the literal
    definition of a language model.
  * WHAT IT WRITES: the sampler's features, each exercised on a fixed prompt with a fixed seed.
  * WHETHER IT COPIES: for every sample, the longest stretch that occurs anywhere in the 39 training works,
    found by gp_thee.memorisation.Corpus. This is the direct test of the verbatim objection.
  * WHAT IT INVENTS: words and phrases from its own output looked up in the corpus and found absent.
  * WHERE IT BREAKS: the failures, shown rather than described.

Writes docs/demonstrations.json (the record) and docs/demonstrations.txt (readable, for quoting in the blog).
"""

import json
from pathlib import Path

import torch

from gp_thee.data import DATA, load_works
from gp_thee.memorisation import Corpus
from gp_thee.sampling import SCENE, SUITE, as_a_character, generate, normalise
from gp_thee.tokenizer import load as load_tokenizer
from gp_thee.train import load_checkpoint

ROOT = Path(__file__).resolve().parent.parent
RUN = "sweep-char-seed-1"

# Prompts chosen to show one thing each, and fixed here so the blog and the record cannot drift apart.
WRITES = [
    ("a speaker label", "\n\nHAMLET.\n", 0.8, 0, 300),
    ("a scene heading", "SCENE III. Rome. The Capitol.\n\nEnter CAESAR and the Conspirators.\n\nCAESAR.\n", 0.8, 7, 330),
    ("a famous opening", "\n\nHAMLET.\nTo be, or not to be, that is the", 0.8, 0, 240),
    ("a sonnet number that does not exist", "\n\n\n                    155\n\nWhen first I looked upon thy", 0.8, 0, 260),
    ("a modern sentence", "The meeting is at nine o’clock, and the budget", 0.8, 0, 200),
    ("an instruction it cannot follow", "Write me a poem about a cat.", 0.8, 0, 200),
    ("a stage direction", "\n\n[_Enter a Messenger._]\n\nMESSENGER.\n", 0.8, 3, 240),
]
TEMPERATURES = (0.0, 0.5, 0.8, 1.2)
PROBE_CONTEXTS = [
    "\n\nHAMLET.\nTo be, or not to b",
    "SCENE III. Rome. The Capit",
    "\n\nROMEO.\nBut soft, what li",
    "\n\nKING HENRY.\nOnce more unto the b",
]
# The line the model ought to know best, and its real run-up, to test the verbatim objection at its strongest.
FAMOUS = "and at last by Marcus\nBrutus._]\n\nCAESAR.\n"
NORMALISER = ["naïve", "I'll not be \"quoted\"", "Señor", "a\ttab", "plain english"]


def longest_copy(corpus: Corpus, text: str) -> dict:
    length, piece = corpus.longest_in(text)
    return {"characters": length, "passage": piece, "of": len(text)}


def main() -> None:
    tokenizer = load_tokenizer(DATA / "tokenizers" / "char.json")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, saved = load_checkpoint(ROOT / "runs" / RUN / "best.pt", device)
    # load_checkpoint does NOT put the model in eval mode, and the probability probe below calls the model
    # directly rather than through generate() (which does set it). Without this, dropout of 0.2 is live and the
    # distributions are randomly thinned — the first run of this script reported 'e' at 81% after "not to b"
    # where the real figure is 97%. generate() and evaluate() were always correct; only the probe was wrong.
    model.eval()
    corpus = Corpus(load_works("train"))
    out: dict = {"about": {"run": RUN, "checkpoint": "best.pt", "step": saved["step"], "device": device,
                           "alphabet": len(tokenizer.chars),
                           "note": "every example in blog part 8 comes from this file"}}

    # ------------------------------------------------------------------ 1. it is a probability distribution
    print("what it predicts")
    predictions = []
    for context in PROBE_CONTEXTS:
        ids = torch.tensor([tokenizer.encode(context)], device=device)
        with torch.no_grad():
            probs = torch.softmax(model(ids)[0][0, -1].float(), dim=-1)
        top = torch.topk(probs, 6)
        row = {"context": context,
               "top": [{"character": tokenizer.decode([int(i)]), "probability": float(p)}
                       for p, i in zip(top.values, top.indices)]}
        predictions.append(row)
        print(f"  {context[-24:]!r} -> " + "  ".join(f"{c['character']!r}={c['probability']:.1%}" for c in row["top"][:4]))
    out["what_it_predicts"] = predictions

    # ------------------------------------------------------------------ 2. what it writes, and whether it copies
    print("\nwhat it writes (and the longest stretch of it that is really Shakespeare's)")
    wrote = []
    for label, prompt, temperature, seed, characters in WRITES:
        got = generate(model, tokenizer, prompt, characters=characters, temperature=temperature, seed=seed)
        row = {"label": label, "prompt": prompt, "continuation": got["continuation"], "why_it_stopped": got["why"],
               "notes": got["notes"], "settings": got["settings"],
               "longest_verbatim": longest_copy(corpus, got["continuation"])}
        wrote.append(row)
        print(f"  {label:36} {len(got['continuation']):>4} chars, longest verbatim "
              f"{row['longest_verbatim']['characters']:>3}  {row['longest_verbatim']['passage']!r}")
    out["what_it_writes"] = wrote

    # ------------------------------------------------------------------ 3. the temperature dial, one prompt
    print("\nthe same prompt at four temperatures")
    dial = []
    for temperature in TEMPERATURES:
        got = generate(model, tokenizer, "\n\nJULIET.\n", characters=200, temperature=temperature, seed=1)
        row = {"temperature": temperature, "continuation": got["continuation"], "why_it_stopped": got["why"],
               "longest_verbatim": longest_copy(corpus, got["continuation"])}
        dial.append(row)
        first = got["continuation"].strip().split("\n")[0][:58]
        print(f"  t={temperature:<4} longest verbatim {row['longest_verbatim']['characters']:>3}   {first!r}")
    out["the_temperature_dial"] = dial

    # ------------------------------------------------------------------ 4. speaking as a character
    print("\nspeaking as a character")
    scenes = []
    for answerer in (None, "MACBETH"):
        got = as_a_character(model, tokenizer, SCENE, answerer=answerer, temperature=0.8, seed=0)
        scenes.append({"answerer_asked_for": answerer, "answerer": got["answerer"],
                       "cast_by_the_model": got["cast_by_the_model"], "reply": got["reply"],
                       "transcript": got["transcript"],
                       "longest_verbatim": longest_copy(corpus, got["reply"])})
        print(f"  answerer={got['answerer']:<10} cast by the model: {got['cast_by_the_model']}  "
              f"longest verbatim {scenes[-1]['longest_verbatim']['characters']}")
    out["speaking_as_a_character"] = scenes

    # ------------------------------------------------------------------ 5. the strongest verbatim test
    print("\nthe line it ought to know best, written greedily from its real run-up")
    greedy = generate(model, tokenizer, FAMOUS, characters=120, temperature=0.0)
    truth = corpus.text[corpus.text.find(FAMOUS) + len(FAMOUS):][:60]
    out["the_famous_line"] = {"run_up": FAMOUS, "what_really_follows": truth,
                              "what_the_model_wrote": greedy["continuation"],
                              "it_reproduced_the_line": truth.split("\n")[0] in greedy["continuation"],
                              "longest_verbatim": longest_copy(corpus, greedy["continuation"])}
    print(f"  really follows: {truth.splitlines()[0]!r}")
    print(f"  model wrote   : {greedy['continuation'].splitlines()[0]!r}")
    print(f"  reproduced it : {out['the_famous_line']['it_reproduced_the_line']}")

    # ------------------------------------------------------------------ 6. what it invents
    print("\nwhat it invents")
    everything = " ".join(r["continuation"] for r in wrote) + " ".join(r["continuation"] for r in dial)
    words = sorted({w.strip(".,;:!?’“”()[]_—-").lower() for w in everything.split()
                    if len(w.strip(".,;:!?’“”()[]_—-")) > 5})
    lowered = corpus.text.lower()
    invented = [w for w in words if w and w not in lowered][:40]
    out["words_it_invented"] = {"checked": len(words), "absent_from_the_corpus": invented,
                                "note": "lowercased comparison against the whole of the 39 training works"}
    print(f"  {len(invented)} of {len(words)} long words appear nowhere in Shakespeare: {invented[:12]}")

    # ------------------------------------------------------------------ 7. the normaliser
    print("\nthe prompt normaliser")
    normalised = []
    for probe in NORMALISER:
        try:
            text, notes = normalise(probe)
            normalised.append({"typed": probe, "refused": False, "fed": text, "notes": notes})
            print(f"  {probe!r:24} -> {text!r} {notes}")
        except Exception as refusal:
            normalised.append({"typed": probe, "refused": True, "because": str(refusal)})
            print(f"  {probe!r:24} -> REFUSED: {str(refusal).splitlines()[0][:70]}")
    out["the_normaliser"] = normalised

    # ------------------------------------------------------------------ 8. the ten fixed prompts
    out["the_ten_prompts"] = [{"name": name, "prompt": prompt} for name, prompt in SUITE]

    (ROOT / "docs" / "demonstrations.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readable(out)
    print("\nwrote docs/demonstrations.json and docs/demonstrations.txt")


def _write_readable(out: dict) -> None:
    lines = ["Every example in blog part 8, generated by scripts/demonstrate.py.",
             f"Model: {out['about']['run']}/best.pt at step {out['about']['step']:,}.", ""]
    lines.append("=" * 100 + "\nWHAT IT PREDICTS\n" + "=" * 100)
    for row in out["what_it_predicts"]:
        lines.append(f"\nafter {row['context']!r}")
        lines.append("   " + "  ".join(f"{c['character']!r}={c['probability']:.1%}" for c in row["top"]))
    lines.append("\n" + "=" * 100 + "\nWHAT IT WRITES\n" + "=" * 100)
    for row in out["what_it_writes"]:
        lines += [f"\n--- {row['label']} (temperature {row['settings']['temperature']}, seed {row['settings']['seed']})",
                  f"    prompt: {row['prompt']!r}",
                  f"    stopped: {row['why_it_stopped']}",
                  f"    longest verbatim: {row['longest_verbatim']['characters']} of "
                  f"{row['longest_verbatim']['of']} chars = {row['longest_verbatim']['passage']!r}", "",
                  row["prompt"] + row["continuation"]]
    lines.append("\n" + "=" * 100 + "\nTHE TEMPERATURE DIAL, one prompt\n" + "=" * 100)
    for row in out["the_temperature_dial"]:
        lines += [f"\n--- temperature {row['temperature']} (longest verbatim "
                  f"{row['longest_verbatim']['characters']})", "\n\nJULIET.\n" + row["continuation"]]
    lines.append("\n" + "=" * 100 + "\nSPEAKING AS A CHARACTER\n" + "=" * 100)
    for row in out["speaking_as_a_character"]:
        lines += [f"\n--- answerer asked for: {row['answerer_asked_for']}, answered by {row['answerer']} "
                  f"(cast by the model: {row['cast_by_the_model']})", row["transcript"]]
    f = out["the_famous_line"]
    lines += ["\n" + "=" * 100 + "\nTHE LINE IT OUGHT TO KNOW BEST\n" + "=" * 100,
              f"\nrun-up given: {f['run_up']!r}",
              f"what really follows: {f['what_really_follows']!r}",
              f"what the model wrote: {f['what_the_model_wrote']!r}",
              f"did it reproduce the line: {f['it_reproduced_the_line']}"]
    lines += ["\n" + "=" * 100 + "\nWORDS IT INVENTED\n" + "=" * 100,
              f"\n{len(out['words_it_invented']['absent_from_the_corpus'])} of "
              f"{out['words_it_invented']['checked']} long words appear nowhere in the 39 training works:",
              ", ".join(out["words_it_invented"]["absent_from_the_corpus"])]
    lines += ["\n" + "=" * 100 + "\nTHE PROMPT NORMALISER\n" + "=" * 100]
    for row in out["the_normaliser"]:
        lines.append(f"\n{row['typed']!r} -> " + ("REFUSED: " + row["because"] if row["refused"]
                                                  else f"{row['fed']!r}  {row['notes']}"))
    (ROOT / "docs" / "demonstrations.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
