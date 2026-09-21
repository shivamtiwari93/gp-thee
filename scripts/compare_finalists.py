"""Characters against the best word-fragment vocabulary, on the very same text. Reported beside the verdict; decides nothing.

Run:  uv run python scripts/compare_finalists.py            (after the sweep; about two minutes on the GPU)
      uv run python scripts/compare_finalists.py --arms char bpe-1024 --prefix sweep-

The rule that picks a tokenizer compares the means of three runs per arm against the wobble between runs
(scripts/summarise_runs.py). This script asks a different question of the same six models: WHERE in the text do
they differ, and would the gap survive a different choice of text? Every choice in it was written down in
docs/BUILD_LOG.md (entry 15) before the sweep ran, so that none of them could be made with the results in view:

  * The finalists are characters and the word-fragment arm with the lowest three-seed mean.
  * Each run is scored at its best checkpoint (best.pt), on the validation works, the usual way.
  * A token's bits are shared equally among its characters. A START token has none, so its bits go to the first
    character of the work it opens. That gives every model one number per CHARACTER of the same text.
  * Per arm, those numbers are averaged over its three seeds.
  * The text is cut into blocks of 5,000 characters, within each play. The statistic is the difference in bits per
    character over the whole text; the interval comes from drawing blocks again with replacement, play by play,
    10,000 times, with numpy's default_rng(0); 95%, by percentiles. The sign of the difference is given per play.

What the interval means, and does not. It holds the six trained models fixed and asks about the choice of TEXT.
It knows nothing about the luck of training, which is what the rule's own test measures. So it can say "these
models differ consistently across the plays"; it cannot say "these tokenizers differ", and it cannot overturn the rule.

Writes docs/finalists.json.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from gp_thee.data import DATA, load_tokens, load_works
from gp_thee.evaluation import piece_lengths, speaker_label_characters
from gp_thee.tokenizer import load as load_tokenizer
from gp_thee.train import evaluate, load_checkpoint

ROOT = Path(__file__).resolve().parent.parent
BLOCK, RESAMPLES = 5_000, 10_000


def bits_by_character(surprise: np.ndarray, stream: np.ndarray, tokenizer) -> np.ndarray:
    """One number per character of the text: each token's bits shared equally among its characters; START's bits on the character after it."""
    stream = np.asarray(stream).astype(np.int64)
    lengths = piece_lengths(tokenizer)[stream]
    bits = np.concatenate([[0.0], surprise]) / math.log(2)           # the first token was given, not predicted
    begins = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    real = lengths > 0
    out = np.repeat(bits[real] / lengths[real], lengths[real])
    np.add.at(out, begins[~real], bits[~real])
    return out


def blocks_of(works: list[str]) -> list[tuple[int, int, int]]:
    """(play, begin, end) for blocks of BLOCK characters, never across two plays. The last block of a play is the short one."""
    blocks, offset = [], 0
    for play, work in enumerate(works):
        blocks += [(play, offset + b, offset + min(b + BLOCK, len(work))) for b in range(0, len(work), BLOCK)]
        offset += len(work)
    return blocks


def interval(difference: np.ndarray, blocks: list[tuple[int, int, int]]) -> tuple[float, float]:
    sums = np.array([difference[b:e].sum() for _, b, e in blocks])
    sizes = np.array([e - b for _, b, e in blocks])
    plays = np.array([play for play, _, _ in blocks])
    dice, draws = np.random.default_rng(0), []
    for _ in range(RESAMPLES):
        picked = np.concatenate([dice.choice(np.flatnonzero(plays == play), size=(plays == play).sum()) for play in np.unique(plays)])
        draws.append(sums[picked].sum() / sizes[picked].sum())
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prefix", default="sweep-")
    parser.add_argument("--arms", nargs=2, help="two tokenizers; by default characters and the word-fragment arm with the lowest mean")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    runs = {}
    for folder in sorted((ROOT / "runs").glob(f"{args.prefix}*")):
        if (folder / "result.json").exists():
            config, result = json.loads((folder / "config.json").read_text()), json.loads((folder / "result.json").read_text())
            runs.setdefault(config["tokenizer"], []).append((folder, config["seed"], result["best"]["validation_bpc"]))
    means = {arm: float(np.mean([score for _, _, score in found])) for arm, found in runs.items()}
    arms = args.arms or ["char", min((arm for arm in means if arm != "char"), key=means.get)]
    works = load_works("validation")
    characters = sum(map(len, works))

    per_arm = {}
    for arm in arms:
        tokenizer, stream = load_tokenizer(DATA / "tokenizers" / f"{arm}.json"), load_tokens(arm, "validation")
        per_run = []
        for folder, seed, recorded in runs[arm]:
            model, _ = load_checkpoint(folder / "best.pt", args.device)
            bits = bits_by_character(evaluate(model, stream, args.device), stream, tokenizer)
            if len(bits) != characters or abs(bits.sum() / characters - recorded) > 1e-3:
                raise AssertionError(f"{folder.name}: the per-character bits do not add up to the run's recorded score")
            per_run.append(bits)
            print(f"  {folder.name:<28} seed {seed}   {bits.sum() / characters:.4f} bits per character")
        per_arm[arm] = np.mean(per_run, axis=0)

    first, second = arms
    difference = per_arm[second] - per_arm[first]
    blocks = blocks_of(works)
    low, high = interval(difference, blocks)
    on_label = np.concatenate([speaker_label_characters(work) for work in works])
    ends = np.cumsum([len(work) for work in works])
    by_play = {work.split("\n", 1)[0]: float(part.mean()) for work, part in zip(works, np.split(difference, ends[:-1]))}
    out = {"arms": arms, "difference_is": f"{second} minus {first}, in bits per character; above zero means {first} predicts better",
           "runs": {arm: [folder.name for folder, _, _ in runs[arm]] for arm in arms}, "difference": float(difference.mean()), "interval_95": [low, high],
           "blocks": len(blocks), "block_characters": BLOCK, "resamples": RESAMPLES, "by_play": by_play,
           "on_speaker_label_lines": float(difference[on_label].mean()), "on_everything_else": float(difference[~on_label].mean()),
           "share_of_blocks_where_the_first_arm_is_better": float(np.mean([difference[b:e].sum() > 0 for _, b, e in blocks]))}
    print(f"\n{second} minus {first}: {out['difference']:+.4f} bits per character  (95% interval over the choice of text: {low:+.4f} to {high:+.4f}; {len(blocks)} blocks)")
    for title, value in by_play.items():
        print(f"    {title:<40}{value:+.4f}")
    print(f"    {'speaker-label lines':<40}{out['on_speaker_label_lines']:+.4f}\n    {'everything else':<40}{out['on_everything_else']:+.4f}")
    print(f"    {first} is the better of the two in {out['share_of_blocks_where_the_first_arm_is_better']:.0%} of the blocks")
    print("\nThis interval holds the six trained models fixed. It measures the choice of text, not the luck of training, and it cannot overturn the rule.")
    (ROOT / "docs" / "finalists.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote docs/finalists.json")


if __name__ == "__main__":
    main()
