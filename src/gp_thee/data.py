"""Loading the works, by set, with a lock on the test works.

The rule of this project is that the three test works are not examined until the final evaluation: no
statistics, no examples, no fitting, not even "just to check". That rule was broken three times by
well-meaning one-off measurements while this project was being built (see docs/BUILD_LOG.md). Good
intentions did not hold, so the rule is now enforced here. Every script and test loads works through
this function, and asking for the test works without saying why is an error.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parent.parent.parent / "data"
PROCESSED = DATA / "processed"
UNLOCK = "this is the final evaluation"


def load_works(which: str, unlock: str = "") -> list[str]:
    """The cleaned works of one set of the frozen split: "train", "validation" or "test".

    Each file is checked against the checksum recorded when the corpus was cleaned.
    """
    if which == "test" and unlock != UNLOCK:
        raise PermissionError("The test works stay closed until the final evaluation. If this IS the final evaluation, "
                              f"pass unlock={UNLOCK!r}. If it is anything else, use the validation works.")
    split = json.loads((PROCESSED / "split.json").read_text(encoding="utf-8"))
    checksums = {w["file"]: w["sha256"] for w in json.loads((PROCESSED / "manifest.json").read_text(encoding="utf-8"))["works"]}
    works = []
    for name in split["sets"][which]["files"]:
        body = (PROCESSED / name).read_bytes()
        if hashlib.sha256(body).hexdigest() != checksums[name]:
            raise ValueError(f"{name} does not match the manifest; re-run scripts/prepare_data.py")
        works.append(body.decode("utf-8"))
    return works


def corpus_alphabet() -> list[str]:
    """Every character of all 44 works, as recorded when the corpus was cleaned, before the split existed.

    It lets us prove that any work can be encoded without opening it.
    """
    manifest = json.loads((PROCESSED / "manifest.json").read_text(encoding="utf-8"))
    return sorted(entry["char"] for entry in manifest["alphabet"])


def load_tokens(tokenizer_name: str, which: str) -> np.ndarray:
    """One set of works as a single stream of token ids, as written by scripts/build_tokenizers.py.

    Only "train" and "validation" exist on disk. The test works are never saved as tokens.
    """
    if which not in ("train", "validation"):
        raise PermissionError(f"there is no saved token stream for {which!r}, on purpose")
    stream = np.load(DATA / "tokens" / tokenizer_name / f"{which}.npy", mmap_mode="r")
    start_id = json.loads((DATA / "tokenizers" / f"{tokenizer_name}.json").read_text(encoding="utf-8"))["start_id"]
    if stream[0] != start_id:  # every stream begins with START, and START's id differs between tokenizers
        raise ValueError(f"data/tokens/{tokenizer_name}/{which}.npy was not written by the {tokenizer_name} tokenizer; re-run build_tokenizers.py")
    return stream


def random_batch(stream: np.ndarray, batch: int, context: int, rng: np.random.Generator, device: str):
    """`batch` windows of `context` tokens cut from random places in the stream, and their targets.

    The target for every position is simply the next token, so `targets` is the same window moved along by one.
    Windows may straddle two works. START sits between them, so the model can tell.
    """
    begins = rng.integers(0, len(stream) - context, size=batch)  # the last window's last target is the stream's last token
    tokens = np.stack([stream[b:b + context] for b in begins]).astype(np.int64)
    targets = np.stack([stream[b + 1:b + context + 1] for b in begins]).astype(np.int64)
    return torch.from_numpy(tokens).to(device), torch.from_numpy(targets).to(device)
