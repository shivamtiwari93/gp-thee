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

PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
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
