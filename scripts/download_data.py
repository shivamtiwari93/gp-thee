"""Verify, or re-fetch, the complete works of Shakespeare (Project Gutenberg eBook #100).

Run:  uv run python scripts/download_data.py

The corpus is committed to this repo at data/raw/100-0.txt, so normally this
just confirms the SHA-256 of that copy. If the file is missing it is fetched a
single time, verified against the known SHA-256, and stored read-only.
Project Gutenberg blocks IPs it thinks are bots, so do not call this in a loop.
Gutenberg also revises its files from time to time; if the checksum of a fresh
download no longer matches, use the committed copy, which is what every result
in this repo was produced from.

Standard library only, so it works before any dependency is installed.
"""

import hashlib
import stat
import sys
import urllib.request
from pathlib import Path

# The master copy, then Project Gutenberg's own mirror as a fallback.
URLS = [
    "https://www.gutenberg.org/files/100/100-0.txt",
    "https://gutenberg.pglaf.org/1/0/100/100-0.txt",
]
EXPECTED_SHA256 = "a023115c2d4e2ee12221bdd780fdf2ac5a864fe225948656f51f8be462c7fffb"
EXPECTED_BYTES = 5_422_721

DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "100-0.txt"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "gp-thee/0.1 (educational project)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> int:
    if DEST.exists():
        digest = sha256(DEST.read_bytes())
        if digest == EXPECTED_SHA256:
            print(f"Already downloaded and verified: {DEST}")
            return 0
        print(f"Existing file has unexpected SHA-256 {digest}; refusing to overwrite {DEST}.")
        return 1

    DEST.parent.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        print(f"Fetching {url}")
        try:
            data = fetch(url)
        except Exception as error:  # network errors, HTTP 403 if blocked, etc.
            print(f"  failed: {error}")
            continue

        digest = sha256(data)
        print(f"  {len(data):,} bytes, sha256 {digest}")
        if digest != EXPECTED_SHA256:
            # Either Gutenberg revised the text or we were served a block page.
            print(f"  checksum mismatch (expected {EXPECTED_BYTES:,} bytes, {EXPECTED_SHA256}); not saving.")
            continue

        DEST.write_bytes(data)
        DEST.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # read-only: nothing should edit the raw corpus
        print(f"Saved (read-only): {DEST}")
        return 0

    print("Could not obtain a verified copy from any source.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
