# Data

## Source

| | |
|---|---|
| Work | *The Complete Works of William Shakespeare*, Project Gutenberg eBook #100 |
| URL | https://www.gutenberg.org/files/100/100-0.txt |
| Mirror | https://gutenberg.pglaf.org/1/0/100/100-0.txt |
| Size | 5,422,721 bytes |
| SHA-256 | `a023115c2d4e2ee12221bdd780fdf2ac5a864fe225948656f51f8be462c7fffb` |
| Contents | 44 works: the Sonnets (154), 38 plays, 5 poems |
| Copyright | Public domain in the United States |

## Folders

- `raw/` holds the untouched download, read-only. It is **not committed**. Get it with `uv run python scripts/download_data.py`, which verifies the checksum above.
- `processed/` will hold the cleaned corpus and the train / validation / test split. These are committed, so every result in this repo can be traced to an exact text.

## Why the raw file is not in git

The raw file includes Project Gutenberg's marker lines. "Project Gutenberg" is a trademark, and their licence attaches conditions to redistributing files that carry the name. The cleaned corpus has those lines removed and contains only public-domain Shakespeare, so that is the version this repo shares.
