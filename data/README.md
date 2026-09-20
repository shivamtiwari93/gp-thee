# Data

## Source

| | |
|---|---|
| Work | *The Complete Works of William Shakespeare*, Project Gutenberg eBook #100 |
| URL | https://www.gutenberg.org/files/100/100-0.txt |
| Mirror | https://gutenberg.pglaf.org/1/0/100/100-0.txt |
| Downloaded | 2026-09-20 |
| Size | 5,422,721 bytes; 196,022 lines; 963,478 words |
| SHA-256 | `a023115c2d4e2ee12221bdd780fdf2ac5a864fe225948656f51f8be462c7fffb` |
| Contents | 44 works: the Sonnets (154), 38 plays, 5 poems |
| Copyright | Public domain in the United States |

## Folders

- `raw/` holds the untouched download, read-only, exactly as Project Gutenberg served it. It is committed so that anyone can see the full dataset this model was trained on, and so the project still reproduces if Gutenberg revises or moves the file. See [raw/README.md](raw/README.md) for the Project Gutenberg notice that travels with it.
- `processed/` will hold the cleaned corpus and the train / validation / test split, generated from `raw/` by script. These are committed too, so every result in this repo can be traced to an exact text.

Nothing in `raw/` is ever edited by hand. If a cleaning rule is wrong, the fix goes in the script and `processed/` is regenerated.
