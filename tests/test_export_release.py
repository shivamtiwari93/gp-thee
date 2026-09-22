"""The Hugging Face release must be the released checkpoint, exactly — not a near-copy.

The weights leave the project as safetensors, with the tied output projection dropped and re-tied on load. That
round trip is the one place the published artifact could silently drift from the run every number was measured on,
so it is pinned here: same weights, same tie, bit-identical forward pass.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "release" / "gp-thee-11m"

pytestmark = pytest.mark.skipif(not (RELEASE / "model.safetensors").exists(),
                                reason="run scripts/export_release.py first")


def _loaded():
    # dont_write_bytecode: importing the loader would otherwise leave a __pycache__ inside the folder that
    # `hf upload` publishes verbatim. One reached the Hub exactly that way.
    was, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        sys.path.insert(0, str(RELEASE))
        from load_release import load
        return load("cpu")
    finally:
        sys.dont_write_bytecode = was


def test_the_export_reloads_bit_identical_to_the_checkpoint():
    from gp_thee.train import load_checkpoint
    original, _ = load_checkpoint(ROOT / "runs" / "sweep-char-seed-1" / "best.pt", "cpu")
    original.eval()
    exported, _ = _loaded()

    so, se = original.state_dict(), exported.state_dict()
    assert set(so) == set(se)
    assert all(torch.equal(so[k], se[k]) for k in so), "a weight changed in the round trip"

    torch.manual_seed(0)
    x = torch.randint(0, 98, (2, 64))
    with torch.no_grad():
        assert torch.equal(original(x)[0], exported(x)[0]), "the exported model scores differently"


def test_the_output_projection_is_dropped_from_the_file_and_re_tied_on_load():
    from safetensors.torch import load_file
    stored = load_file(RELEASE / "model.safetensors")
    assert "to_scores.weight" not in stored, "the tied copy must not be stored twice"
    assert "token_embedding.weight" in stored
    model, _ = _loaded()
    assert model.token_embedding.weight is model.to_scores.weight, "the tie was not restored"


def test_the_card_and_config_quote_the_committed_scores():
    import json
    config = json.loads((RELEASE / "config.json").read_text())
    release = json.loads((ROOT / "docs" / "release.json").read_text())
    final = json.loads((ROOT / "docs" / "final-evaluation.json").read_text())
    assert config["scores_bits_per_character"]["validation"] == release["validation_bpc"]
    assert config["scores_bits_per_character"]["test"] == final["headline"]["the_artifact"]["test_bits_per_character"]
    assert config["provenance"]["source_checkpoint_sha256"] == release["checkpoint_sha256"]
    card = (RELEASE / "README.md").read_text()
    assert f"{release['validation_bpc']:.4f}" in card and "safetensors" in card


def test_the_loader_survives_hugging_faces_symlink_layout(tmp_path):
    """huggingface_hub lays a download out as symlinks into a shared blobs/ store.

    The first published loader used Path(__file__).resolve(), which follows that symlink into blobs/ and then
    looks for config.json beside hash-named objects, where it is not. Anyone using snapshot_download -- the
    normal way -- got FileNotFoundError. Pinned here, because the repository cannot see the Hub's layout.
    """
    import shutil
    import sys
    blobs, snapshot = tmp_path / "blobs", tmp_path / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for i, source in enumerate(sorted(f for f in RELEASE.iterdir() if f.is_file())):
        blob = blobs / f"{i:040x}"
        shutil.copy(source, blob)
        (snapshot / source.name).symlink_to(blob)
    assert all(p.is_symlink() for p in snapshot.iterdir())

    was, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    sys.path.insert(0, str(snapshot))
    sys.modules.pop("load_release", None)
    try:
        import load_release
        model, _ = load_release.load("cpu")
    finally:
        sys.path.remove(str(snapshot))
        sys.modules.pop("load_release", None)
        sys.dont_write_bytecode = was
    assert sum(p.numel() for p in model.parameters()) == 10_757_760
    assert model.token_embedding.weight is model.to_scores.weight


def test_the_published_folder_holds_exactly_the_five_files():
    # A stray __pycache__ from importing the loader reached the Hub once; hf upload publishes whatever is there.
    assert {f.name for f in RELEASE.iterdir()} == {
        "model.safetensors", "config.json", "tokenizer.json", "load_release.py", "README.md"}
