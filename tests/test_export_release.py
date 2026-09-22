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
    sys.path.insert(0, str(RELEASE))
    from load_release import load
    return load("cpu")


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
