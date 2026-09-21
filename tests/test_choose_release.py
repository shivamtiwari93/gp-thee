"""The rule that picks the released model decides what the world sees, so it gets a test of its own."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "choose_release.py"
spec = importlib.util.spec_from_file_location("choose_release", SCRIPT)
choose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(choose)


def config(**changed) -> dict:
    from dataclasses import asdict
    from gp_thee.train import RunConfig
    return {**asdict(RunConfig(name="r")), "steps": 9985, "parameters": 10_757_760,
            "git_commit": choose.COMMIT, "seed": 1, **changed}


@pytest.mark.parametrize("changed, refused", [
    ({}, None),
    ({"git_commit": "deadbeef" * 5}, "not the tokenizer commit"),
    ({"git_commit": choose.COMMIT + " (with uncommitted changes to the code or data)"}, "not the tokenizer commit"),
    ({"seed": 0}, "seed 0"),
    ({"seed": 4}, "seed 4"),
    ({"sixteen_bit": True}, "16-bit"),
    ({"dropout": 0.1}, "not the default settings"),
    ({"evaluations": 20}, "not the default settings"),
])
def test_only_a_run_from_the_tokenizer_commit_with_the_pre_registered_settings_can_be_released(changed, refused):
    answer = choose.why_not(config(**changed))
    assert (answer is None) if refused is None else (refused in answer)


def test_the_number_of_passes_may_differ_because_the_rule_never_fixed_it_in_steps():
    assert choose.why_not(config(passes=34.0)) is None and choose.why_not(config(passes=34.0015567275467)) is None


def test_the_released_run_is_the_one_the_rule_names_and_the_record_says_what_it_cost():
    release = json.loads((Path(SCRIPT).parent.parent / "docs" / "release.json").read_text())
    assert release["model"] == "GP-Thee-11M" and release["checkpoint"] == "best.pt"
    scores = [r["validation_bpc"] for r in release["eligible"]]
    assert scores == sorted(scores) and release["run"] == release["eligible"][0]["run"]    # the lowest score among the eligible
    assert release["validation_bpc"] == min(scores) and len(release["eligible"]) == 3
    assert all(row["excluded_because"] for row in release["excluded"])            # every exclusion is explained
    assert len(release["checkpoint_sha256"]) == 64 and release["parameters"] == 10_757_760
    cost = release["what_the_choice_costs"]
    assert cost["the_released_run_is_below_their_mean_by"] > 0 and "not flattered" in cost["note"]
    assert "tie" in cost["the_three_eligible_runs"]["the_rule_calls_them"]
