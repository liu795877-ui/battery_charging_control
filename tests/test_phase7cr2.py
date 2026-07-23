import json
from pathlib import Path

from battery_fast_charge.phase7cr2_config import load_phase7cr2_config
from battery_fast_charge.phase7cr2_runner import verify_frozen_artifacts


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase7cr2_temperature_voltage_guards.yaml"


def test_phase7cr2_verifies_r1_and_legacy_failure_evidence() -> None:
    config = load_phase7cr2_config(CONFIG)
    verification = verify_frozen_artifacts(config, ROOT)
    assert len(verification) == 9
    assert all(item["matched"] for item in verification.values())


def test_phase7cr2_dataset_contract_is_development_only() -> None:
    config = load_phase7cr2_config(CONFIG)
    assert config.datasets["temperatures_c"] == [15.0, 30.0]
    assert config.datasets["development_count_per_temperature"] == 16
    assert config.datasets["internal_validation_count_per_temperature"] == 8
    freeze = json.loads(
        (
            ROOT
            / config.output["data_directory"]
            / "initial_state_freeze.json"
        ).read_text(encoding="utf-8")
    )
    assert freeze["frozen_before_rollout"]
    assert freeze["development_only"]
    assert freeze["not_r3_confirmation"]
    assert not freeze["ann_execution_authorized"]


def test_phase7cr2_guard_contract_never_reduces_margin() -> None:
    config = load_phase7cr2_config(CONFIG)
    original = config.voltage["original_25c_guard_v"]
    assert original == 0.011305522502741638
    assert config.voltage["legacy_minimum_v"][15] > original
    assert config.voltage["legacy_minimum_v"][30] == original
    assert config.voltage["development_margin_v"] == 0.0005
    path = (
        ROOT
        / config.output["data_directory"]
        / "derived_voltage_guards.json"
    )
    if not path.exists():
        return
    guards = json.loads(path.read_text(encoding="utf-8"))["guards_v"]
    assert guards["25"] == original
    assert guards["15"] > 0.0113871030773613
    assert guards["30"] > original


def test_phase7cr2_never_authorizes_ann_or_r3_confirmation() -> None:
    config = load_phase7cr2_config(CONFIG)
    path = ROOT / config.output["result_directory"] / "metrics.json"
    if not path.exists():
        return
    decision = json.loads(path.read_text(encoding="utf-8"))["decision"]
    assert not decision["run_ann"]
    assert not decision["r3_confirmation_generated"]


def test_phase7cr2_success_requires_every_role_temperature() -> None:
    config = load_phase7cr2_config(CONFIG)
    path = ROOT / config.output["result_directory"] / "metrics.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    strict = all(
        result["success"]
        for temperatures in payload["role_results"].values()
        for result in temperatures.values()
    )
    assert payload["success"] == strict
