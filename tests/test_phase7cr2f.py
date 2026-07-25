import hashlib
import json
from pathlib import Path

from battery_fast_charge.phase7cr2f_config import load_phase7cr2f_config
from battery_fast_charge.phase7cr2f_runner import (
    verify_known_teacher_regressions,
)
from battery_fast_charge.phase7cr2f2_residual_audit import (
    load_config as load_residual_audit_config,
    verify_frozen_r2f,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase7cr2f_teacher_selection_voltage_guard.yaml"
AUDIT_CONFIG = ROOT / "configs/phase7cr2f2_residual_initialization_audit.yaml"


def test_r2f_preserves_every_r2_frozen_artifact() -> None:
    config = load_residual_audit_config(AUDIT_CONFIG)
    verification = verify_frozen_r2f(config, ROOT)
    assert verification["r2f_failure_preserved"]
    assert len(verification["records"]) == 14
    assert all(
        item["matched"] for item in verification["records"].values()
    )


def test_r2f_freezes_single_guard_design_before_internal_validation() -> None:
    config = load_phase7cr2f_config(CONFIG)
    assert config.voltage["design"] == "single_guard"
    assert config.voltage["design_frozen_before_internal_validation"]
    assert config.voltage["historical_minimum_30c_v"] == 0.0209508831158355
    assert config.voltage["engineering_margin_v"] == 0.0005


def test_r2f_new_sets_are_frozen_and_trajectory_isolated() -> None:
    config = load_phase7cr2f_config(CONFIG)
    freeze = json.loads(
        (
            ROOT
            / config.output["data_directory"]
            / "initial_state_freeze.json"
        ).read_text(encoding="utf-8")
    )
    assert freeze["frozen_before_any_rollout"]
    assert freeze["development_internal_trajectory_isolation"]
    assert freeze["files"]["development_initial_states.csv"][
        "trajectory_count"
    ] == 32
    assert freeze["files"]["internal_validation_initial_states.csv"][
        "trajectory_count"
    ] == 16
    assert not freeze["ann_execution_authorized"]
    assert freeze["not_r3_confirmation"]


def test_r2f_two_known_teacher_regressions_pass() -> None:
    config = load_phase7cr2f_config(CONFIG)
    result = verify_known_teacher_regressions(config, ROOT)
    assert result["case_count"] == 2
    assert result["all_passed"]


def test_r2f_never_generates_r3_or_runs_ann() -> None:
    config = load_phase7cr2f_config(CONFIG)
    path = ROOT / config.output["result_directory"] / "metrics.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert not payload["decision"]["r3_initial_states_generated"]
    assert not payload["decision"]["run_ann"]
    assert not payload["decision"]["internal_validation_used_for_retuning"]


def test_r2f_failure_is_not_retuned_after_internal_validation() -> None:
    config = load_phase7cr2f_config(CONFIG)
    path = ROOT / config.output["result_directory"] / "metrics.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    diagnosis = payload["failure_diagnosis"]
    assert diagnosis["teacher_selection_contract_repaired"]
    assert not diagnosis["new_internal_validation_used_for_retuning"]
    if not payload["success"]:
        assert diagnosis["new_internal_validation_downgraded_to_history"]
        assert len(diagnosis["guard_exceedance_events"]) == 1
        assert diagnosis["next_historical_minimum_30c_v"] > 0.025


def test_r2f_freeze_manifest_matches_every_artifact() -> None:
    path = (
        ROOT
        / "outputs/phase7cr2f_teacher_selection_voltage_guard"
        / "freeze_manifest.json"
    )
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "strict_stop_failed"
    assert manifest["teacher_selection_contract_repaired"]
    assert not manifest["voltage_guard_contract_passed"]
    assert not manifest["ann_execution_authorized"]
    for relative, expected in manifest["artifacts"].items():
        artifact = ROOT / relative
        payload = artifact.read_bytes()
        raw = hashlib.sha256(payload).hexdigest()
        normalized = hashlib.sha256(
            payload.replace(b"\r\n", b"\n")
        ).hexdigest()
        assert expected in {raw, normalized}
