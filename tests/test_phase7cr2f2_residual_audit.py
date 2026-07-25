import hashlib
import json
from pathlib import Path

from battery_fast_charge.phase7cr2f2_residual_audit import (
    load_config,
    verify_frozen_r2f,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase7cr2f2_residual_initialization_audit.yaml"
METRICS = (
    ROOT
    / "outputs/phase7cr2f2_residual_initialization_audit/metrics.json"
)


def test_residual_audit_preserves_r2f_failure_evidence() -> None:
    config = load_config(CONFIG)
    verification = verify_frozen_r2f(config, ROOT)
    assert verification["r2f_failure_preserved"]
    assert len(verification["records"]) == 14
    assert all(
        record["matched"]
        for record in verification["records"].values()
    )


def test_residual_audit_does_not_authorize_f2_r3_or_ann() -> None:
    config = load_config(CONFIG)
    contracts = config["audit"]["contracts"]
    assert not contracts["f2_data_generation_authorized"]
    assert not contracts["r3_generation_authorized"]
    assert not contracts["ann_execution_authorized"]


def test_initial_measurement_is_available_but_histories_are_inconsistent() -> None:
    if not METRICS.exists():
        return
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    assert payload["state_count"] == 48
    assert payload["initial_measurement"]["available_for_all_states"]
    history = payload["state_history_contract"]
    assert not history["shared_current_history_available"]
    assert not history["dfn_and_2rc_share_consistent_current_history"]
    assert not history["zero_initialization_documented_in_config_or_plan"]


def test_measured_initialization_exposes_boot_transient() -> None:
    if not METRICS.exists():
        return
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    measured = payload["mode_summary"]["measured"]
    zero = payload["mode_summary"]["zero"]
    assert measured["maximum_boot_positive_growth_v"] > 0.034
    assert measured["maximum_boot_positive_growth_v"] > zero[
        "maximum_boot_positive_growth_v"
    ]
    assert measured["maximum_running_positive_growth_v"] < 0.012
    assert measured["empty_interval_count"] == 0


def test_audit_freezes_measured_initialization_and_two_stage_structure() -> None:
    if not METRICS.exists():
        return
    decision = json.loads(METRICS.read_text(encoding="utf-8"))["decision"]
    assert decision["freeze_residual_initialization"] == "measured"
    assert decision["freeze_two_stage_guard_structure"]
    assert decision["boot_steps"] == [0, 1]
    assert decision["running_starts_at_step"] == 2
    assert not decision["generate_f2_data"]
    assert not decision["generate_r3"]
    assert not decision["run_ann"]


def test_residual_audit_freeze_manifest_matches() -> None:
    path = (
        ROOT
        / "outputs/phase7cr2f2_residual_initialization_audit"
        / "freeze_manifest.json"
    )
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "diagnostic_complete"
    assert manifest["frozen_decision"]["residual_initialization"] == "measured"
    assert not manifest["f2_data_generated"]
    for relative, expected in manifest["artifacts"].items():
        payload = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(payload).hexdigest() == expected
