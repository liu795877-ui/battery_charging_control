from pathlib import Path

from battery_fast_charge.phase7cr3_config import load_phase7cr3_config
import json
import hashlib

import pandas as pd

from battery_fast_charge.phase7cr3_runner import _model_hashes, verify_r2f5


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_phase7cr3_config(ROOT / "configs/phase7cr3_independent_confirmation.yaml")


def test_r3_preregistration_requires_safe_mpc_before_ann() -> None:
    contract = CONFIG.section("control_contract")
    assert contract["safe_mpc_must_pass_before_ann"] is True
    assert contract["ann_retraining_authorized"] is False
    assert contract["level4_entry_authorized"] is False
    assert CONFIG.section("datasets")["confirmation_count_per_temperature"] == 24


def test_r2f5_strict_pass_is_frozen() -> None:
    result = verify_r2f5(CONFIG, ROOT)
    assert result["status"] == "strict_passed"


def test_safe_mpc_pass_authorizes_but_does_not_execute_ann() -> None:
    metrics = json.loads(
        (ROOT / "outputs/phase7cr3_independent_confirmation/safe_mpc_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["success"] is True
    assert metrics["decision"] == {
        "frozen_ann_authorized": True,
        "ann_executed": False,
        "level4_entered": False,
    }


def test_all_five_frozen_ann_hashes_match() -> None:
    records = _model_hashes(ROOT)
    assert len(records) == 5
    assert all(item["expected"] == item["actual"] for item in records.values())


def test_frozen_ann_confirmation_strictly_stops_only_on_speedup() -> None:
    result_dir = ROOT / "outputs/phase7cr3_independent_confirmation"
    audit = json.loads((result_dir / "strict_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "strict_stop_failed"
    assert audit["physical_summary"]["trajectory_count"] == 240
    assert audit["failed_checks"] == ["speedup_above_100"]
    assert audit["diagnosis"]["policy_accuracy_failed"] is False
    assert audit["diagnosis"]["multitemperature_safety_failed"] is False
    assert audit["diagnosis"]["only_strict_failure_is_end_to_end_speedup"] is True
    assert audit["decision"]["level4_authorized"] is False
    assert audit["decision"]["level4_entered"] is False
    assert audit["decision"]["ann_retrained"] is False


def test_each_frozen_seed_uses_all_48_confirmation_states() -> None:
    frame = pd.read_csv(
        ROOT / "data/phase7cr3_independent_confirmation/ann_confirmation_trajectories.csv",
        usecols=["seed", "trajectory_id"],
    ).drop_duplicates()
    assert frame.groupby("seed").trajectory_id.nunique().to_dict() == {
        22: 48,
        42: 48,
        73: 48,
        101: 48,
        137: 48,
    }


def test_r3_freeze_manifest_hashes_match() -> None:
    manifest = json.loads(
        (ROOT / "outputs/phase7cr3_independent_confirmation/freeze_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "strict_stop_failed"
    assert manifest["level4_authorized"] is False
    assert manifest["level4_entered"] is False
    for relative, expected in manifest["artifacts"].items():
        payload = (ROOT / relative).read_bytes()
        if Path(relative).suffix in {".py", ".yaml", ".md"}:
            payload = payload.replace(b"\r\n", b"\n")
        assert hashlib.sha256(payload).hexdigest() == expected
