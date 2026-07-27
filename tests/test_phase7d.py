from __future__ import annotations

import hashlib
import json
from pathlib import Path

from battery_fast_charge.phase7d_config import load_phase7d_config
from battery_fast_charge.phase7d_runner import run_phase7d_baseline


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_phase7d_config(ROOT / "configs/phase7d_level4_performance.yaml")


def test_level4_entry_contract_protects_r3_confirmation() -> None:
    contract = CONFIG.section("level4_contract")
    assert contract["level4_entered"] is True
    assert contract["stage"] == "baseline_attribution_only"
    assert contract["ann_retraining_authorized"] is False
    assert contract["safety_contract_change_authorized"] is False
    assert contract["r3_confirmation_tuning_authorized"] is False
    assert contract["optimization_parameters_changed"] is False
    assert contract["next_stage_requires_new_development_data"] is True
    assert contract["next_stage_requires_new_internal_validation_data"] is True


def test_level4_baseline_audit_enters_level4_without_retuning() -> None:
    result = run_phase7d_baseline(CONFIG, ROOT)
    assert result["status"] == "baseline_frozen"
    assert result["success"] is True
    assert all(result["checks"].values())
    assert result["decision"]["level4_entered"] is True
    assert result["decision"]["level4_1_authorized"] is True
    assert result["decision"]["ann_retrained"] is False
    assert result["decision"]["safety_contract_changed"] is False
    assert result["decision"]["optimization_parameters_changed"] is False
    assert result["decision"]["r3_confirmation_used_for_tuning"] is False
    assert result["baseline_by_temperature"]["15"]["trajectory_count"] == 60
    assert result["baseline_by_temperature"]["30"]["trajectory_count"] == 60


def test_level4_output_records_frozen_baseline() -> None:
    output = ROOT / CONFIG.section("output")["directory"]
    metrics_path = output / "metrics.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["status"] == "baseline_frozen"
    assert metrics["canonical_source"]["confirmation_used_for_tuning"] is False
    assert metrics["baseline_by_temperature"]["15"][
        "mean_voltage_intervention_fraction"
    ] > 0.0
    assert metrics["baseline_by_temperature"]["30"][
        "mean_thermal_intervention_fraction"
    ] > 0.0
    manifest = json.loads(
        (output / "freeze_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["level4_entered"] is True
    assert manifest["r3_confirmation_used_for_tuning"] is False
    for relative, expected in manifest["artifacts"].items():
        path = ROOT / relative
        payload = path.read_bytes()
        if path.suffix in {".py", ".yaml", ".md"}:
            payload = payload.replace(b"\r\n", b"\n")
        assert hashlib.sha256(payload).hexdigest() == expected
