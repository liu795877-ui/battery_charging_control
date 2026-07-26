from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from battery_fast_charge.phase7cr1_config import load_phase7cr1_config
from battery_fast_charge.phase7cr2_runner import _thermal_current_limit
from battery_fast_charge.phase7cr3t_config import load_phase7cr3t_config
from battery_fast_charge.phase7cr3t_runner import verify_frozen_r3
from battery_fast_charge.phase7cr3t_thermal import optimized_thermal_current_limit


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_phase7cr3t_config(ROOT / "configs/phase7cr3t_supervisor_runtime.yaml")
R1 = load_phase7cr1_config(ROOT / "configs/phase7cr1_thermal_supervisor.yaml")


def test_r3t_scope_is_runtime_only_and_level4_is_not_preapproved() -> None:
    contract = CONFIG.section("optimization_contract")
    assert contract["scope"] == "shared_thermal_supervisor_runtime_only"
    assert contract["ann_retraining_authorized"] is False
    assert contract["voltage_guard_change_authorized"] is False
    assert contract["thermal_model_change_authorized"] is False
    assert contract["control_output_change_authorized"] is False
    assert contract["level4_entry_authorized"] is False


def test_r3_speed_only_failure_is_frozen() -> None:
    result = verify_frozen_r3(CONFIG, ROOT)
    assert result["r3_failed_checks"] == ["speedup_above_100"]
    assert all(record["matched"] for record in result["records"].values())


@pytest.mark.parametrize("temperature,ambient,current", [(30.0, 30.0, 10.0), (34.7, 30.0, 6.5), (15.0, 15.0, 8.0)])
@pytest.mark.parametrize("braking", [False, True])
def test_optimized_limit_matches_legacy_boundaries(
    temperature: float, ambient: float, current: float, braking: bool
) -> None:
    legacy = _thermal_current_limit(temperature, ambient, current, R1, braking)
    optimized = optimized_thermal_current_limit(temperature, ambient, current, R1, braking)
    assert optimized[0] == legacy[0]
    assert optimized[1] == pytest.approx(legacy[1], abs=1.0e-12, rel=0.0)


def test_full_trace_equivalence_is_frozen_before_confirmation() -> None:
    path = ROOT / "outputs/phase7cr3t_supervisor_runtime/equivalence_freeze.json"
    if not path.exists():
        pytest.skip("R3T equivalence audit has not completed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "optimization_frozen_before_confirmation"
    assert payload["evaluated_trace_rows"] == 60332
    assert payload["evaluated_supervisor_calls"] == 120664
    assert payload["maximum_current_limit_difference_a"] == 0.0
    assert payload["maximum_peak_temperature_difference_c"] <= 1.0e-12
    assert payload["current_mismatch_count"] == 0
    assert payload["microbenchmark_speedup"] > 1.0
    assert all(payload["checks"].values())
    assert payload["confirmation_started"] is False
    assert payload["level4_entered"] is False


def test_r3t_confirmation_strictly_stops_only_on_cross_run_exactness() -> None:
    result_dir = ROOT / "outputs/phase7cr3t_supervisor_runtime"
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    audit = json.loads((result_dir / "strict_audit.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "strict_stop_failed"
    assert metrics["failed_checks"] == ["closed_loop_current_exact"]
    assert metrics["checks"]["all_seed_speedups_above_100"] is True
    assert metrics["checks"]["closed_loop_current_exact"] is False
    assert metrics["maximum_closed_loop_current_difference_a"] == pytest.approx(
        2.475499059073627e-06
    )
    assert audit["failed_checks"] == ["closed_loop_current_exact"]
    assert audit["same_state_current_limit_difference_a"] == 0.0
    assert audit["speed_summary"]["minimum_end_to_end_speedup"] > 100.0
    assert audit["decision"]["level4_authorized"] is False
    assert audit["decision"]["level4_entered"] is False


def test_r3t_freeze_manifest_hashes_match() -> None:
    path = ROOT / "outputs/phase7cr3t_supervisor_runtime/freeze_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "strict_stop_failed"
    assert manifest["level4_authorized"] is False
    assert manifest["level4_entered"] is False
    for relative, expected in manifest["artifacts"].items():
        payload = (ROOT / relative).read_bytes()
        if Path(relative).suffix in {".py", ".yaml", ".md"}:
            payload = payload.replace(b"\r\n", b"\n")
        assert hashlib.sha256(payload).hexdigest() == expected
