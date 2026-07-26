from __future__ import annotations

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
