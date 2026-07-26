from __future__ import annotations

from pathlib import Path

from battery_fast_charge.phase7cr3t2_config import load_phase7cr3t2_config
from battery_fast_charge.phase7cr3t2_runner import verify_frozen_r3t


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_phase7cr3t2_config(
    ROOT / "configs/phase7cr3t2_dfn_repeatability.yaml"
)


def test_r3t2_scope_and_sequencing_are_preregistered() -> None:
    control = CONFIG.section("control_contract")
    repeatability = CONFIG.section("repeatability_contract")
    assert control["optimized_supervisor_frozen"] is True
    assert control["ann_retraining_authorized"] is False
    assert control["safety_contract_change_authorized"] is False
    assert control["same_state_current_limit_difference_a"] == 0.0
    assert control["level4_entry_authorized"] is False
    assert repeatability["confirmation_may_not_modify_tolerances"] is True
    assert repeatability["expected_development_pair_count"] == 16
    assert repeatability["expected_confirmation_pair_count"] == 120


def test_r3t_speed_pass_and_repeatability_failure_are_frozen() -> None:
    result = verify_frozen_r3t(CONFIG, ROOT)
    assert result["r3t_failed_checks"] == ["closed_loop_current_exact"]
    assert result["same_state_current_limit_difference_a"] == 0.0
    assert all(record["matched"] for record in result["records"].values())


def test_development_and_confirmation_designs_are_isolated() -> None:
    designs = CONFIG.section("datasets")["designs"]
    assert designs["development"][15]["seed"] == 20261201
    assert designs["development"][30]["seed"] == 20261202
    assert designs["confirmation"][15]["seed"] == 20261203
    assert designs["confirmation"][30]["seed"] == 20261204
    assert set(CONFIG.section("repeatability_contract")["confirmation_seeds"]) == {
        22,
        42,
        73,
        101,
        137,
    }
