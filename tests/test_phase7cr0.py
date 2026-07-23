import json
from pathlib import Path

from battery_fast_charge.phase7cr0_config import load_phase7cr0_config
from battery_fast_charge.phase7cr0_runner import verify_frozen_artifacts


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase7cr0_diagnostics.yaml"


def test_phase7cr0_preserves_phase7c_failure_evidence() -> None:
    config = load_phase7cr0_config(CONFIG)
    verification = verify_frozen_artifacts(config, ROOT)
    assert len(verification) == 5
    assert all(item["matched"] for item in verification.values())
    assert config.thermal.ambient_temperature_c == 30.0
    assert config.thermal.maximum_average_temperature_c == 35.0


def test_phase7cr0_never_authorizes_ann_execution() -> None:
    config = load_phase7cr0_config(CONFIG)
    path = ROOT / config.result_directory / "metrics.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["decision"]["phase7c_original_failure_unchanged"]
    assert not payload["decision"]["run_ann"]


def test_phase7cr0_finds_30c_thermal_feasibility_before_r1() -> None:
    config = load_phase7cr0_config(CONFIG)
    path = ROOT / config.result_directory / "metrics.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    thermal = payload["thermal_feasibility"]
    assert thermal["feasible_strategy_exists"]
    fastest = thermal["fastest_dfn_feasible_policy"]
    assert fastest["charge_time_s"] == 1895.0
    assert fastest["maximum_temperature_c"] <= 35.0
    assert thermal["one_step_empty_interval_count"] > 0
    assert thermal["conservative_300s_braking_boundary"] is not None
    assert payload["decision"]["proceed_to_phase7cr1_development"]


def test_phase7cr0_solver_and_reversal_attribution() -> None:
    config = load_phase7cr0_config(CONFIG)
    path = ROOT / config.result_directory / "metrics.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    solver = payload["solver_failure_audit"]
    assert solver["identical_repeat_failure_count"] == 20
    assert solver["multistart_success_count"] >= 14
    assert solver["all_returned_plans_strictly_feasible"]
    assert solver["finite_retry_rule_is_candidate"]

    reversal = payload["direction_reversal_audit"]
    assert reversal["original_middle_action_reproduced"]
    assert reversal["branch_sensitive"]
    assert (
        reversal["middle_step_best_branch"][
            "objective_improvement_fraction"
        ]
        > 0.3
    )
