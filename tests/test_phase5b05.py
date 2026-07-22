from types import SimpleNamespace

import numpy as np
import pandas as pd

from battery_fast_charge.mpc import ReducedState
from battery_fast_charge.phase5b05_config import load_phase_five_b_zero_five_config
from battery_fast_charge.phase5b05_mpc import (
    RecoverableConstrainedMPC,
    project_current_to_slew_interval,
    slew_safe_interval,
)
from battery_fast_charge.phase5b05_runner import select_representative_scenarios


class OneStepSafetyModel:
    def __init__(self, safe_current_a: float) -> None:
        self.safe_current_a = safe_current_a

    def step(self, state: ReducedState, current_a: float):
        output = SimpleNamespace(
            terminal_voltage_v=4.0 if current_a <= self.safe_current_a else 4.3,
            average_temperature_c=30.0 if current_a <= self.safe_current_a else 36.0,
        )
        return state, output


def _emergency_controller(safe_current_a: float) -> RecoverableConstrainedMPC:
    controller = object.__new__(RecoverableConstrainedMPC)
    controller.model = OneStepSafetyModel(safe_current_a)
    controller.scan_points = 101
    controller.config = SimpleNamespace(
        constraints=SimpleNamespace(
            maximum_current_a=10.0,
            maximum_current_change_a_per_step=2.0,
            physical_maximum_voltage_v=4.2,
            physical_maximum_temperature_c=35.0,
        ),
        validation=SimpleNamespace(physical_constraint_tolerance=1.0e-3),
    )
    return controller


def test_slew_projection_is_intersection_of_absolute_and_change_limits() -> None:
    assert slew_safe_interval(9.0) == (7.0, 10.0)
    assert project_current_to_slew_interval(3.0, 9.0) == 7.0
    assert project_current_to_slew_interval(12.0, 9.0) == 10.0
    assert project_current_to_slew_interval(-1.0, 0.5) == 0.0


def test_emergency_fallback_stays_slew_safe_when_hard_safe_current_exists() -> None:
    controller = _emergency_controller(safe_current_a=7.0)
    state = ReducedState(0.5, 0.0, 0.0, 30.0, 30.0, 8.0)
    current, conflict, source = controller._emergency_current(state)
    assert np.isclose(current, 7.0)
    assert not conflict
    assert source == "slope_safe_emergency"
    assert abs(current - state.previous_current_a) <= 2.0


def test_hard_safety_slew_conflict_is_not_ordinary_failure() -> None:
    controller = _emergency_controller(safe_current_a=5.0)
    state = ReducedState(0.5, 0.0, 0.0, 30.0, 30.0, 8.0)
    current, conflict, source = controller._emergency_current(state)
    assert 4.9 < current <= 5.0
    assert conflict
    assert source == "hard_safety_emergency"
    assert abs(current - state.previous_current_a) > 2.0


def test_representative_selection_preserves_required_groups_and_extremes() -> None:
    config = load_phase_five_b_zero_five_config("configs/phase5b05_mpc_recovery.yaml")
    records = []
    for index in range(5):
        records.append({"scenario_id": f"feasible_{index}", "nominal_mpc_feasible": True, "oracle_mpc_feasible": True, "scenario_class": "teacher_and_ann_feasible"})
        records.append({"scenario_id": f"unresolved_{index}", "nominal_mpc_feasible": False, "oracle_mpc_feasible": False, "scenario_class": "ann_feasible_teachers_failed_unresolved"})
        records.append({"scenario_id": f"failed_{index}", "nominal_mpc_feasible": False, "oracle_mpc_feasible": False, "scenario_class": "teacher_and_ann_infeasible"})
    for scenario_id in ("nominal", "corner_hot_resistive_optimistic", "corner_cold_resistive"):
        records.append({"scenario_id": scenario_id, "nominal_mpc_feasible": False, "oracle_mpc_feasible": False, "scenario_class": "teacher_and_ann_infeasible"})
    selected = select_representative_scenarios(pd.DataFrame.from_records(records), config)
    assert {"nominal", "corner_hot_resistive_optimistic", "corner_cold_resistive"} <= set(selected["scenario_id"])
    assert selected["selection_labels"].str.contains("teacher_feasible").sum() == 5
    assert selected["selection_labels"].str.contains("unresolved").sum() == 5
