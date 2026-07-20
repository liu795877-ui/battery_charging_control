import json
from pathlib import Path

import pandas as pd

from battery_fast_charge.identification import build_ocv_function
from battery_fast_charge.filtered_baseline import filtered_baseline_metrics
from battery_fast_charge.mpc import ReducedBatteryModel, ReducedState
from battery_fast_charge.phase3_config import load_phase_three_config
from battery_fast_charge.phase3b_config import load_phase_three_b_config
from battery_fast_charge.teacher_data import (
    assign_trajectory_splits,
    filter_feasible_current,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _setup() -> tuple[ReducedBatteryModel, object, object]:
    phase3 = load_phase_three_config(PROJECT_ROOT / "configs" / "phase3.yaml")
    phase3b = load_phase_three_b_config(PROJECT_ROOT / "configs" / "phase3b.yaml")
    parameters = json.loads(
        (PROJECT_ROOT / phase3.artifacts.identified_parameters).read_text(
            encoding="utf-8"
        )
    )
    ocv = pd.read_csv(PROJECT_ROOT / phase3.artifacts.ocv_curve)
    return (
        ReducedBatteryModel(phase3, build_ocv_function(ocv), parameters),
        phase3,
        phase3b,
    )


def test_one_step_filter_respects_tightened_constraints() -> None:
    """探索策略即使要求10 A，也必须先通过电压和平均温度过滤。"""
    model, phase3, _ = _setup()
    state = ReducedState(0.70, 0.05, 0.05, 33.45, 33.45, 4.0)
    filtered = filter_feasible_current(model, state, 10.0, phase3)

    assert 0.0 <= filtered.current_a <= phase3.constraints.maximum_current_a
    assert filtered.next_voltage_v <= phase3.constraints.mpc_maximum_voltage_v + 1.0e-4
    assert filtered.next_temperature_c <= phase3.constraints.mpc_maximum_temperature_c + 1.0e-4


def test_split_assignment_keeps_whole_trajectories_together() -> None:
    """同一轨迹只能属于一个集合，轨迹数必须严格等于8/2/2。"""
    _, _, phase3b = _setup()
    ids = [rollout.name for rollout in phase3b.dataset.rollouts]
    mapping = assign_trajectory_splits(ids, phase3b)
    counts = pd.Series(mapping).value_counts().to_dict()

    assert len(mapping) == 12
    assert counts == {"train": 8, "validation": 2, "test": 2}


def test_fair_baseline_rejects_current_slew_violation() -> None:
    """公平基线的成功判据必须包含每个控制周期2 A的电流变化约束。"""
    _, phase3, _ = _setup()
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 5.0],
            "charge_current_a": [0.0, 5.0],
            "soc": [0.10, 0.80],
            "terminal_voltage_v": [3.3, 4.0],
            "average_temperature_c": [25.0, 30.0],
            "safety_override": [False, False],
            "source": ["test", "test"],
        }
    )

    metrics = filtered_baseline_metrics(frame, phase3)

    assert metrics["current_change_limit_exceeded"] is True
    assert metrics["success"] is False
