import pandas as pd

from battery_fast_charge.phase3_config import load_phase_three_config
from battery_fast_charge.phase6b_config import load_phase_six_b_config
from battery_fast_charge.phase6b_runner import (
    _project_current,
    build_error_partition_table,
)


def test_phase6b_projection_respects_current_and_slew_limits() -> None:
    config = load_phase_three_config("configs/phase3.yaml")

    assert _project_current(15.0, 5.0, config) == 7.0
    assert _project_current(-1.0, 5.0, config) == 3.0
    assert _project_current(9.5, 9.5, config) == 9.5


def test_phase6b_error_partitions_include_slew_boundary() -> None:
    phase3 = load_phase_three_config("configs/phase3.yaml")
    config = load_phase_six_b_config("configs/phase6b_dnn_failure_diagnosis.yaml")
    predictions = pd.DataFrame(
        {
            "trajectory_id": ["a", "a", "b", "b"],
            "split": ["test", "test", "train", "validation"],
            "state_soc": [0.2, 0.6, 0.3, 0.7],
            "state_polarization_fast_v": [0.01, 0.02, 0.03, 0.04],
            "state_polarization_slow_v": [0.01, 0.02, 0.03, 0.04],
            "state_average_temperature_c": [26.0, 32.0, 28.0, 30.0],
            "state_previous_current_a": [4.0, 6.0, 1.0, 9.0],
            "teacher_current_a": [6.0, 4.0, 3.0, 7.0],
            "dnn_current_a": [6.5, 5.5, 3.1, 8.0],
            "active_voltage_constraint": [False, False, False, True],
            "active_temperature_constraint": [False, True, False, False],
            "active_current_upper_constraint": [False, False, False, False],
            "active_current_change_constraint": [True, True, True, True],
        }
    )

    table = build_error_partition_table(predictions, phase3, config)

    assert "slew_active" in set(table["partition_family"])
    assert "slew_near_boundary" in set(table["partition_family"])
