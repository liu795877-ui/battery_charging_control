import numpy as np
from pathlib import Path

from battery_fast_charge.mpc import ReducedBatteryModel, ReducedState
from battery_fast_charge.phase6b_runner import _load_context
from battery_fast_charge.phase6r_config import (
    ROLLING_STATE_FEATURES,
    load_phase_six_r_config,
)
from battery_fast_charge.phase6r_teacher import (
    assign_trajectory_splits,
    compare_independent_rolling_teachers,
    design_initial_states,
    state_features,
)


def test_phase6r_config_declares_new_split_and_three_seeds() -> None:
    config = load_phase_six_r_config("configs/phase6r_corrected_policy_distillation.yaml")

    assert config.teacher_data.initial_trajectory_count == 240
    assert len(config.network.initialization_seeds) >= 3
    assert config.validation.maximum_offline_nrmse == 0.01


def test_phase6r_design_contains_full_thermal_state() -> None:
    config = load_phase_six_r_config("configs/phase6r_corrected_policy_distillation.yaml")
    design = design_initial_states(config, 25.0)

    assert len(design) == config.teacher_data.initial_trajectory_count
    assert set(ROLLING_STATE_FEATURES).issubset(design.columns)
    assert design["state_ambient_temperature_c"].eq(25.0).all()


def test_phase6r_split_is_deterministic_and_trajectory_level() -> None:
    config = load_phase_six_r_config("configs/phase6r_corrected_policy_distillation.yaml")
    ids = [f"trajectory-{index}" for index in range(20)]

    assert assign_trajectory_splits(ids, config) == assign_trajectory_splits(ids, config)
    assert set(assign_trajectory_splits(ids, config).values()) == {"train", "validation", "test"}


def test_phase6r_state_features_do_not_collapse_core_surface_temperature() -> None:
    state = ReducedState(
        soc=0.2,
        polarization_fast_v=0.01,
        polarization_slow_v=0.02,
        core_temperature_c=31.0,
        surface_temperature_c=29.0,
        previous_current_a=4.0,
    )
    features = state_features(state, 25.0)

    assert tuple(features) == ROLLING_STATE_FEATURES
    assert not np.isclose(features["state_core_temperature_c"], features["state_surface_temperature_c"])


def test_phase6r_independent_teachers_execute_identical_first_actions() -> None:
    config = load_phase_six_r_config("configs/phase6r_corrected_policy_distillation.yaml")
    phase3, parameters, ocv_function = _load_context(config, Path("."))
    model = ReducedBatteryModel(phase3, ocv_function, parameters)
    state = ReducedState(
        soc=0.2,
        polarization_fast_v=0.01,
        polarization_slow_v=0.01,
        core_temperature_c=26.0,
        surface_temperature_c=25.5,
        previous_current_a=2.0,
    )

    comparison = compare_independent_rolling_teachers(state, model, phase3, steps=2)

    assert comparison["absolute_difference_a"].max() <= config.teacher_data.consistency_tolerance_a
