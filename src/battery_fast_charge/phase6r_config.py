"""Configuration for Phase 6R corrected rolling-policy distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROLLING_STATE_FEATURES = (
    "state_soc",
    "state_polarization_fast_v",
    "state_polarization_slow_v",
    "state_core_temperature_c",
    "state_surface_temperature_c",
    "state_ambient_temperature_c",
    "state_previous_current_a",
)

DESIGN_FEATURES = tuple(
    feature for feature in ROLLING_STATE_FEATURES if feature != "state_ambient_temperature_c"
)


@dataclass(frozen=True)
class PhaseSixRTeacherDataConfig:
    initial_trajectory_count: int
    trajectory_steps: int
    train_fraction: float
    validation_fraction: float
    checkpoint_interval_trajectories: int
    initial_state_ranges: dict[str, tuple[float, float]]
    minimum_teacher_acceptance_fraction: float
    consistency_state_count: int
    consistency_steps_per_state: int
    consistency_tolerance_a: float


@dataclass(frozen=True)
class PhaseSixRNetworkConfig:
    hidden_layer_sizes: tuple[int, ...]
    activation: str
    regularization_alpha: float
    initialization_seeds: tuple[int, ...]
    maximum_iterations: int
    convergence_tolerance: float
    learning_rate_init: float
    no_improvement_iterations: int


@dataclass(frozen=True)
class PhaseSixRValidationConfig:
    current_nrmse_normalization_a: float
    maximum_offline_nrmse: float
    maximum_reduced_closed_loop_nrmse: float
    maximum_dfn_closed_loop_nrmse: float
    maximum_charge_time_gap_fraction: float
    minimum_inference_speedup_over_mpc: float
    maximum_voltage_violation_v: float
    maximum_temperature_violation_c: float
    maximum_current_violation_a: float
    maximum_current_change_violation_a: float
    maximum_simulation_time_s: float


@dataclass(frozen=True)
class PhaseSixRConfig:
    study_name: str
    random_seed: int
    source_phase3_config: str
    teacher_data: PhaseSixRTeacherDataConfig
    network: PhaseSixRNetworkConfig
    validation: PhaseSixRValidationConfig


def _ranges(values: Any) -> dict[str, tuple[float, float]]:
    result = {
        str(key): tuple(float(value) for value in pair)
        for key, pair in dict(values).items()
    }
    if tuple(result) != DESIGN_FEATURES:
        raise ValueError("Phase 6R initial-state ranges must follow the six design features.")
    if any(len(pair) != 2 or pair[1] <= pair[0] for pair in result.values()):
        raise ValueError("Every Phase 6R initial-state range must be increasing.")
    return result


def load_phase_six_r_config(path: str | Path) -> PhaseSixRConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    teacher = dict(raw["teacher_data"])
    teacher["initial_state_ranges"] = _ranges(teacher["initial_state_ranges"])
    network = dict(raw["network"])
    network["hidden_layer_sizes"] = tuple(int(value) for value in network["hidden_layer_sizes"])
    network["initialization_seeds"] = tuple(int(value) for value in network["initialization_seeds"])
    config = PhaseSixRConfig(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        source_phase3_config=str(raw["source_phase3_config"]),
        teacher_data=PhaseSixRTeacherDataConfig(**teacher),
        network=PhaseSixRNetworkConfig(**network),
        validation=PhaseSixRValidationConfig(**raw["validation"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseSixRConfig) -> None:
    teacher = config.teacher_data
    if teacher.initial_trajectory_count < 100 or teacher.trajectory_steps < 2:
        raise ValueError("Phase 6R requires at least 100 initial trajectories and two rolling steps.")
    if not 0.0 < teacher.train_fraction < 1.0:
        raise ValueError("Phase 6R training fraction must lie in (0, 1).")
    if not 0.0 < teacher.validation_fraction < 1.0:
        raise ValueError("Phase 6R validation fraction must lie in (0, 1).")
    if teacher.train_fraction + teacher.validation_fraction >= 1.0:
        raise ValueError("Phase 6R requires a nonempty frozen test fraction.")
    if not 0.0 < teacher.minimum_teacher_acceptance_fraction <= 1.0:
        raise ValueError("Teacher acceptance fraction must lie in (0, 1].")
    if teacher.consistency_tolerance_a <= 0.0:
        raise ValueError("Teacher consistency tolerance must be positive.")
    if len(set(config.network.initialization_seeds)) < 3:
        raise ValueError("Phase 6R requires at least three independent seeds.")
    if config.network.activation != "tanh":
        raise ValueError("Phase 6R auditable NumPy export requires tanh activation.")
