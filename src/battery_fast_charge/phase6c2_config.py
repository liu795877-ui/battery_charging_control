"""Configuration for Phase 6C-2 targeted teacher data generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .phase6_config import FEATURE_NAMES


@dataclass(frozen=True)
class PhaseSixC2FrozenBaselineConfig:
    dataset: str
    dataset_sha256: str
    pure_dnn_trajectory: str
    pure_dnn_trajectory_sha256: str


@dataclass(frozen=True)
class PhaseSixC2SamplingConfig:
    targeted_boundary_trajectory_count: int
    closed_loop_dagger_trajectory_count: int
    dagger_candidate_count: int
    trajectory_steps: int
    new_validation_fraction: float
    checkpoint_interval: int
    targeted_ranges: dict[str, tuple[float, float]]
    dagger_jitter_standard_deviation: dict[str, float]
    global_state_bounds: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class PhaseSixC2AcceptanceConfig:
    minimum_teacher_acceptance_fraction: float
    minimum_accepted_trajectory_count: int


@dataclass(frozen=True)
class PhaseSixC2Config:
    study_name: str
    random_seed: int
    source_phase3_config: str
    frozen_baseline: PhaseSixC2FrozenBaselineConfig
    sampling: PhaseSixC2SamplingConfig
    acceptance: PhaseSixC2AcceptanceConfig


def _feature_pairs(values: Any, name: str) -> dict[str, tuple[float, float]]:
    result = {str(key): tuple(float(value) for value in pair) for key, pair in dict(values).items()}
    if tuple(result) != FEATURE_NAMES:
        raise ValueError(f"{name} must follow the fixed five-feature order.")
    if any(len(pair) != 2 or pair[1] <= pair[0] for pair in result.values()):
        raise ValueError(f"{name} must contain increasing lower/upper pairs.")
    return result


def load_phase_six_c2_config(path: str | Path) -> PhaseSixC2Config:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    sampling = dict(raw["sampling"])
    sampling["targeted_ranges"] = _feature_pairs(sampling["targeted_ranges"], "targeted_ranges")
    sampling["global_state_bounds"] = _feature_pairs(
        sampling["global_state_bounds"], "global_state_bounds"
    )
    sampling["dagger_jitter_standard_deviation"] = {
        str(key): float(value)
        for key, value in sampling["dagger_jitter_standard_deviation"].items()
    }
    config = PhaseSixC2Config(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        source_phase3_config=str(raw["source_phase3_config"]),
        frozen_baseline=PhaseSixC2FrozenBaselineConfig(**raw["frozen_baseline"]),
        sampling=PhaseSixC2SamplingConfig(**sampling),
        acceptance=PhaseSixC2AcceptanceConfig(**raw["acceptance"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseSixC2Config) -> None:
    sampling = config.sampling
    total = sampling.targeted_boundary_trajectory_count + sampling.closed_loop_dagger_trajectory_count
    if not 500 <= total <= 1000:
        raise ValueError("Phase 6C-2 must request 500 to 1000 initial trajectories in total.")
    if sampling.dagger_candidate_count < sampling.closed_loop_dagger_trajectory_count:
        raise ValueError("DAgger candidate count must cover the requested selected states.")
    if set(sampling.dagger_jitter_standard_deviation) != set(FEATURE_NAMES):
        raise ValueError("DAgger jitter must be specified for every fixed feature.")
    if not 0.0 < sampling.new_validation_fraction < 0.5:
        raise ValueError("The new validation fraction must be between zero and one half.")
    if sampling.trajectory_steps < 2 or sampling.checkpoint_interval < 1:
        raise ValueError("Trajectory and checkpoint lengths must be positive.")
    if config.acceptance.minimum_accepted_trajectory_count > total:
        raise ValueError("Accepted trajectory gate cannot exceed the requested total.")
