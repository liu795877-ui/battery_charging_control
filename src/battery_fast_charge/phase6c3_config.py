"""Configuration for Phase 6C-3 controller-output comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .phase6_config import NominalValidationConfig, PhaseSixSuccessCriteria


@dataclass(frozen=True)
class PhaseSixC3DataConfig:
    frozen_phase6b_dataset: str
    frozen_phase6b_dataset_sha256: str
    phase6c2_dataset: str
    phase6c2_dataset_sha256: str


@dataclass(frozen=True)
class PhaseSixC3NetworkConfig:
    hidden_layer_sizes: tuple[int, ...]
    activation: str
    regularization_alpha: float
    initialization_seeds: tuple[int, ...]
    maximum_iterations: int
    convergence_tolerance: float
    learning_rate_init: float
    no_improvement_iterations: int


@dataclass(frozen=True)
class PhaseSixC3StructuredOutputConfig:
    maximum_current_change_a_per_step: float
    inverse_tanh_clip: float


@dataclass(frozen=True)
class PhaseSixC3Config:
    study_name: str
    random_seed: int
    source_phase3_config: str
    data: PhaseSixC3DataConfig
    network: PhaseSixC3NetworkConfig
    structured_output: PhaseSixC3StructuredOutputConfig
    nominal_validation: NominalValidationConfig
    success_criteria: PhaseSixSuccessCriteria


def load_phase_six_c3_config(path: str | Path) -> PhaseSixC3Config:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    network = dict(raw["network"])
    network["hidden_layer_sizes"] = tuple(int(value) for value in network["hidden_layer_sizes"])
    network["initialization_seeds"] = tuple(int(value) for value in network["initialization_seeds"])
    config = PhaseSixC3Config(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        source_phase3_config=str(raw["source_phase3_config"]),
        data=PhaseSixC3DataConfig(**raw["data"]),
        network=PhaseSixC3NetworkConfig(**network),
        structured_output=PhaseSixC3StructuredOutputConfig(**raw["structured_output"]),
        nominal_validation=NominalValidationConfig(**raw["nominal_validation"]),
        success_criteria=PhaseSixSuccessCriteria(**raw["success_criteria"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseSixC3Config) -> None:
    if config.network.hidden_layer_sizes != (16, 16):
        raise ValueError("Phase 6C-3 uses the Phase 6C-1 selected 16-16 architecture.")
    if len(set(config.network.initialization_seeds)) < 5:
        raise ValueError("Phase 6C-3 requires at least five seeds.")
    if config.network.activation != "tanh":
        raise ValueError("Only the auditable tanh NumPy export is supported.")
    if not 0.0 < config.structured_output.inverse_tanh_clip < 1.0:
        raise ValueError("The inverse-tanh clip must lie strictly between zero and one.")
    if config.structured_output.maximum_current_change_a_per_step <= 0.0:
        raise ValueError("The structured slew magnitude must be positive.")
