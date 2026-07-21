"""Configuration for Phase 6B DNN failure diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .phase6_config import (
    FEATURE_NAMES,
    NominalValidationConfig,
    PaperMethodConfig,
    PaperNetworkConfig,
    PhaseSixSuccessCriteria,
)


@dataclass(frozen=True)
class PhaseSixBDiagnosticsConfig:
    """Offline error partitions used to locate where imitation fails."""

    soc_bins: tuple[float, ...]
    temperature_bins_c: tuple[float, ...]
    previous_current_bins_a: tuple[float, ...]
    slew_margin_close_a: float


@dataclass(frozen=True)
class PhaseSixBConfig:
    """Independent Phase 6B experiment settings."""

    study_name: str
    random_seed: int
    source_phase3_config: str
    paper_method: PaperMethodConfig
    network: PaperNetworkConfig
    nominal_validation: NominalValidationConfig
    diagnostics: PhaseSixBDiagnosticsConfig
    success_criteria: PhaseSixSuccessCriteria


def _pair(values: Any) -> tuple[float, float]:
    pair = tuple(float(value) for value in values)
    if len(pair) != 2:
        raise ValueError("State ranges must contain exactly two values.")
    return pair


def _float_tuple(values: Any) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) < 2 or any(result[i + 1] <= result[i] for i in range(len(result) - 1)):
        raise ValueError("Diagnostic bins must be strictly increasing.")
    return result


def load_phase_six_b_config(path: str | Path) -> PhaseSixBConfig:
    """Read and validate the independent Phase 6B YAML file."""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if "diagnostics" not in raw:
        raise ValueError("Phase 6B configuration requires a diagnostics section.")

    paper = dict(raw["paper_method"])
    paper["factorial_levels"] = tuple(int(v) for v in paper["factorial_levels"])
    paper["state_ranges"] = {
        str(name): _pair(values) for name, values in paper["state_ranges"].items()
    }

    network = dict(raw["network"])
    network["candidate_hidden_layer_sizes"] = tuple(
        tuple(int(width) for width in architecture)
        for architecture in network["candidate_hidden_layer_sizes"]
    )
    network["regularization_candidates"] = tuple(
        float(value) for value in network["regularization_candidates"]
    )
    network["initialization_seeds"] = tuple(
        int(value) for value in network["initialization_seeds"]
    )

    diagnostics = dict(raw["diagnostics"])
    diagnostics["soc_bins"] = _float_tuple(diagnostics["soc_bins"])
    diagnostics["temperature_bins_c"] = _float_tuple(diagnostics["temperature_bins_c"])
    diagnostics["previous_current_bins_a"] = _float_tuple(
        diagnostics["previous_current_bins_a"]
    )

    config = PhaseSixBConfig(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        source_phase3_config=str(raw["source_phase3_config"]),
        paper_method=PaperMethodConfig(**paper),
        network=PaperNetworkConfig(**network),
        nominal_validation=NominalValidationConfig(**raw["nominal_validation"]),
        diagnostics=PhaseSixBDiagnosticsConfig(**diagnostics),
        success_criteria=PhaseSixSuccessCriteria(**raw["success_criteria"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseSixBConfig) -> None:
    paper = config.paper_method
    if tuple(paper.state_ranges) != FEATURE_NAMES:
        raise ValueError("Phase 6B state ranges must follow the fixed feature order.")
    if paper.initial_state_count < 1000:
        raise ValueError("Phase 6B should use at least 1000 initial states.")
    if paper.trajectory_steps < 2:
        raise ValueError("At least two unfolded trajectory steps are required.")
    factorial_count = 1
    for level in paper.factorial_levels:
        if level < 1:
            raise ValueError("Factorial levels must be positive integers.")
        factorial_count *= level
    if factorial_count >= paper.initial_state_count:
        raise ValueError("The Hammersley part of the design must be non-empty.")
    if any(len(architecture) != 3 for architecture in config.network.candidate_hidden_layer_sizes):
        raise ValueError("Phase 6B keeps the paper-style three-hidden-layer DNN.")
    if not any(max(architecture) >= 32 for architecture in config.network.candidate_hidden_layer_sizes):
        raise ValueError("Phase 6B must include a larger DNN than Phase 6A.")
    if config.network.activation != "tanh" or config.network.solver != "lbfgs":
        raise ValueError("The exported NumPy DNN currently supports tanh/L-BFGS only.")
    if paper.train_fraction + paper.validation_fraction >= 1.0:
        raise ValueError("A held-out test split is required.")
