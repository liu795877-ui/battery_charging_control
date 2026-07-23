"""Configuration for the Phase 6C-1 optimization/generalization ablation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FrozenBaselineConfig:
    phase6b_commit: str
    dataset: str
    dataset_sha256: str
    expected_sample_count: int
    expected_split_sample_counts: dict[str, int]
    expected_split_trajectory_counts: dict[str, int]
    current_nrmse_normalization_a: float


@dataclass(frozen=True)
class AblationNetworkConfig:
    hidden_layer_sizes: tuple[tuple[int, ...], ...]
    activation: str
    regularization_alpha: float
    initialization_seeds: tuple[int, ...]


@dataclass(frozen=True)
class AblationOptimizerConfig:
    methods: tuple[str, ...]
    convergence_tolerance: float
    lbfgs_maximum_iterations: int
    adam_maximum_iterations: int
    adam_pretrain_iterations: int
    finetune_lbfgs_maximum_iterations: int
    adam_learning_rate_init: float
    adam_no_improvement_iterations: int


@dataclass(frozen=True)
class AblationAnalysisConfig:
    low_training_nrmse_threshold: float
    high_test_nrmse_threshold: float
    seed_instability_test_nrmse_std_threshold: float
    seed_instability_test_nrmse_range_threshold: float


@dataclass(frozen=True)
class PhaseSixC1Config:
    study_name: str
    random_seed: int
    baseline: FrozenBaselineConfig
    network: AblationNetworkConfig
    optimizers: AblationOptimizerConfig
    analysis: AblationAnalysisConfig


def _positive_int_mapping(values: Any, name: str) -> dict[str, int]:
    result = {str(key): int(value) for key, value in dict(values).items()}
    if set(result) != {"train", "validation", "test"} or any(value < 1 for value in result.values()):
        raise ValueError(f"{name} must contain positive train/validation/test counts.")
    return result


def load_phase_six_c1_config(path: str | Path) -> PhaseSixC1Config:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    baseline = dict(raw["baseline"])
    baseline["phase6b_commit"] = str(baseline["phase6b_commit"])
    baseline["expected_split_sample_counts"] = _positive_int_mapping(
        baseline["expected_split_sample_counts"], "expected_split_sample_counts"
    )
    baseline["expected_split_trajectory_counts"] = _positive_int_mapping(
        baseline["expected_split_trajectory_counts"], "expected_split_trajectory_counts"
    )
    network = dict(raw["network"])
    network["hidden_layer_sizes"] = tuple(
        tuple(int(width) for width in architecture)
        for architecture in network["hidden_layer_sizes"]
    )
    network["initialization_seeds"] = tuple(
        int(seed) for seed in network["initialization_seeds"]
    )
    optimizers = dict(raw["optimizers"])
    optimizers["methods"] = tuple(str(method) for method in optimizers["methods"])
    config = PhaseSixC1Config(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        baseline=FrozenBaselineConfig(**baseline),
        network=AblationNetworkConfig(**network),
        optimizers=AblationOptimizerConfig(**optimizers),
        analysis=AblationAnalysisConfig(**raw["analysis"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseSixC1Config) -> None:
    if len(config.baseline.dataset_sha256) != 64:
        raise ValueError("The frozen dataset requires a SHA-256 digest.")
    if config.baseline.expected_sample_count != sum(
        config.baseline.expected_split_sample_counts.values()
    ):
        raise ValueError("Expected split sample counts do not sum to the dataset count.")
    required_architectures = {(16, 16), (32, 32, 16), (64, 64, 32)}
    if set(config.network.hidden_layer_sizes) != required_architectures:
        raise ValueError("Phase 6C-1 requires the three declared DNN architectures.")
    if len(set(config.network.initialization_seeds)) < 5:
        raise ValueError("Phase 6C-1 requires at least five distinct initialization seeds.")
    if set(config.optimizers.methods) != {"lbfgs", "adam", "adam_lbfgs"}:
        raise ValueError("Phase 6C-1 requires LBFGS, Adam, and Adam-to-LBFGS.")
    if config.network.activation != "tanh":
        raise ValueError("The exported NumPy controller currently requires tanh activation.")
    if any(width < 1 for architecture in config.network.hidden_layer_sizes for width in architecture):
        raise ValueError("All hidden-layer widths must be positive.")
