"""Phase 7B-1B/1C 电压感知安全层配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SafetyConfig:
    voltage_limit_v: float
    voltage_tolerance_v: float
    residual_growth_guard_v: float
    diagnostic_fixed_margin_v: float
    current_search_tolerance_a: float
    empty_interval_tolerance_a: float
    intervention_tolerance_a: float


@dataclass(frozen=True)
class ValidationConfig:
    oscillation_delta_threshold_a: float
    maximum_workers: int
    run_fixed_margin_on_regression_only: bool


@dataclass(frozen=True)
class OutputConfig:
    data_directory: str
    result_directory: str


@dataclass(frozen=True)
class Phase7B1BConfig:
    study_name: str
    source_phase7b0_config: str
    source_level3_config: str
    regression_initial_states: str
    regression_baseline_trajectories: str
    confirmation_initial_states: str
    phase7b1a_metrics: str
    model_directory: str
    frozen_artifacts: dict[str, str]
    safety: SafetyConfig
    validation: ValidationConfig
    output: OutputConfig


def load_phase7b1b_config(path: str | Path) -> Phase7B1BConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    source = raw["sources"]
    config = Phase7B1BConfig(
        study_name=str(raw["study"]["name"]),
        source_phase7b0_config=str(source["phase7b0_config"]),
        source_level3_config=str(source["level3_config"]),
        regression_initial_states=str(source["regression_initial_states"]),
        regression_baseline_trajectories=str(
            source["regression_baseline_trajectories"]
        ),
        confirmation_initial_states=str(
            source["confirmation_initial_states"]
        ),
        phase7b1a_metrics=str(source["phase7b1a_metrics"]),
        model_directory=str(source["model_directory"]),
        frozen_artifacts={
            str(key): str(value)
            for key, value in source["frozen_artifacts"].items()
        },
        safety=SafetyConfig(**raw["safety"]),
        validation=ValidationConfig(**raw["validation"]),
        output=OutputConfig(**raw["output"]),
    )
    if config.validation.maximum_workers < 1:
        raise ValueError("maximum_workers 必须至少为 1。")
    return config
