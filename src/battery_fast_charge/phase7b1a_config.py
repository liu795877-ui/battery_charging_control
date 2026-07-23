"""Phase 7B-1A 电压失配与制动可行性审计配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AuditConfig:
    voltage_limit_v: float
    thresholds_v: tuple[float, ...]
    residual_growth_guard_method: str
    current_search_tolerance_a: float
    feasibility_tolerance_a: float


@dataclass(frozen=True)
class ConfirmationConfig:
    trajectory_count: int
    design_start_index: int
    design_seed: int
    soc_bounds: tuple[float, float]
    v1_bounds_v: tuple[float, float]
    v2_bounds_v: tuple[float, float]
    previous_current_bounds_a: tuple[float, float]
    initial_voltage_margin_v: float


@dataclass(frozen=True)
class OutputConfig:
    data_directory: str
    result_directory: str


@dataclass(frozen=True)
class Phase7B1AConfig:
    study_name: str
    source_level3_config: str
    source_phase7b0_config: str
    source_trajectories: str
    source_regression_initial_states: str
    frozen_artifacts: dict[str, str]
    audit: AuditConfig
    confirmation: ConfirmationConfig
    output: OutputConfig


def _pair(value: object) -> tuple[float, float]:
    values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    if len(values) != 2:
        raise ValueError("边界必须恰好包含两个值。")
    return values


def load_phase7b1a_config(path: str | Path) -> Phase7B1AConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    sources = raw["sources"]
    audit = dict(raw["audit"])
    audit["thresholds_v"] = tuple(float(v) for v in audit["thresholds_v"])
    confirmation = dict(raw["confirmation"])
    for key in (
        "soc_bounds",
        "v1_bounds_v",
        "v2_bounds_v",
        "previous_current_bounds_a",
    ):
        confirmation[key] = _pair(confirmation[key])
    config = Phase7B1AConfig(
        study_name=str(raw["study"]["name"]),
        source_level3_config=str(sources["level3_config"]),
        source_phase7b0_config=str(sources["phase7b0_config"]),
        source_trajectories=str(sources["phase7b0_trajectories"]),
        source_regression_initial_states=str(
            sources["regression_initial_states"]
        ),
        frozen_artifacts={
            str(key): str(value)
            for key, value in sources["frozen_artifacts"].items()
        },
        audit=AuditConfig(**audit),
        confirmation=ConfirmationConfig(**confirmation),
        output=OutputConfig(**raw["output"]),
    )
    if config.audit.residual_growth_guard_method != "maximum_positive_growth":
        raise ValueError("7B-1A 预注册裕量必须使用最大正向一步残差增长。")
    if config.confirmation.trajectory_count not in range(24, 37):
        raise ValueError("独立确认集必须包含 24–36 个初态。")
    return config
