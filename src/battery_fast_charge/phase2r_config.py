"""Phase 2R 降阶模型与控制状态充分性审计配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PhaseTwoRSources:
    phase2_config: str
    phase3_config: str
    phase6r_config: str
    fixed_parameters: str
    fixed_ocv_curve: str
    rolling_teacher_dataset: str


@dataclass(frozen=True)
class ModelAuditConfig:
    initial_soc_points: tuple[float, ...]
    parameter_anchor_soc_points: tuple[float, ...]
    ocv_soc_points: tuple[float, ...]
    temperatures_c: tuple[float, ...]
    pulse_c_rates: tuple[float, ...]
    sample_period_s: float
    rest_before_s: float
    pulse_duration_s: float
    rest_after_s: float
    prediction_horizons_s: tuple[float, ...]
    physical_maximum_voltage_v: float
    physical_maximum_temperature_c: float
    dfn_upper_voltage_cutoff_v: float
    electrical_fit_maximum_evaluations: int
    voltage_rmse_limit_mv: float
    temperature_rmse_limit_c: float


@dataclass(frozen=True)
class StateAuditConfig:
    neighbor_count: int
    replay_current_tolerance_a: float
    replay_p95_tolerance_a: float
    large_replay_difference_a: float
    maximum_large_replay_difference_fraction: float
    significant_variance_reduction_fraction: float
    sufficient_local_standard_deviation_a: float
    sufficient_p95_neighbor_label_difference_a: float


@dataclass(frozen=True)
class PhaseTwoRConfig:
    study_name: str
    random_seed: int
    sources: PhaseTwoRSources
    model_audit: ModelAuditConfig
    state_audit: StateAuditConfig


def _floats(values: object) -> tuple[float, ...]:
    return tuple(float(value) for value in values)  # type: ignore[arg-type]


def load_phase_two_r_config(path: str | Path) -> PhaseTwoRConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    model = dict(raw["model_audit"])
    for name in (
        "initial_soc_points",
        "parameter_anchor_soc_points",
        "ocv_soc_points",
        "temperatures_c",
        "pulse_c_rates",
        "prediction_horizons_s",
    ):
        model[name] = _floats(model[name])
    config = PhaseTwoRConfig(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        sources=PhaseTwoRSources(**raw["sources"]),
        model_audit=ModelAuditConfig(**model),
        state_audit=StateAuditConfig(**raw["state_audit"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseTwoRConfig) -> None:
    model = config.model_audit
    if model.sample_period_s != 5.0:
        raise ValueError("Phase 2R 必须保持项目 MPC 的 5 s 采样周期。")
    if model.initial_soc_points != (0.60, 0.65, 0.70, 0.75, 0.80):
        raise ValueError("Phase 2R-A 必须完整覆盖 60%–80% SOC。")
    if model.temperatures_c != (15.0, 25.0, 30.0):
        raise ValueError("Phase 2R-A 必须覆盖 15/25/30 ℃。")
    if model.prediction_horizons_s != (5.0, 25.0, 300.0):
        raise ValueError("预测误差审计必须固定为 5/25/300 s。")
    if not set(model.parameter_anchor_soc_points).issubset(model.initial_soc_points):
        raise ValueError("局部参数锚点必须属于脉冲初始 SOC 集合。")
    if config.state_audit.neighbor_count < 5:
        raise ValueError("局部条件方差至少需要 5 个邻居。")
    if not 0.0 < config.state_audit.maximum_large_replay_difference_fraction < 1.0:
        raise ValueError("重放大误差比例门槛必须位于 (0,1)。")
