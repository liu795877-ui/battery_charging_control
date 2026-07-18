"""读取第二阶段的虚拟试验与降阶模型辨识配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PhaseTwoBatteryConfig:
    """高保真虚拟电芯设置；温度单位为摄氏度。"""

    parameter_set: str
    model: str
    thermal_model: str
    nominal_capacity_ah: float
    initial_temperature_c: float
    ambient_temperature_c: float


@dataclass(frozen=True)
class ProfileSegment:
    """一段恒定工况；充电倍率为正，静置倍率为零。"""

    mode: str
    c_rate: float
    duration_s: float


@dataclass(frozen=True)
class ExperimentConfig:
    """OCV、脉冲、热训练和独立验证试验协议。"""

    sample_period_s: float
    ocv_soc_points: tuple[float, ...]
    pulse_soc_points: tuple[float, ...]
    pulse_c_rates: tuple[float, ...]
    pulse_duration_s: float
    rest_before_s: float
    rest_after_s: float
    thermal_training_initial_soc: float
    thermal_training_profile: tuple[ProfileSegment, ...]
    validation_initial_soc: float
    validation_profile: tuple[ProfileSegment, ...]


@dataclass(frozen=True)
class IdentificationConfig:
    """参数辨识算法设置。"""

    core_heat_capacity_fraction: float
    maximum_function_evaluations: int


@dataclass(frozen=True)
class SuccessCriteria:
    """独立验证集上的第一版验收阈值。"""

    validation_voltage_rmse_mv: float
    validation_average_temperature_rmse_c: float


@dataclass(frozen=True)
class PhaseTwoConfig:
    """第二阶段全部配置的顶层容器。"""

    study_name: str
    random_seed: int
    battery: PhaseTwoBatteryConfig
    experiment: ExperimentConfig
    identification: IdentificationConfig
    success_criteria: SuccessCriteria


def _profile(raw_segments: list[dict[str, object]]) -> tuple[ProfileSegment, ...]:
    """把 YAML 中的工况列表转换成不可变的数据类。"""
    return tuple(ProfileSegment(**segment) for segment in raw_segments)


def load_phase_two_config(path: str | Path) -> PhaseTwoConfig:
    """读取并检查第二阶段 YAML 配置。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    experiment_raw = raw["experiment"]
    experiment = ExperimentConfig(
        sample_period_s=float(experiment_raw["sample_period_s"]),
        ocv_soc_points=tuple(float(x) for x in experiment_raw["ocv_soc_points"]),
        pulse_soc_points=tuple(float(x) for x in experiment_raw["pulse_soc_points"]),
        pulse_c_rates=tuple(float(x) for x in experiment_raw["pulse_c_rates"]),
        pulse_duration_s=float(experiment_raw["pulse_duration_s"]),
        rest_before_s=float(experiment_raw["rest_before_s"]),
        rest_after_s=float(experiment_raw["rest_after_s"]),
        thermal_training_initial_soc=float(
            experiment_raw["thermal_training_initial_soc"]
        ),
        thermal_training_profile=_profile(experiment_raw["thermal_training_profile"]),
        validation_initial_soc=float(experiment_raw["validation_initial_soc"]),
        validation_profile=_profile(experiment_raw["validation_profile"]),
    )
    config = PhaseTwoConfig(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        battery=PhaseTwoBatteryConfig(**raw["battery"]),
        experiment=experiment,
        identification=IdentificationConfig(**raw["identification"]),
        success_criteria=SuccessCriteria(**raw["success_criteria"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseTwoConfig) -> None:
    """在高保真仿真开始前拦截单位、范围和协议错误。"""
    if config.battery.model.upper() != "DFN":
        raise ValueError("第二阶段第一版只支持 DFN 高保真模型。")
    if config.battery.thermal_model != "lumped":
        raise ValueError("Chen2020 第一版热辨识必须使用已验证的 lumped 热模型。")
    if config.battery.nominal_capacity_ah <= 0:
        raise ValueError("标称容量必须为正数。")
    if config.experiment.sample_period_s <= 0:
        raise ValueError("采样周期必须为正数。")
    for points in (
        config.experiment.ocv_soc_points,
        config.experiment.pulse_soc_points,
    ):
        if not points or any(not 0.0 < soc < 1.0 for soc in points):
            raise ValueError("SOC 采样点必须位于 0 和 1 之间。")
    if not 0.0 < config.identification.core_heat_capacity_fraction < 1.0:
        raise ValueError("核心热容量比例必须位于 0 和 1 之间。")
    for segment in (
        *config.experiment.thermal_training_profile,
        *config.experiment.validation_profile,
    ):
        if segment.mode not in {"charge", "rest"}:
            raise ValueError(f"不支持的工况模式：{segment.mode}")
        if segment.duration_s <= 0:
            raise ValueError("每段工况时长必须为正数。")
        if segment.mode == "charge" and segment.c_rate <= 0:
            raise ValueError("充电工况的 C-rate 必须为正数。")
