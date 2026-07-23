"""阶段5A有界鲁棒性压力测试配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ReducedStressTestConfig:
    """降阶模型参数域、初始条件域和状态估计误差域。"""

    random_scenario_count: int
    initial_soc_range: tuple[float, float]
    ambient_temperature_c_range: tuple[float, float]
    capacity_multiplier_range: tuple[float, float]
    resistance_multiplier_range: tuple[float, float]
    time_constant_multiplier_range: tuple[float, float]
    heat_capacity_multiplier_range: tuple[float, float]
    thermal_resistance_multiplier_range: tuple[float, float]
    heat_gain_multiplier_range: tuple[float, float]
    soc_bias_range: tuple[float, float]
    soc_noise_standard_deviation: float
    temperature_bias_c_range: tuple[float, float]
    temperature_noise_standard_deviation_c: float
    polarization_bias_v_range: tuple[float, float]
    polarization_noise_standard_deviation_v: float
    terminal_true_soc_tolerance: float
    maximum_simulation_time_s: float


@dataclass(frozen=True)
class DFNTemperatureAnchorConfig:
    """少量高保真温度锚点，而非全面概率扫描。"""

    temperatures_c: tuple[float, ...]
    initial_soc: float
    maximum_simulation_time_s: float


@dataclass(frozen=True)
class PhaseFiveASuccessCriteria:
    """完成率、物理安全、时间和安全层依赖的联合闸门。"""

    minimum_reduced_completion_fraction: float
    minimum_reduced_physical_safety_fraction: float
    maximum_reduced_charge_time_min: float
    maximum_reduced_material_intervention_fraction: float
    require_all_dfn_anchors_complete: bool
    require_all_dfn_anchors_physically_safe: bool
    maximum_dfn_anchor_charge_time_min: float
    maximum_dfn_material_intervention_fraction: float


@dataclass(frozen=True)
class PhaseFiveAConfig:
    """阶段5A全部配置。"""

    study_name: str
    random_seed: int
    source_phase3_config: str
    source_phase4a_config: str
    ann_model: str
    phase4b2_metrics: str
    reduced_stress_test: ReducedStressTestConfig
    dfn_temperature_anchors: DFNTemperatureAnchorConfig
    success_criteria: PhaseFiveASuccessCriteria


def _pair(values) -> tuple[float, float]:
    pair = tuple(float(value) for value in values)
    if len(pair) != 2:
        raise ValueError("扰动范围必须包含下限和上限两个数。")
    return pair


def load_phase_five_a_config(path: str | Path) -> PhaseFiveAConfig:
    """读取并验证阶段5A YAML。"""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    stress = dict(raw["reduced_stress_test"])
    for key in (
        "initial_soc_range",
        "ambient_temperature_c_range",
        "capacity_multiplier_range",
        "resistance_multiplier_range",
        "time_constant_multiplier_range",
        "heat_capacity_multiplier_range",
        "thermal_resistance_multiplier_range",
        "heat_gain_multiplier_range",
        "soc_bias_range",
        "temperature_bias_c_range",
        "polarization_bias_v_range",
    ):
        stress[key] = _pair(stress[key])
    anchors = dict(raw["dfn_temperature_anchors"])
    anchors["temperatures_c"] = tuple(
        float(value) for value in anchors["temperatures_c"]
    )
    config = PhaseFiveAConfig(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        source_phase3_config=str(raw["source_phase3_config"]),
        source_phase4a_config=str(raw["source_phase4a_config"]),
        ann_model=str(raw["ann_model"]),
        phase4b2_metrics=str(raw["phase4b2_metrics"]),
        reduced_stress_test=ReducedStressTestConfig(**stress),
        dfn_temperature_anchors=DFNTemperatureAnchorConfig(**anchors),
        success_criteria=PhaseFiveASuccessCriteria(**raw["success_criteria"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseFiveAConfig) -> None:
    """拒绝退化范围、过热初态和无意义的成功率。"""
    stress = config.reduced_stress_test
    range_fields = [
        value
        for key, value in stress.__dict__.items()
        if key.endswith("_range")
    ]
    if any(high <= low for low, high in range_fields):
        raise ValueError("所有扰动范围都必须满足上限大于下限。")
    if stress.random_scenario_count < 1:
        raise ValueError("至少需要一个随机压力场景。")
    if not 0.0 <= stress.initial_soc_range[0] < stress.initial_soc_range[1] < 0.8:
        raise ValueError("初始SOC压力范围必须位于0到80%之间。")
    if max(stress.ambient_temperature_c_range) >= 33.5:
        raise ValueError("环境温度必须低于当前33.5 ℃收紧控制边界。")
    if any(
        temperature >= 33.5
        for temperature in config.dfn_temperature_anchors.temperatures_c
    ):
        raise ValueError("DFN温度锚点必须低于收紧控制边界。")
    for value in (
        config.success_criteria.minimum_reduced_completion_fraction,
        config.success_criteria.minimum_reduced_physical_safety_fraction,
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError("成功率阈值必须位于(0,1]。")
