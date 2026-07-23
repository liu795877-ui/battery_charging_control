"""阶段4B-1热预算MPC教师配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ThermalBudgetReferenceConfig:
    """由可行最短时间策略搜索得到的状态触发电流参考。"""

    peak_current_a: float
    sustainable_current_a: float
    switch_soc: float
    switch_average_temperature_c: float
    reference_release_soc: float
    reference_tracking_weight: float
    reoptimize_every_control_step: bool


@dataclass(frozen=True)
class DiagnosticSweepConfig:
    """用于复核速度空间的窄范围参数搜索。"""

    peak_currents_a: tuple[float, ...]
    switch_times_min: tuple[float, ...]
    sustainable_currents_a: tuple[float, ...]


@dataclass(frozen=True)
class PhaseFourBSuccessCriteria:
    """允许进入主动数据聚合前的教师验收闸门。"""

    minimum_improvement_fraction_over_filtered_1c: float
    minimum_optimizer_success_fraction: float
    require_zero_fallbacks: bool


@dataclass(frozen=True)
class PhaseFourBConfig:
    """阶段4B-1全部设置。"""

    study_name: str
    random_seed: int
    source_phase3_config: str
    thermal_budget_reference: ThermalBudgetReferenceConfig
    diagnostic_sweep: DiagnosticSweepConfig
    success_criteria: PhaseFourBSuccessCriteria


def load_phase_four_b_config(path: str | Path) -> PhaseFourBConfig:
    """读取YAML并验证状态触发参考和教师验收条件。"""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    sweep = raw["diagnostic_sweep"]
    config = PhaseFourBConfig(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        source_phase3_config=str(raw["source_phase3_config"]),
        thermal_budget_reference=ThermalBudgetReferenceConfig(
            **raw["thermal_budget_reference"]
        ),
        diagnostic_sweep=DiagnosticSweepConfig(
            peak_currents_a=tuple(float(v) for v in sweep["peak_currents_a"]),
            switch_times_min=tuple(float(v) for v in sweep["switch_times_min"]),
            sustainable_currents_a=tuple(
                float(v) for v in sweep["sustainable_currents_a"]
            ),
        ),
        success_criteria=PhaseFourBSuccessCriteria(**raw["success_criteria"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseFourBConfig) -> None:
    """检查参考电流、切换状态和百分比阈值。"""
    reference = config.thermal_budget_reference
    if not 0.0 < reference.sustainable_current_a < reference.peak_current_a <= 10.0:
        raise ValueError("热预算参考必须满足0 < 可持续电流 < 峰值电流 <= 10 A。")
    if not 0.1 < reference.switch_soc < 0.8:
        raise ValueError("SOC切换点必须位于研究SOC区间内部。")
    if not reference.switch_soc < reference.reference_release_soc < 0.8:
        raise ValueError("参考释放SOC必须高于切换SOC且低于目标SOC。")
    if reference.switch_average_temperature_c <= 25.0:
        raise ValueError("温度切换点必须高于初始温度。")
    if reference.reference_tracking_weight <= 0.0:
        raise ValueError("参考跟踪权重必须为正数。")
    if not reference.reoptimize_every_control_step:
        raise ValueError("改进教师必须每个5 s控制周期重新求解。")
    improvement = config.success_criteria.minimum_improvement_fraction_over_filtered_1c
    if not 0.0 < improvement < 1.0:
        raise ValueError("相对时间改善阈值必须位于(0,1)。")
