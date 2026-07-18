"""读取并检查第一阶段实验配置。

配置写在 ``configs/phase1.yaml`` 中。本模块把 YAML 中容易修改的参数转换成
带类型提示的只读数据类，后续仿真代码只需访问 ``config.battery`` 等字段，
不必反复处理字典键名。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BatteryConfig:
    """电芯模型配置；SOC 无量纲，温度单位为摄氏度（°C）。"""

    parameter_set: str
    model: str
    thermal_model: str
    nominal_capacity_ah: float
    initial_soc: float
    target_soc: float
    initial_temperature_c: float
    ambient_temperature_c: float


@dataclass(frozen=True)
class ConstraintConfig:
    """充电约束；电压 V、电流 A、温度 °C、电流变化量 A/控制步。"""

    maximum_voltage_v: float
    maximum_current_a: float
    maximum_temperature_c: float
    maximum_current_change_a_per_control_step: float


@dataclass(frozen=True)
class ControlConfig:
    """离散控制配置：控制周期和 MPC 预测步数。"""

    control_interval_s: float
    prediction_horizon_steps: int

    @property
    def prediction_horizon_s(self) -> float:
        """把“预测步数”换算成更直观的预测总时长（秒）。"""
        return self.control_interval_s * self.prediction_horizon_steps


@dataclass(frozen=True)
class BaselineConfig:
    """公平对照用的 CC–CV 工况配置。"""

    c_rates: tuple[float, ...]
    cv_cutoff_c_rate: float


@dataclass(frozen=True)
class PhaseOneConfig:
    """第一阶段全部配置的顶层容器。"""

    study_name: str
    random_seed: int
    battery: BatteryConfig
    constraints: ConstraintConfig
    control: ControlConfig
    baseline: BaselineConfig


def load_config(path: str | Path) -> PhaseOneConfig:
    """读取 YAML、完成类型转换，并在返回前检查关键参数是否合法。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    # YAML 的四个区域分别转换为数据类。使用冻结数据类可以避免仿真过程中
    # 某段代码无意中修改配置，导致各工况不再具有可比性。
    battery = BatteryConfig(**raw["battery"])
    constraints = ConstraintConfig(**raw["constraints"])
    control = ControlConfig(**raw["control"])
    baseline_raw = raw["baseline"]
    baseline = BaselineConfig(
        c_rates=tuple(float(value) for value in baseline_raw["c_rates"]),
        cv_cutoff_c_rate=float(baseline_raw["cv_cutoff_c_rate"]),
    )
    config = PhaseOneConfig(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        battery=battery,
        constraints=constraints,
        control=control,
        baseline=baseline,
    )
    _validate_config(config)
    return config


def _validate_config(config: PhaseOneConfig) -> None:
    """尽早拦截明显错误，避免运行耗时仿真后才发现配置无效。"""
    battery = config.battery
    constraints = config.constraints
    control = config.control

    if not 0.0 <= battery.initial_soc < battery.target_soc <= 1.0:
        raise ValueError("SOC bounds must satisfy 0 <= initial < target <= 1.")
    if battery.nominal_capacity_ah <= 0.0:
        raise ValueError("Nominal capacity must be positive.")
    if constraints.maximum_voltage_v <= 0.0:
        raise ValueError("Maximum voltage must be positive.")
    if constraints.maximum_current_a <= 0.0:
        raise ValueError("Maximum current must be positive.")
    if control.control_interval_s <= 0.0:
        raise ValueError("Control interval must be positive.")
    if control.prediction_horizon_steps < 1:
        raise ValueError("Prediction horizon must contain at least one step.")
    if any(rate <= 0.0 for rate in config.baseline.c_rates):
        raise ValueError("All baseline C-rates must be positive.")
