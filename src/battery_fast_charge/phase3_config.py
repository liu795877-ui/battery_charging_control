"""第三阶段约束 MPC 教师控制器的配置读取与范围检查。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PhaseThreeBatteryConfig:
    """高保真虚拟电池和本阶段 SOC、温度范围。"""

    parameter_set: str
    model: str
    thermal_model: str
    nominal_capacity_ah: float
    initial_soc: float
    target_soc: float
    initial_temperature_c: float
    ambient_temperature_c: float


@dataclass(frozen=True)
class PhaseThreeArtifactConfig:
    """第二阶段输出文件相对于项目根目录的位置。"""

    identified_parameters: str
    validation_metrics: str
    ocv_curve: str


@dataclass(frozen=True)
class PhaseThreeConstraintConfig:
    """物理边界和 MPC 为模型误差预留的安全余量。"""

    physical_maximum_voltage_v: float
    physical_maximum_temperature_c: float
    maximum_current_a: float
    maximum_current_change_a_per_step: float
    voltage_uncertainty_margin_v: float
    temperature_uncertainty_margin_c: float

    @property
    def mpc_maximum_voltage_v(self) -> float:
        """降阶模型优化时使用的保守电压上限。"""
        return self.physical_maximum_voltage_v - self.voltage_uncertainty_margin_v

    @property
    def mpc_maximum_temperature_c(self) -> float:
        """降阶模型优化时使用的保守平均温度上限。"""
        return (
            self.physical_maximum_temperature_c
            - self.temperature_uncertainty_margin_c
        )


@dataclass(frozen=True)
class PhaseThreeControlConfig:
    """闭环周期、预测长度和控制分块。"""

    control_interval_s: float
    prediction_horizon_steps: int
    control_block_steps: int
    maximum_simulation_time_s: float

    @property
    def number_of_control_blocks(self) -> int:
        """优化变量数量；预测步数必须能被分块长度整除。"""
        return self.prediction_horizon_steps // self.control_block_steps


@dataclass(frozen=True)
class PhaseThreeObjectiveConfig:
    """快速充电目标与轻微平滑项的权重。"""

    soc_tracking_weight: float
    terminal_soc_weight: float
    current_change_weight: float


@dataclass(frozen=True)
class PhaseThreeOptimizerConfig:
    """SLSQP 数值求解设置。"""

    maximum_iterations: int
    function_tolerance: float
    constraint_tolerance: float


@dataclass(frozen=True)
class PhaseThreeValidationConfig:
    """第三阶段第一版的验收阈值。"""

    target_soc_tolerance: float
    physical_constraint_tolerance: float
    minimum_optimizer_success_fraction: float


@dataclass(frozen=True)
class PhaseThreeConfig:
    """第三阶段全部配置的只读容器。"""

    study_name: str
    random_seed: int
    battery: PhaseThreeBatteryConfig
    artifacts: PhaseThreeArtifactConfig
    constraints: PhaseThreeConstraintConfig
    control: PhaseThreeControlConfig
    objective: PhaseThreeObjectiveConfig
    optimizer: PhaseThreeOptimizerConfig
    validation: PhaseThreeValidationConfig


def load_phase_three_config(path: str | Path) -> PhaseThreeConfig:
    """从 YAML 读取配置，并在耗时仿真前拦截明显错误。"""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    config = PhaseThreeConfig(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        battery=PhaseThreeBatteryConfig(**raw["battery"]),
        artifacts=PhaseThreeArtifactConfig(**raw["artifacts"]),
        constraints=PhaseThreeConstraintConfig(**raw["constraints"]),
        control=PhaseThreeControlConfig(**raw["control"]),
        objective=PhaseThreeObjectiveConfig(**raw["objective"]),
        optimizer=PhaseThreeOptimizerConfig(**raw["optimizer"]),
        validation=PhaseThreeValidationConfig(**raw["validation"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseThreeConfig) -> None:
    """检查本阶段已经明确的研究范围和数值约定。"""
    battery = config.battery
    constraints = config.constraints
    control = config.control

    if battery.model.upper() != "DFN" or battery.thermal_model != "lumped":
        raise ValueError("第三阶段第一版必须使用 Chen2020 DFN + lumped thermal。")
    if not 0.0 <= battery.initial_soc < battery.target_soc <= 1.0:
        raise ValueError("SOC 必须满足 0 <= 初始值 < 目标值 <= 1。")
    if battery.nominal_capacity_ah <= 0.0:
        raise ValueError("标称容量必须为正数。")
    if constraints.mpc_maximum_voltage_v <= 0.0:
        raise ValueError("电压安全余量不能吃掉全部物理电压范围。")
    if constraints.mpc_maximum_temperature_c <= battery.ambient_temperature_c:
        raise ValueError("MPC 温度上限必须高于环境温度。")
    if constraints.maximum_current_a <= 0.0:
        raise ValueError("最大充电电流必须为正数。")
    if constraints.maximum_current_change_a_per_step <= 0.0:
        raise ValueError("每步最大电流变化必须为正数。")
    if control.control_interval_s <= 0.0 or control.maximum_simulation_time_s <= 0.0:
        raise ValueError("控制周期和最大仿真时间必须为正数。")
    if control.prediction_horizon_steps < 1 or control.control_block_steps < 1:
        raise ValueError("预测步数和控制分块长度必须至少为 1。")
    if control.prediction_horizon_steps % control.control_block_steps:
        raise ValueError("预测步数必须能被控制分块长度整除。")
    if not 0.0 < config.validation.minimum_optimizer_success_fraction <= 1.0:
        raise ValueError("优化成功率阈值必须位于 (0, 1]。")
