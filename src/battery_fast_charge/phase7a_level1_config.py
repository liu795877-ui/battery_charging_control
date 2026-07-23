"""Phase 7A Level 1 的独立、可审计配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Level1ModelConfig:
    nominal_capacity_ah: float
    sample_period_s: float
    r0_ohm: float
    r1_ohm: float
    tau1_s: float
    ocv_curve: str


@dataclass(frozen=True)
class Level1MPCConfig:
    prediction_horizon_steps: int
    control_block_steps: int
    target_soc: float
    current_bounds_a: tuple[float, float]
    terminal_voltage_max_v: float
    soc_tracking_weight: float
    terminal_soc_weight: float
    current_smoothness_weight: float
    optimizer_max_iterations: int
    optimizer_ftol: float
    constraint_tolerance: float


@dataclass(frozen=True)
class Level1DataConfig:
    trajectory_count: int
    trajectory_steps: int
    train_fraction: float
    validation_fraction: float
    soc_bounds: tuple[float, float]
    polarization_bounds_v: tuple[float, float]
    random_seed: int
    audit_state_count: int
    warm_starts_per_state: int
    closed_loop_trajectory_count: int
    closed_loop_soc_bounds: tuple[float, float]
    maximum_closed_loop_steps: int


@dataclass(frozen=True)
class Level1NetworkConfig:
    hidden_layer_sizes: tuple[int, ...]
    activation: str
    regularization_alpha: float
    initialization_seeds: tuple[int, ...]
    maximum_iterations: int
    convergence_tolerance: float
    learning_rate_init: float
    no_improvement_iterations: int


@dataclass(frozen=True)
class Level1GateConfig:
    minimum_teacher_acceptance_fraction: float
    maximum_multivalued_state_fraction: float
    maximum_near_optimal_action_range_p95_a: float
    relative_objective_tolerance: float
    absolute_objective_tolerance: float
    offline_nrmse_max: float
    closed_loop_current_nrmse_max: float
    charge_time_gap_fraction_max: float
    minimum_target_reach_fraction: float
    maximum_constraint_violation: float
    minimum_speedup: float


@dataclass(frozen=True)
class Phase7ALevel1Config:
    study_name: str
    model: Level1ModelConfig
    mpc: Level1MPCConfig
    data: Level1DataConfig
    network: Level1NetworkConfig
    gates: Level1GateConfig


def _floats(value: object) -> tuple[float, ...]:
    return tuple(float(item) for item in value)  # type: ignore[arg-type]


def load_phase7a_level1_config(path: str | Path) -> Phase7ALevel1Config:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    model = Level1ModelConfig(**raw["model"])
    mpc_raw = dict(raw["mpc"])
    mpc_raw["current_bounds_a"] = _floats(mpc_raw["current_bounds_a"])
    data_raw = dict(raw["data"])
    for key in ("soc_bounds", "polarization_bounds_v", "closed_loop_soc_bounds"):
        data_raw[key] = _floats(data_raw[key])
    network_raw = dict(raw["network"])
    network_raw["hidden_layer_sizes"] = tuple(int(v) for v in network_raw["hidden_layer_sizes"])
    network_raw["initialization_seeds"] = tuple(int(v) for v in network_raw["initialization_seeds"])
    config = Phase7ALevel1Config(
        study_name=str(raw["study"]["name"]),
        model=model,
        mpc=Level1MPCConfig(**mpc_raw),
        data=Level1DataConfig(**data_raw),
        network=Level1NetworkConfig(**network_raw),
        gates=Level1GateConfig(**raw["gates"]),
    )
    _validate(config)
    return config


def _validate(config: Phase7ALevel1Config) -> None:
    if (config.data.trajectory_count, config.data.trajectory_steps) != (240, 8):
        raise ValueError("Level 1 教师候选数据合同必须为 240 条轨迹、每条 8 步。")
    if (config.data.audit_state_count, config.data.warm_starts_per_state) != (100, 15):
        raise ValueError("Level 1 教师审计合同必须为 100 个状态、每状态 15 个 warm start。")
    if len(config.network.initialization_seeds) != 5:
        raise ValueError("Level 1 必须训练恰好 5 个独立随机种子。")
    if config.mpc.current_bounds_a[0] < 0 or config.mpc.current_bounds_a[1] <= config.mpc.current_bounds_a[0]:
        raise ValueError("电流上下限无效。")
    if config.mpc.prediction_horizon_steps % config.mpc.control_block_steps:
        raise ValueError("预测步数必须能被控制块长度整除。")
    if config.data.train_fraction + config.data.validation_fraction >= 1.0:
        raise ValueError("必须保留非空冻结测试集。")
