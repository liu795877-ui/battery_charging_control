"""Phase 6P-0 论文 NDC 原位复现的独立配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class NDCParameters:
    bulk_capacitance_f: float
    surface_capacitance_f: float
    bulk_resistance_ohm: float
    surface_resistance_ohm: float
    beta: tuple[float, float, float]
    alpha: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class NDCMPCConfig:
    sample_period_s: float
    prediction_horizon: int
    control_horizon: int
    constraint_horizon: int
    maximum_closed_loop_steps: int
    initial_previous_current_a: float
    current_bounds_a: tuple[float, float]
    surface_voltage_max_v: float
    terminal_voltage_max_v: float
    target_soc: float
    health_slope: float
    health_intercept_v: float
    soc_tracking_weight: float
    current_increment_weight: float
    optimizer_max_iterations: int
    optimizer_ftol: float
    constraint_tolerance: float


@dataclass(frozen=True)
class NDCDataConfig:
    state_bounds_v: tuple[float, float]
    hammersley_count: int
    factorial_count: int
    training_trajectory_steps: int
    independent_test_trajectories: int
    internal_training_fraction: float
    random_seed: int


@dataclass(frozen=True)
class NDCDNNConfig:
    hidden_layer_sizes: tuple[int, int, int]
    activation: str
    initialization_seeds: tuple[int, ...]
    regularization_initial: float
    noise_precision_initial: float
    bayesian_updates: int
    maximum_function_evaluations: int
    optimization_tolerance: float


@dataclass(frozen=True)
class NDCGateConfig:
    offline_nrmse_percent_max: float
    closed_loop_current_nrmse_percent_max: float
    maximum_constraint_violation_order_a_or_v: float
    minimum_target_reach_fraction: float
    minimum_speedup: float


@dataclass(frozen=True)
class PhaseSixPZeroConfig:
    study_name: str
    source_doi: str
    ndc: NDCParameters
    mpc: NDCMPCConfig
    data: NDCDataConfig
    dnn: NDCDNNConfig
    gates: NDCGateConfig


def _pair(values: object) -> tuple[float, float]:
    result = tuple(float(value) for value in values)  # type: ignore[arg-type]
    if len(result) != 2:
        raise ValueError("区间必须恰好包含两个端点。")
    return result


def load_phase_six_p_zero_config(path: str | Path) -> PhaseSixPZeroConfig:
    """读取并严格校验 Phase 6P-0 YAML。"""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    ndc = dict(raw["ndc"])
    ndc["beta"] = tuple(float(value) for value in ndc["beta"])
    ndc["alpha"] = tuple(float(value) for value in ndc["alpha"])
    mpc = dict(raw["mpc"])
    mpc["current_bounds_a"] = _pair(mpc["current_bounds_a"])
    data = dict(raw["data"])
    data["state_bounds_v"] = _pair(data["state_bounds_v"])
    dnn = dict(raw["dnn"])
    dnn["hidden_layer_sizes"] = tuple(int(value) for value in dnn["hidden_layer_sizes"])
    dnn["initialization_seeds"] = tuple(int(value) for value in dnn["initialization_seeds"])
    config = PhaseSixPZeroConfig(
        study_name=str(raw["study"]["name"]),
        source_doi=str(raw["study"]["source_doi"]),
        ndc=NDCParameters(**ndc),
        mpc=NDCMPCConfig(**mpc),
        data=NDCDataConfig(**data),
        dnn=NDCDNNConfig(**dnn),
        gates=NDCGateConfig(**raw["gates"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseSixPZeroConfig) -> None:
    if len(config.ndc.beta) != 3 or len(config.ndc.alpha) != 6:
        raise ValueError("NDC 参数必须包含 3 个 beta 和 6 个 alpha。")
    if config.mpc.sample_period_s != 60.0:
        raise ValueError("Phase 6P-0 必须保持论文的 60 s 控制周期。")
    if (config.mpc.prediction_horizon, config.mpc.control_horizon, config.mpc.constraint_horizon) != (10, 2, 1):
        raise ValueError("Phase 6P-0 必须保持论文的 Np=10、Nu=2、Nc=1。")
    if config.data.hammersley_count != 324 or config.data.factorial_count != 76:
        raise ValueError("论文 NDC 训练设计必须由 324+76 个初态组成。")
    if config.data.training_trajectory_steps != 5:
        raise ValueError("每个训练初态必须闭环展开 5 步。")
    if config.data.independent_test_trajectories != 30:
        raise ValueError("独立冻结测试集必须包含 30 条轨迹。")
    if config.dnn.hidden_layer_sizes != (7, 5, 3) or config.dnn.activation != "logistic_sigmoid":
        raise ValueError("论文阳性对照固定使用 2-7-5-3-1 sigmoid 网络。")
