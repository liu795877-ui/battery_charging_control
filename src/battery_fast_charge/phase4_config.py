"""阶段4A小型ANN训练与闭环验证配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TinyNetworkConfig:
    """固定网络规模以及用验证集选择的候选超参数。"""

    hidden_layer_sizes: tuple[int, ...]
    activation: str
    solver: str
    regularization_candidates: tuple[float, ...]
    initialization_seeds: tuple[int, ...]
    maximum_iterations: int
    convergence_tolerance: float


@dataclass(frozen=True)
class ANNClosedLoopConfig:
    """ANN闭环最长时间和安全包装设置。"""

    maximum_simulation_time_s: float
    use_safety_filter: bool


@dataclass(frozen=True)
class ANNSuccessCriteria:
    """第一版模仿学习的研究性验收阈值，不代表实物安全标准。"""

    maximum_test_mae_a: float
    maximum_test_rmse_a: float
    maximum_active_temperature_mae_a: float
    maximum_dfn_time_gap_fraction: float
    minimum_inference_speedup_over_mpc: float


@dataclass(frozen=True)
class PhaseFourAConfig:
    """阶段4A全部配置。"""

    study_name: str
    random_seed: int
    source_phase3_config: str
    teacher_dataset: str
    features: tuple[str, ...]
    target: str
    network: TinyNetworkConfig
    closed_loop: ANNClosedLoopConfig
    success_criteria: ANNSuccessCriteria


def load_phase_four_a_config(path: str | Path) -> PhaseFourAConfig:
    """读取并验证YAML，避免训练后才发现数据协议不一致。"""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    network = dict(raw["network"])
    network["hidden_layer_sizes"] = tuple(
        int(value) for value in network["hidden_layer_sizes"]
    )
    network["regularization_candidates"] = tuple(
        float(value) for value in network["regularization_candidates"]
    )
    network["initialization_seeds"] = tuple(
        int(value) for value in network["initialization_seeds"]
    )
    config = PhaseFourAConfig(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        source_phase3_config=str(raw["source_phase3_config"]),
        teacher_dataset=str(raw["teacher_dataset"]),
        features=tuple(str(value) for value in raw["features"]),
        target=str(raw["target"]),
        network=TinyNetworkConfig(**network),
        closed_loop=ANNClosedLoopConfig(**raw["closed_loop"]),
        success_criteria=ANNSuccessCriteria(**raw["success_criteria"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseFourAConfig) -> None:
    """锁定五个可用状态、一个输出以及合理的网络搜索范围。"""
    expected_features = (
        "state_soc",
        "state_polarization_fast_v",
        "state_polarization_slow_v",
        "state_average_temperature_c",
        "state_previous_current_a",
    )
    if config.features != expected_features:
        raise ValueError("阶段4A必须使用已经约定的五个DNN输入特征。")
    if config.target != "teacher_current_a":
        raise ValueError("阶段4A监督标签必须是MPC第一步电流。")
    if config.network.activation != "tanh":
        raise ValueError("NumPy部署器当前只实现tanh隐藏层。")
    if config.network.solver != "lbfgs":
        raise ValueError("小样本第一版固定使用确定性较强的L-BFGS求解器。")
    if not config.network.hidden_layer_sizes or any(
        width < 1 for width in config.network.hidden_layer_sizes
    ):
        raise ValueError("隐藏层必须至少包含一个正整数宽度。")
    if any(value <= 0.0 for value in config.network.regularization_candidates):
        raise ValueError("正则化候选值必须为正数。")
    if len(set(config.network.initialization_seeds)) != len(
        config.network.initialization_seeds
    ):
        raise ValueError("随机初始化种子不能重复。")
    if not config.closed_loop.use_safety_filter:
        raise ValueError("第一版ANN闭环禁止关闭安全过滤器。")
