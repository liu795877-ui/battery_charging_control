"""Phase 6 论文式显式 MPC 迁移验证配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


FEATURE_NAMES = (
    "state_soc",
    "state_polarization_fast_v",
    "state_polarization_slow_v",
    "state_average_temperature_c",
    "state_previous_current_a",
)


@dataclass(frozen=True)
class PaperMethodConfig:
    """论文式初始状态设计和短轨迹展开设置。"""

    article: str
    doi: str
    initial_state_count: int
    trajectory_steps: int
    hammersley_fraction: float
    factorial_levels: tuple[int, ...]
    state_ranges: dict[str, tuple[float, float]]
    train_fraction: float
    validation_fraction: float


@dataclass(frozen=True)
class PaperNetworkConfig:
    """限制在论文搜索范围内的三隐层候选网络。"""

    candidate_hidden_layer_sizes: tuple[tuple[int, ...], ...]
    activation: str
    solver: str
    regularization_candidates: tuple[float, ...]
    initialization_seeds: tuple[int, ...]
    maximum_iterations: int
    convergence_tolerance: float


@dataclass(frozen=True)
class NominalValidationConfig:
    """25 ℃ 名义 DFN 闭环与 MPC 教师的对照设置。"""

    temperature_c: float
    maximum_simulation_time_s: float
    teacher_trajectory: str
    teacher_metrics: str
    current_nrmse_normalization_a: float


@dataclass(frozen=True)
class PhaseSixSuccessCriteria:
    """名义门槛和后续多温度门槛；不因实验结果而自动放宽。"""

    minimum_accepted_initial_states: int
    minimum_teacher_acceptance_fraction: float
    maximum_nominal_current_nrmse: float
    maximum_nominal_charge_time_gap_fraction: float
    minimum_inference_speedup_over_mpc: float
    maximum_voltage_violation_v: float
    maximum_temperature_violation_c: float
    maximum_current_violation_a: float
    maximum_current_change_violation_a: float
    require_all_temperature_anchors_complete: bool
    require_all_temperature_anchors_without_serious_violation: bool


@dataclass(frozen=True)
class PhaseSixConfig:
    """Phase 6 全部配置。"""

    study_name: str
    random_seed: int
    source_phase3_config: str
    source_phase5a_config: str
    paper_method: PaperMethodConfig
    network: PaperNetworkConfig
    nominal_validation: NominalValidationConfig
    success_criteria: PhaseSixSuccessCriteria


def _pair(values: Any) -> tuple[float, float]:
    pair = tuple(float(value) for value in values)
    if len(pair) != 2:
        raise ValueError("状态范围必须包含下限和上限两个值。")
    return pair


def load_phase_six_config(path: str | Path) -> PhaseSixConfig:
    """读取 YAML，并在耗时实验开始前锁定数据与验收契约。"""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    paper = dict(raw["paper_method"])
    paper["factorial_levels"] = tuple(int(v) for v in paper["factorial_levels"])
    paper["state_ranges"] = {
        str(name): _pair(values) for name, values in paper["state_ranges"].items()
    }
    network = dict(raw["network"])
    network["candidate_hidden_layer_sizes"] = tuple(
        tuple(int(width) for width in architecture)
        for architecture in network["candidate_hidden_layer_sizes"]
    )
    network["regularization_candidates"] = tuple(
        float(value) for value in network["regularization_candidates"]
    )
    network["initialization_seeds"] = tuple(
        int(value) for value in network["initialization_seeds"]
    )
    config = PhaseSixConfig(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        source_phase3_config=str(raw["source_phase3_config"]),
        source_phase5a_config=str(raw["source_phase5a_config"]),
        paper_method=PaperMethodConfig(**paper),
        network=PaperNetworkConfig(**network),
        nominal_validation=NominalValidationConfig(**raw["nominal_validation"]),
        success_criteria=PhaseSixSuccessCriteria(**raw["success_criteria"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseSixConfig) -> None:
    paper = config.paper_method
    if tuple(paper.state_ranges) != FEATURE_NAMES:
        raise ValueError("Phase 6 状态范围必须按固定五维输入顺序声明。")
    if paper.initial_state_count < 10 or paper.trajectory_steps < 2:
        raise ValueError("论文式数据生成至少需要 10 个初态和 2 个轨迹步。")
    if len(paper.factorial_levels) != len(FEATURE_NAMES):
        raise ValueError("全因子层级数必须与五维状态一致。")
    factorial_count = 1
    for level in paper.factorial_levels:
        if level < 1:
            raise ValueError("每个全因子层级数必须为正整数。")
        factorial_count *= level
    if factorial_count >= paper.initial_state_count:
        raise ValueError("Hammersley 样本数必须大于零。")
    if any(high <= low for low, high in paper.state_ranges.values()):
        raise ValueError("所有状态范围都必须满足上限大于下限。")
    if not 0.0 < paper.train_fraction < 1.0:
        raise ValueError("训练比例必须位于 (0,1)。")
    if not 0.0 < paper.validation_fraction < 1.0:
        raise ValueError("验证比例必须位于 (0,1)。")
    if paper.train_fraction + paper.validation_fraction >= 1.0:
        raise ValueError("必须为独立测试轨迹保留非零比例。")
    if config.network.activation != "tanh" or config.network.solver != "lbfgs":
        raise ValueError("当前可审计 NumPy 导出固定使用 tanh 与 L-BFGS。")
    if any(len(a) != 3 or any(w < 2 or w > 20 for w in a)
           for a in config.network.candidate_hidden_layer_sizes):
        raise ValueError("候选网络必须含三层且每层宽度位于论文的 2–20 范围。")
