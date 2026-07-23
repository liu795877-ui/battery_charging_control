"""阶段4B-2主动数据聚合与第二版小型ANN配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ActiveRolloutConfig:
    """旧ANN周围一条受约束探索轨迹。"""

    name: str
    kind: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ActiveDataConfig:
    """SOC分层采样数量和整轨迹划分。"""

    soc_bin_edges: tuple[float, ...]
    samples_per_middle_soc_bin_per_trajectory: int
    samples_per_edge_soc_bin_per_trajectory: int
    train_trajectory_count: int
    validation_trajectory_count: int
    test_trajectory_count: int
    rollouts: tuple[ActiveRolloutConfig, ...]


@dataclass(frozen=True)
class ActiveConstraintTolerances:
    """判断教师预测约束是否活跃的数值邻域。"""

    voltage_v: float
    temperature_c: float
    current_a: float
    current_change_a: float


@dataclass(frozen=True)
class DAggerRefinementConfig:
    """第一轮网络在策略轨迹的加密聚合设置。"""

    enabled: bool
    samples_per_soc_bin: int
    split: str


@dataclass(frozen=True)
class FinalNetworkConfig:
    """最终ANN v2容量、候选超参数和在策略权重。"""

    hidden_layer_sizes: tuple[int, ...]
    regularization_candidates: tuple[float, ...]
    initialization_seeds: tuple[int, ...]
    maximum_iterations: int
    convergence_tolerance: float
    on_policy_training_weight: float


@dataclass(frozen=True)
class PhaseFourB2SuccessCriteria:
    """数据、离线拟合、闭环安全层和实时性的联合闸门。"""

    minimum_accepted_label_count: int
    minimum_teacher_acceptance_fraction: float
    maximum_test_mae_a: float
    maximum_test_rmse_a: float
    material_intervention_threshold_a: float
    maximum_material_intervention_fraction: float
    maximum_mean_filter_correction_a: float
    maximum_dfn_time_gap_fraction_from_hybrid_teacher: float
    require_improvement_over_phase4a: bool
    minimum_inference_speedup_over_hybrid_mpc: float


@dataclass(frozen=True)
class PhaseFourB2Config:
    """阶段4B-2全部配置。"""

    study_name: str
    random_seed: int
    source_phase3_config: str
    source_phase4a_config: str
    source_phase4b_config: str
    legacy_teacher_dataset: str
    seed_ann_model: str
    active_data: ActiveDataConfig
    active_constraint_tolerances: ActiveConstraintTolerances
    dagger_refinement: DAggerRefinementConfig
    dfn_refinement: DAggerRefinementConfig
    final_network: FinalNetworkConfig
    success_criteria: PhaseFourB2SuccessCriteria


def load_phase_four_b2_config(path: str | Path) -> PhaseFourB2Config:
    """读取YAML并锁定12条轨迹、七个SOC分箱和三组隔离划分。"""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    active_raw = dict(raw["active_data"])
    rollouts = []
    for item in active_raw.pop("rollouts"):
        values = dict(item)
        name = str(values.pop("name"))
        kind = str(values.pop("kind"))
        rollouts.append(ActiveRolloutConfig(name, kind, values))
    active_raw["soc_bin_edges"] = tuple(
        float(value) for value in active_raw["soc_bin_edges"]
    )
    active_raw["rollouts"] = tuple(rollouts)
    final_network_raw = dict(raw["final_network"])
    final_network_raw["hidden_layer_sizes"] = tuple(
        int(value) for value in final_network_raw["hidden_layer_sizes"]
    )
    final_network_raw["regularization_candidates"] = tuple(
        float(value) for value in final_network_raw["regularization_candidates"]
    )
    final_network_raw["initialization_seeds"] = tuple(
        int(value) for value in final_network_raw["initialization_seeds"]
    )
    config = PhaseFourB2Config(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        source_phase3_config=str(raw["source_phase3_config"]),
        source_phase4a_config=str(raw["source_phase4a_config"]),
        source_phase4b_config=str(raw["source_phase4b_config"]),
        legacy_teacher_dataset=str(raw["legacy_teacher_dataset"]),
        seed_ann_model=str(raw["seed_ann_model"]),
        active_data=ActiveDataConfig(**active_raw),
        active_constraint_tolerances=ActiveConstraintTolerances(
            **raw["active_constraint_tolerances"]
        ),
        dagger_refinement=DAggerRefinementConfig(**raw["dagger_refinement"]),
        dfn_refinement=DAggerRefinementConfig(**raw["dfn_refinement"]),
        final_network=FinalNetworkConfig(**final_network_raw),
        success_criteria=PhaseFourB2SuccessCriteria(**raw["success_criteria"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseFourB2Config) -> None:
    """拒绝会破坏可达性、SOC覆盖或整轨迹隔离的设置。"""
    active = config.active_data
    if len(active.rollouts) != (
        active.train_trajectory_count
        + active.validation_trajectory_count
        + active.test_trajectory_count
    ):
        raise ValueError("主动轨迹总数必须等于训练、验证和测试轨迹数之和。")
    if len(set(item.name for item in active.rollouts)) != len(active.rollouts):
        raise ValueError("主动轨迹名称不能重复。")
    if active.soc_bin_edges[0] != 0.10 or active.soc_bin_edges[-1] != 0.80:
        raise ValueError("主动数据必须覆盖10%到80% SOC。")
    if any(
        right <= left
        for left, right in zip(active.soc_bin_edges[:-1], active.soc_bin_edges[1:])
    ):
        raise ValueError("SOC分箱边界必须严格递增。")
    if config.success_criteria.material_intervention_threshold_a <= 0.0:
        raise ValueError("实质介入阈值必须为正电流。")
    if config.dagger_refinement.samples_per_soc_bin < 1:
        raise ValueError("DAgger每个SOC分箱至少需要一个状态。")
    if not config.dagger_refinement.enabled:
        raise ValueError("阶段4B-2固定要求一次在策略DAgger精炼。")
    if config.dagger_refinement.split != "train":
        raise ValueError("在策略DAgger状态只能加入训练集，禁止污染验证或测试集。")
    if not config.dfn_refinement.enabled:
        raise ValueError("阶段4B-2固定要求一次DFN在策略精炼。")
    if config.dfn_refinement.samples_per_soc_bin < 1:
        raise ValueError("DFN精炼每个SOC分箱至少需要一个状态。")
    if config.dfn_refinement.split != "train":
        raise ValueError("DFN在策略状态只能加入训练集，禁止污染验证或测试集。")
    if any(width < 1 for width in config.final_network.hidden_layer_sizes):
        raise ValueError("最终网络隐藏层宽度必须为正整数。")
    if config.final_network.on_policy_training_weight < 1.0:
        raise ValueError("在策略训练权重不能小于普通训练状态。")
