"""第三阶段 B：公平基线、可达轨迹和教师数据集配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FairBaselineConfig:
    """不使用预测优化的受约束 1C 基线设置。"""

    desired_current_a: float
    maximum_simulation_time_s: float


@dataclass(frozen=True)
class RolloutConfig:
    """一条探索轨迹的名称、策略类型和策略参数。"""

    name: str
    kind: str
    parameters: dict[str, float | int]


@dataclass(frozen=True)
class TeacherDatasetConfig:
    """分层抽样和按轨迹划分协议。"""

    samples_per_soc_bin_per_trajectory: int
    soc_bin_edges: tuple[float, ...]
    train_trajectory_count: int
    validation_trajectory_count: int
    test_trajectory_count: int
    rollouts: tuple[RolloutConfig, ...]


@dataclass(frozen=True)
class ActiveConstraintTolerances:
    """判断预测约束是否活跃的邻域宽度。"""

    voltage_v: float
    temperature_c: float
    current_a: float
    current_change_a: float


@dataclass(frozen=True)
class DatasetSuccessCriteria:
    """允许进入 DNN 训练之前的数据质量闸门。"""

    minimum_candidate_count: int
    minimum_teacher_acceptance_fraction: float
    minimum_samples_per_split: int
    minimum_samples_per_soc_bin: int
    require_voltage_active_samples: bool
    require_temperature_active_samples: bool


@dataclass(frozen=True)
class PhaseThreeBConfig:
    """第三阶段 B 全部设置。"""

    study_name: str
    random_seed: int
    source_phase3_config: str
    fair_baseline: FairBaselineConfig
    dataset: TeacherDatasetConfig
    active_constraint_tolerances: ActiveConstraintTolerances
    success_criteria: DatasetSuccessCriteria


def load_phase_three_b_config(path: str | Path) -> PhaseThreeBConfig:
    """读取 YAML 并验证数据协议，避免在批量求解后才发现划分错误。"""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    rollout_records = []
    for item in raw["dataset"]["rollouts"]:
        item = dict(item)
        name = str(item.pop("name"))
        kind = str(item.pop("kind"))
        rollout_records.append(RolloutConfig(name, kind, item))

    dataset_raw = raw["dataset"]
    dataset = TeacherDatasetConfig(
        samples_per_soc_bin_per_trajectory=int(
            dataset_raw["samples_per_soc_bin_per_trajectory"]
        ),
        soc_bin_edges=tuple(float(x) for x in dataset_raw["soc_bin_edges"]),
        train_trajectory_count=int(dataset_raw["train_trajectory_count"]),
        validation_trajectory_count=int(dataset_raw["validation_trajectory_count"]),
        test_trajectory_count=int(dataset_raw["test_trajectory_count"]),
        rollouts=tuple(rollout_records),
    )
    config = PhaseThreeBConfig(
        study_name=str(raw["study"]["name"]),
        random_seed=int(raw["study"]["random_seed"]),
        source_phase3_config=str(raw["source_phase3_config"]),
        fair_baseline=FairBaselineConfig(**raw["fair_baseline"]),
        dataset=dataset,
        active_constraint_tolerances=ActiveConstraintTolerances(
            **raw["active_constraint_tolerances"]
        ),
        success_criteria=DatasetSuccessCriteria(**raw["success_criteria"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseThreeBConfig) -> None:
    """检查轨迹数量、SOC分箱、比例和策略名称。"""
    dataset = config.dataset
    split_total = (
        dataset.train_trajectory_count
        + dataset.validation_trajectory_count
        + dataset.test_trajectory_count
    )
    if split_total != len(dataset.rollouts):
        raise ValueError("训练、验证、测试轨迹数之和必须等于探索轨迹总数。")
    if len(set(rollout.name for rollout in dataset.rollouts)) != len(dataset.rollouts):
        raise ValueError("探索轨迹名称必须唯一。")
    supported = {"constant", "soc_switch", "pulse", "sine", "random_blocks"}
    if any(rollout.kind not in supported for rollout in dataset.rollouts):
        raise ValueError(f"探索策略只支持：{sorted(supported)}")
    if dataset.samples_per_soc_bin_per_trajectory < 1:
        raise ValueError("每条轨迹每个SOC分箱至少抽取一个状态。")
    edges = dataset.soc_bin_edges
    if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError("SOC分箱边界必须严格递增。")
    if not 0.0 <= edges[0] < edges[-1] <= 1.0:
        raise ValueError("SOC分箱必须位于0到1之间。")
    if config.fair_baseline.desired_current_a <= 0.0:
        raise ValueError("公平基线目标电流必须为正数。")
    if not 0.0 < config.success_criteria.minimum_teacher_acceptance_fraction <= 1.0:
        raise ValueError("教师接受率阈值必须位于(0,1]。")
