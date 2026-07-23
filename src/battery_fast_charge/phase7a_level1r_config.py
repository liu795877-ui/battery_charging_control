"""Phase 7A Level 1R：只修改数据覆盖合同。"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Level1RCoverageConfig:
    terminal_trajectory_count: int
    trajectory_steps: int
    train_trajectory_count: int
    validation_trajectory_count: int
    terminal_test_trajectory_count: int
    tail_training_trajectory_count: int
    tail_soc_bounds: tuple[float, float]
    soc_bounds: tuple[float, float]
    polarization_bounds_v: tuple[float, float]
    random_seed: int
    original_frozen_test_trajectory_count: int
    minimum_low_current_label_count: int
    minimum_taper_current_label_count: int
    low_current_threshold_a: float
    taper_current_bounds_a: tuple[float, float]


@dataclass(frozen=True)
class Phase7ALevel1RConfig:
    study_name: str
    source_level1_config: str
    source_level1_dataset: str
    coverage: Level1RCoverageConfig


def _pair(value: object) -> tuple[float, float]:
    result = tuple(float(v) for v in value)  # type: ignore[arg-type]
    if len(result) != 2:
        raise ValueError("区间必须含两个端点。")
    return result


def load_phase7a_level1r_config(path: str | Path) -> Phase7ALevel1RConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    coverage = dict(raw["coverage"])
    for key in ("soc_bounds", "polarization_bounds_v", "tail_soc_bounds", "taper_current_bounds_a"):
        coverage[key] = _pair(coverage[key])
    config = Phase7ALevel1RConfig(
        study_name=str(raw["study"]["name"]),
        source_level1_config=str(raw["source_level1_config"]),
        source_level1_dataset=str(raw["source_level1_dataset"]),
        coverage=Level1RCoverageConfig(**coverage),
    )
    _validate(config)
    return config


def _validate(config: Phase7ALevel1RConfig) -> None:
    c = config.coverage
    if not 120 <= c.terminal_trajectory_count + c.tail_training_trajectory_count <= 160:
        raise ValueError("Level 1R 末端轨迹总数必须在 120–160 条之间。")
    if not 16 <= c.trajectory_steps <= 24:
        raise ValueError("Level 1R 每条末端轨迹必须展开 16–24 步。")
    if c.train_trajectory_count + c.validation_trajectory_count + c.terminal_test_trajectory_count != c.terminal_trajectory_count:
        raise ValueError("末端训练、验证和冻结测试轨迹数之和必须等于总数。")
    if c.soc_bounds != (0.74, 0.799) or c.polarization_bounds_v != (0.0, 0.10):
        raise ValueError("Level 1R 必须严格覆盖 SOC 0.74–0.799 与 Vp 0–0.10 V。")
    if c.tail_soc_bounds != (0.795, 0.799) or c.tail_training_trajectory_count != 20:
        raise ValueError("Level 1R 尾端加密必须为 20 条仅训练轨迹，覆盖 SOC 0.795–0.799。")
    if c.original_frozen_test_trajectory_count != 36:
        raise ValueError("原始 Level 1 冻结测试轨迹合同必须保持 36 条。")
