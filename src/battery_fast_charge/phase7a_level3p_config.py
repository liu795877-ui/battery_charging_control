"""Phase 7A Level 3P：冻结 Level 3 后的最小输出投影配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Level3PProjectionConfig:
    intervention_tolerance_a: float
    frozen_raw_violation_tolerance_a: float
    neighborhood_radius_steps: int


@dataclass(frozen=True)
class Level3POutputConfig:
    data_directory: str
    result_directory: str


@dataclass(frozen=True)
class Phase7ALevel3PConfig:
    study_name: str
    source_level3_config: str
    frozen_artifacts: dict[str, str]
    projection: Level3PProjectionConfig
    output: Level3POutputConfig


def load_phase7a_level3p_config(path: str | Path) -> Phase7ALevel3PConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    config = Phase7ALevel3PConfig(
        study_name=str(raw["study"]["name"]),
        source_level3_config=str(raw["source"]["level3_config"]),
        frozen_artifacts={
            str(key): str(value).lower()
            for key, value in raw["source"]["frozen_artifacts"].items()
        },
        projection=Level3PProjectionConfig(**raw["projection"]),
        output=Level3POutputConfig(**raw["output"]),
    )
    if config.projection.intervention_tolerance_a <= 0:
        raise ValueError("投影介入容差必须为正。")
    if config.projection.frozen_raw_violation_tolerance_a < 0:
        raise ValueError("冻结原始违约容差不能为负。")
    if config.projection.neighborhood_radius_steps != 1:
        raise ValueError("Level 3P 的附近动作定义必须冻结为 ±1 步。")
    if len(config.frozen_artifacts) != 13:
        raise ValueError("Level 3P 必须冻结 13 个 Level 3 核心工件。")
    return config
