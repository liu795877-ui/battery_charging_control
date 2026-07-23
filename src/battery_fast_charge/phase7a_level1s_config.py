"""Phase 7A Level 1S：只比较训练稳定性的冻结配置。"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Level1SArchitecture:
    name: str
    hidden_layer_sizes: tuple[int, ...]


@dataclass(frozen=True)
class Level1SSelectionConfig:
    low_current_threshold_a: float
    rank_metrics: tuple[str, ...]


@dataclass(frozen=True)
class Phase7ALevel1SConfig:
    study_name: str
    source_level1_config: str
    source_level1r_metrics: str
    source_combined_dataset: str
    source_original_dataset: str
    source_terminal_dataset: str
    source_tail_dataset: str
    source_closed_loop_initial_states: str
    architectures: tuple[Level1SArchitecture, ...]
    optimizers: tuple[str, ...]
    selection: Level1SSelectionConfig


def load_phase7a_level1s_config(path: str | Path) -> Phase7ALevel1SConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    architectures = tuple(
        Level1SArchitecture(name=str(item["name"]), hidden_layer_sizes=tuple(int(v) for v in item["hidden_layer_sizes"]))
        for item in raw["training_schemes"]["architectures"]
    )
    selection = dict(raw["selection"])
    selection["rank_metrics"] = tuple(str(v) for v in selection["rank_metrics"])
    config = Phase7ALevel1SConfig(
        study_name=str(raw["study"]["name"]),
        source_level1_config=str(raw["sources"]["level1_config"]),
        source_level1r_metrics=str(raw["sources"]["level1r_metrics"]),
        source_combined_dataset=str(raw["sources"]["combined_dataset"]),
        source_original_dataset=str(raw["sources"]["original_dataset"]),
        source_terminal_dataset=str(raw["sources"]["terminal_dataset"]),
        source_tail_dataset=str(raw["sources"]["tail_dataset"]),
        source_closed_loop_initial_states=str(raw["sources"]["closed_loop_initial_states"]),
        architectures=architectures,
        optimizers=tuple(str(v) for v in raw["training_schemes"]["optimizers"]),
        selection=Level1SSelectionConfig(**selection),
    )
    _validate(config)
    return config


def _validate(config: Phase7ALevel1SConfig) -> None:
    expected = {"deep_32_32_16": (32, 32, 16), "shallow_16": (16,), "shallow_32": (32,)}
    if {v.name: v.hidden_layer_sizes for v in config.architectures} != expected:
        raise ValueError("Level 1S 只允许比较 2-32-32-16-1、2-16-1 和 2-32-1。")
    if config.optimizers != ("adam", "lbfgs"):
        raise ValueError("Level 1S 只允许比较 Adam 和 LBFGS。")
    if config.selection.rank_metrics != ("validation_nrmse", "validation_abs_bias_a", "validation_low_current_abs_bias_a"):
        raise ValueError("Level 1S 必须仅按验证集 NRMSE、总体 bias 和低电流 bias 联合排序。")
