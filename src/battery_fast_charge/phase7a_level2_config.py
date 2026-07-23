"""Phase 7A Level 2：只增加第二极化状态的独立配置。"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Level2ModelConfig:
    nominal_capacity_ah: float
    sample_period_s: float
    r0_ohm: float
    r1_ohm: float
    tau1_s: float
    r2_ohm: float
    tau2_s: float
    ocv_curve: str


@dataclass(frozen=True)
class Level2DomainConfig:
    trajectory_count: int
    trajectory_steps: int
    train_trajectory_count: int
    validation_trajectory_count: int
    test_trajectory_count: int
    soc_bounds: tuple[float, float]
    v1_bounds_v: tuple[float, float]
    v2_bounds_v: tuple[float, float]
    soc_sampling_power: float


@dataclass(frozen=True)
class Level2DataConfig:
    global_domain: Level2DomainConfig
    terminal_domain: Level2DomainConfig
    initial_voltage_margin_v: float
    random_seed: int
    audit_state_count: int
    warm_starts_per_state: int
    closed_loop_trajectory_count: int
    closed_loop_soc_bounds: tuple[float, float]
    closed_loop_v1_bounds_v: tuple[float, float]
    closed_loop_v2_bounds_v: tuple[float, float]
    maximum_closed_loop_steps: int
    minimum_low_current_label_count: int
    low_current_threshold_a: float


@dataclass(frozen=True)
class Phase7ALevel2Config:
    study_name: str
    source_level1_config: str
    source_identified_parameters: str
    model: Level2ModelConfig
    data: Level2DataConfig


def _pair(value: object) -> tuple[float, float]:
    result = tuple(float(v) for v in value)  # type: ignore[arg-type]
    if len(result) != 2:
        raise ValueError("区间必须含两个端点。")
    return result


def _domain(raw: dict[str, object]) -> Level2DomainConfig:
    values = dict(raw)
    for key in ("soc_bounds", "v1_bounds_v", "v2_bounds_v"):
        values[key] = _pair(values[key])
    return Level2DomainConfig(**values)


def load_phase7a_level2_config(path: str | Path) -> Phase7ALevel2Config:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    data = dict(raw["data"])
    data["global_domain"] = _domain(data["global_domain"])
    data["terminal_domain"] = _domain(data["terminal_domain"])
    for key in ("closed_loop_soc_bounds", "closed_loop_v1_bounds_v", "closed_loop_v2_bounds_v"):
        data[key] = _pair(data[key])
    config = Phase7ALevel2Config(
        study_name=str(raw["study"]["name"]),
        source_level1_config=str(raw["sources"]["level1_config"]),
        source_identified_parameters=str(raw["sources"]["identified_parameters"]),
        model=Level2ModelConfig(**raw["model"]), data=Level2DataConfig(**data),
    )
    _validate(config)
    return config


def _validate(config: Phase7ALevel2Config) -> None:
    global_domain, terminal = config.data.global_domain, config.data.terminal_domain
    if (global_domain.trajectory_count, global_domain.trajectory_steps) != (240, 8):
        raise ValueError("Level 2 全域教师合同必须为 240×8。")
    if (global_domain.train_trajectory_count, global_domain.validation_trajectory_count, global_domain.test_trajectory_count) != (168, 36, 36):
        raise ValueError("Level 2 全域轨迹必须冻结为 168/36/36。")
    if (terminal.trajectory_count, terminal.trajectory_steps) != (160, 24):
        raise ValueError("Level 2 末端教师合同必须为 160×24。")
    if (terminal.train_trajectory_count, terminal.validation_trajectory_count, terminal.test_trajectory_count) != (120, 20, 20):
        raise ValueError("Level 2 末端轨迹必须冻结为 120/20/20。")
    if global_domain.soc_bounds != (0.10, 0.799) or terminal.soc_bounds != (0.74, 0.799):
        raise ValueError("Level 2 必须同时覆盖 SOC 0.10–0.799 全域与 0.74–0.799 末端域。")
    if global_domain.soc_sampling_power != 1.0 or terminal.soc_sampling_power != 0.5:
        raise ValueError("Level 2 全域 SOC 均匀采样，末端域必须向 0.799 显式加密。")
    if (config.data.audit_state_count, config.data.warm_starts_per_state) != (100, 15):
        raise ValueError("Level 2 教师审计必须为 100×15。")
