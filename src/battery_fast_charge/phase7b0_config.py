"""Phase 7B-0：冻结 Level 3P 控制器的 25 ℃ DFN 跨模型审计配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DFNConfig:
    parameter_set: str
    temperature_c: float
    upper_voltage_cutoff_v: float
    physical_voltage_limit_v: float
    voltage_tolerance_v: float
    target_soc_tolerance: float
    maximum_steps: int


@dataclass(frozen=True)
class DiagnosticConfig:
    early_taper_soc_threshold: float
    early_taper_current_threshold_a: float
    oscillation_delta_threshold_a: float


@dataclass(frozen=True)
class OutputConfig:
    data_directory: str
    result_directory: str


@dataclass(frozen=True)
class Phase7B0Config:
    study_name: str
    source_level3_config: str
    source_level3p_config: str
    source_initial_states: str
    source_model_directory: str
    frozen_artifacts: dict[str, str]
    dfn: DFNConfig
    diagnostics: DiagnosticConfig
    output: OutputConfig


def load_phase7b0_config(path: str | Path) -> Phase7B0Config:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    sources = raw["sources"]
    return Phase7B0Config(
        study_name=str(raw["study"]["name"]),
        source_level3_config=str(sources["level3_config"]),
        source_level3p_config=str(sources["level3p_config"]),
        source_initial_states=str(sources["closed_loop_initial_states"]),
        source_model_directory=str(sources["model_directory"]),
        frozen_artifacts={
            str(key): str(value) for key, value in sources["frozen_artifacts"].items()
        },
        dfn=DFNConfig(**raw["dfn"]),
        diagnostics=DiagnosticConfig(**raw["diagnostics"]),
        output=OutputConfig(**raw["output"]),
    )
