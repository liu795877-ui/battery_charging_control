"""Phase 7C 多温度 DFN 零调参外推验证配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Phase7CContract:
    temperatures_c: tuple[float, ...]
    trajectories_per_temperature: int
    design_start_index: int
    design_seed: int
    initial_soc_bounds: tuple[float, float]
    initial_v1_bounds_v: tuple[float, float]
    initial_v2_bounds_v: tuple[float, float]
    initial_previous_current_bounds_a: tuple[float, float]
    initial_voltage_margin_v: float
    maximum_average_temperature_c: float
    thermal_model: str
    maximum_workers: int
    oscillation_delta_threshold_a: float
    residual_growth_guard_v: float


@dataclass(frozen=True)
class Phase7CGates:
    maximum_voltage_v: float
    maximum_average_temperature_c: float
    maximum_current_nrmse_percent: float
    maximum_mean_charge_time_gap_fraction: float
    minimum_target_reach_fraction: float
    minimum_speedup: float
    numerical_tolerance: float


@dataclass(frozen=True)
class Phase7CConfig:
    study_name: str
    phase7b1b_config: str
    level3_config: str
    model_directory: str
    reference_confirmation_states: str
    frozen_artifacts: dict[str, str]
    contract: Phase7CContract
    gates: Phase7CGates
    data_directory: str
    result_directory: str
    notebook: str


def _pair(values: list[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError("边界必须包含两个数值。")
    return float(values[0]), float(values[1])


def load_phase7c_config(path: str | Path) -> Phase7CConfig:
    payload: dict[str, Any] = yaml.safe_load(
        Path(path).read_text(encoding="utf-8")
    )
    sources = payload["sources"]
    contract = payload["contract"]
    gates = payload["gates"]
    output = payload["output"]
    return Phase7CConfig(
        study_name=str(payload["study"]["name"]),
        phase7b1b_config=str(sources["phase7b1b_config"]),
        level3_config=str(sources["level3_config"]),
        model_directory=str(sources["model_directory"]),
        reference_confirmation_states=str(
            sources["reference_confirmation_states"]
        ),
        frozen_artifacts=dict(sources["frozen_artifacts"]),
        contract=Phase7CContract(
            temperatures_c=tuple(float(x) for x in contract["temperatures_c"]),
            trajectories_per_temperature=int(
                contract["trajectories_per_temperature"]
            ),
            design_start_index=int(contract["design_start_index"]),
            design_seed=int(contract["design_seed"]),
            initial_soc_bounds=_pair(contract["initial_soc_bounds"]),
            initial_v1_bounds_v=_pair(contract["initial_v1_bounds_v"]),
            initial_v2_bounds_v=_pair(contract["initial_v2_bounds_v"]),
            initial_previous_current_bounds_a=_pair(
                contract["initial_previous_current_bounds_a"]
            ),
            initial_voltage_margin_v=float(
                contract["initial_voltage_margin_v"]
            ),
            maximum_average_temperature_c=float(
                contract["maximum_average_temperature_c"]
            ),
            thermal_model=str(contract["thermal_model"]),
            maximum_workers=int(contract["maximum_workers"]),
            oscillation_delta_threshold_a=float(
                contract["oscillation_delta_threshold_a"]
            ),
            residual_growth_guard_v=float(
                contract["residual_growth_guard_v"]
            ),
        ),
        gates=Phase7CGates(**{key: float(value) for key, value in gates.items()}),
        data_directory=str(output["data_directory"]),
        result_directory=str(output["result_directory"]),
        notebook=str(output["notebook"]),
    )
