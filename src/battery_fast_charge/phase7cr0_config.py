"""Phase 7C-R0 三项独立诊断配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ThermalFeasibilityConfig:
    ambient_temperature_c: float
    maximum_average_temperature_c: float
    target_soc: float
    initial_soc: float
    initial_polarization_1_v: float
    initial_polarization_2_v: float
    initial_previous_current_a: float
    constant_current_levels_a: tuple[float, ...]
    maximum_steps: int
    oracle_high_currents_a: tuple[float, ...]
    oracle_low_currents_a: tuple[float, ...]
    oracle_switch_temperatures_c: tuple[float, ...]
    oracle_switch_socs: tuple[float, ...]
    validated_oracle_candidates: int
    thermal_guard_c: float


@dataclass(frozen=True)
class SolverAuditConfig:
    failure_temperature_c: float
    failure_trajectory_id: str
    failure_step_index: int
    repeated_runs: int
    multistart_count: int
    multistart_seed: int
    active_constraint_tolerance: float
    finite_difference_step: float


@dataclass(frozen=True)
class ReversalAuditConfig:
    temperature_c: float
    trajectory_id: str
    step_indices: tuple[int, ...]
    multistart_count: int
    multistart_seed: int


@dataclass(frozen=True)
class Phase7CR0Config:
    study_name: str
    phase7c_config: str
    phase7b1b_config: str
    phase7c_trajectories: str
    phase7c_initial_states_15c: str
    phase7c_initial_states_30c: str
    frozen_artifacts: dict[str, str]
    thermal: ThermalFeasibilityConfig
    solver: SolverAuditConfig
    reversal: ReversalAuditConfig
    data_directory: str
    result_directory: str


def load_phase7cr0_config(path: str | Path) -> Phase7CR0Config:
    payload: dict[str, Any] = yaml.safe_load(
        Path(path).read_text(encoding="utf-8")
    )
    sources = payload["sources"]
    thermal = payload["thermal_feasibility"]
    solver = payload["solver_audit"]
    reversal = payload["reversal_audit"]
    output = payload["output"]
    return Phase7CR0Config(
        study_name=str(payload["study"]["name"]),
        phase7c_config=str(sources["phase7c_config"]),
        phase7b1b_config=str(sources["phase7b1b_config"]),
        phase7c_trajectories=str(sources["phase7c_trajectories"]),
        phase7c_initial_states_15c=str(
            sources["phase7c_initial_states_15c"]
        ),
        phase7c_initial_states_30c=str(
            sources["phase7c_initial_states_30c"]
        ),
        frozen_artifacts=dict(sources["frozen_artifacts"]),
        thermal=ThermalFeasibilityConfig(
            **{
                **thermal,
                "constant_current_levels_a": tuple(
                    thermal["constant_current_levels_a"]
                ),
                "oracle_high_currents_a": tuple(
                    thermal["oracle_high_currents_a"]
                ),
                "oracle_low_currents_a": tuple(
                    thermal["oracle_low_currents_a"]
                ),
                "oracle_switch_temperatures_c": tuple(
                    thermal["oracle_switch_temperatures_c"]
                ),
                "oracle_switch_socs": tuple(
                    thermal["oracle_switch_socs"]
                ),
            }
        ),
        solver=SolverAuditConfig(**solver),
        reversal=ReversalAuditConfig(
            **{
                **reversal,
                "step_indices": tuple(reversal["step_indices"]),
            }
        ),
        data_directory=str(output["data_directory"]),
        result_directory=str(output["result_directory"]),
    )
