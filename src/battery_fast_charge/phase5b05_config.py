"""Configuration for Phase 5B-0.5 MPC recovery and feasibility recheck."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RepresentativeSelectionConfig:
    teacher_feasible_count: int
    unresolved_count: int
    teacher_and_ann_infeasible_count: int
    include_nominal: bool
    include_hot_extreme: bool
    include_cold_extreme: bool


@dataclass(frozen=True)
class RecoveryConfig:
    maximum_current_a: float
    maximum_current_change_a_per_step: float
    one_step_scan_points: int
    candidate_priority: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionConfig:
    worker_count: int
    maximum_simulation_time_s: float
    checkpoint_every_completed_runs: int
    save_all_trajectories: bool


@dataclass(frozen=True)
class AcceptanceConfig:
    maximum_fallback_slew_violation_count: int
    minimum_matched_nominal_feasible_gain: int
    require_oracle_not_weaker_than_nominal: bool
    require_all_failure_types_auditable: bool
    require_full_69_before_phase5b1: bool


@dataclass(frozen=True)
class PhaseFiveBZeroFiveConfig:
    study_name: str
    random_seed: int
    execution_status: str
    source_phase3_config: str
    source_phase5a_config: str
    source_phase5b0_table: str
    source_phase5b0_runs: str
    ann_model: str
    representative_selection: RepresentativeSelectionConfig
    recovery: RecoveryConfig
    execution: ExecutionConfig
    acceptance: AcceptanceConfig


def load_phase_five_b_zero_five_config(path: str | Path) -> PhaseFiveBZeroFiveConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    study = raw["study"]
    recovery = dict(raw["recovery"])
    recovery["candidate_priority"] = tuple(str(value) for value in recovery["candidate_priority"])
    config = PhaseFiveBZeroFiveConfig(
        study_name=str(study["name"]),
        random_seed=int(study["random_seed"]),
        execution_status=str(study["execution_status"]),
        source_phase3_config=str(raw["source_phase3_config"]),
        source_phase5a_config=str(raw["source_phase5a_config"]),
        source_phase5b0_table=str(raw["source_phase5b0_table"]),
        source_phase5b0_runs=str(raw["source_phase5b0_runs"]),
        ann_model=str(raw["ann_model"]),
        representative_selection=RepresentativeSelectionConfig(**raw["representative_selection"]),
        recovery=RecoveryConfig(**recovery),
        execution=ExecutionConfig(**raw["execution"]),
        acceptance=AcceptanceConfig(**raw["acceptance"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseFiveBZeroFiveConfig) -> None:
    expected = (
        "shifted_previous_feasible",
        "projected_ann_sequence",
        "conservative_slew_down",
        "slope_safe_emergency",
        "hard_safety_emergency",
    )
    if config.recovery.candidate_priority != expected:
        raise ValueError("Phase 5B-0.5 recovery priority must remain frozen.")
    if config.recovery.maximum_current_a != 10.0:
        raise ValueError("Phase 5B-0.5 must preserve the 10 A current limit.")
    if config.recovery.maximum_current_change_a_per_step != 2.0:
        raise ValueError("Phase 5B-0.5 must preserve the 2 A per-step slew limit.")
    if config.execution.worker_count != 1:
        raise ValueError("Use one worker; multi-process SLSQP was slower in Phase 5B-0.")
    if not config.acceptance.require_full_69_before_phase5b1:
        raise ValueError("Representative recovery cannot directly authorize Phase 5B-1.")
