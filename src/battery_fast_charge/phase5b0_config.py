"""Configuration contract for the Phase 5B-0 MPC feasibility envelope."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ScopeConfig:
    reuse_all_phase5a_reduced_scenarios: bool
    expected_reduced_scenario_count: int
    maximum_charge_time_s: float
    target_soc: float
    change_battery: bool
    run_dfn_anchors: bool
    run_cross_battery: bool


@dataclass(frozen=True)
class ExecutionConfig:
    worker_count: int
    control_block_execution: str
    checkpoint_every_completed_runs: int
    save_all_trajectories: bool


@dataclass(frozen=True)
class ActivityMarginsConfig:
    voltage_v: float
    temperature_c: float
    current_a: float
    current_change_a: float


@dataclass(frozen=True)
class FeasibilityConfig:
    terminal_true_soc_tolerance: float
    minimum_optimizer_success_fraction: float
    maximum_fallback_count: int
    require_physical_safety: bool


@dataclass(frozen=True)
class PhaseFiveBZeroConfig:
    study_name: str
    random_seed: int
    execution_status: str
    source_phase3_config: str
    source_phase5a_config: str
    source_phase5a_summary: str
    information_boundary: dict[str, object]
    scope: ScopeConfig
    execution: ExecutionConfig
    activity_margins: ActivityMarginsConfig
    feasibility: FeasibilityConfig
    required_outputs: tuple[str, ...]
    scenario_classes: tuple[str, ...]


def load_phase_five_b_zero_config(path: str | Path) -> PhaseFiveBZeroConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    study = raw["study"]
    config = PhaseFiveBZeroConfig(
        study_name=str(study["name"]),
        random_seed=int(study["random_seed"]),
        execution_status=str(study["execution_status"]),
        source_phase3_config=str(raw["source_phase3_config"]),
        source_phase5a_config=str(raw["source_phase5a_config"]),
        source_phase5a_summary=str(raw["source_phase5a_summary"]),
        information_boundary=dict(raw["information_boundary"]),
        scope=ScopeConfig(**raw["scope"]),
        execution=ExecutionConfig(**raw["execution"]),
        activity_margins=ActivityMarginsConfig(**raw["activity_margins"]),
        feasibility=FeasibilityConfig(**raw["feasibility"]),
        required_outputs=tuple(str(value) for value in raw["required_outputs"]),
        scenario_classes=tuple(str(value) for value in raw["scenario_classes"]),
    )
    _validate(config)
    return config


def _validate(config: PhaseFiveBZeroConfig) -> None:
    if config.execution_status not in {"planned_not_run", "active", "completed"}:
        raise ValueError("Invalid Phase 5B-0 execution status.")
    if config.scope.expected_reduced_scenario_count != 69:
        raise ValueError("Phase 5B-0 must preserve all 69 frozen Phase 5A scenarios.")
    if config.scope.change_battery or config.scope.run_cross_battery:
        raise ValueError("Phase 5B-0 may not change the battery or run cross-battery tests.")
    if config.scope.run_dfn_anchors:
        raise ValueError("This Phase 5B-0 implementation establishes the reduced-model envelope first.")
    if config.execution.control_block_execution != "phase3_slew_aware_block_schedule":
        raise ValueError("Phase 5B-0 must preserve the Phase 3 MPC execution schedule.")
    if config.execution.worker_count < 1:
        raise ValueError("Phase 5B-0 worker count must be positive.")
    if not 0.0 < config.feasibility.minimum_optimizer_success_fraction <= 1.0:
        raise ValueError("Optimizer success fraction must lie in (0, 1].")
    if config.information_boundary.get("oracle_has_perfect_state") is not False:
        raise ValueError("Oracle MPC may know parameters but must share the same state-estimation errors.")
