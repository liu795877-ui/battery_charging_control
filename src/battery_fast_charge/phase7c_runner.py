"""Phase 7C：冻结控制器在 15/30 ℃ Chen2020 DFN 上的零调参审计。"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm

from .phase7a_level1_runner import _regression_metrics
from .phase7a_level3_model import Level3MPC, Level3State
from .phase7a_level3p_runner import project_current
from .phase7b1b_config import load_phase7b1b_config
from .phase7b1b_runner import (
    _load_context,
    _maximum_safe_current,
    _predicted_next_voltage,
)
from .phase7c_config import Phase7CConfig


def _sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".py", ".yaml", ".yml"}:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def verify_frozen_artifacts(
    config: Phase7CConfig, root: Path
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    mismatches = []
    for relative, expected in config.frozen_artifacts.items():
        actual = _sha256(root / relative)
        matched = actual == expected
        records[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": matched,
        }
        if not matched:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"Phase 7C 冻结工件哈希不匹配：{mismatches}")
    phase7b1b = load_phase7b1b_config(root / config.phase7b1b_config)
    if not np.isclose(
        config.contract.residual_growth_guard_v,
        phase7b1b.safety.residual_growth_guard_v,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise RuntimeError("Phase 7C 电压残差增长裕量偏离 25 ℃ 冻结值。")
    return records


def _van_der_corput(index: int, base: int) -> float:
    result, denominator = 0.0, 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        result += remainder / denominator
    return result


def freeze_initial_states(config: Phase7CConfig, root: Path) -> dict[str, Any]:
    data_dir = root / config.data_directory
    data_dir.mkdir(parents=True, exist_ok=True)
    phase7b1b = load_phase7b1b_config(root / config.phase7b1b_config)
    level3, _, model, _, _ = _load_context(phase7b1b, root)
    reference = pd.read_csv(root / config.reference_confirmation_states)
    reference_values = reference[
        [
            "initial_soc",
            "initial_polarization_1_v",
            "initial_polarization_2_v",
            "initial_previous_current_a",
        ]
    ].to_numpy(float)
    bounds = config.contract
    records: list[dict[str, Any]] = []
    candidate = bounds.design_start_index
    voltage_limit = (
        model.inherited.mpc.terminal_voltage_max_v
        - bounds.initial_voltage_margin_v
    )
    while len(records) < bounds.trajectories_per_temperature:
        soc = bounds.initial_soc_bounds[0] + np.ptp(
            bounds.initial_soc_bounds
        ) * _van_der_corput(candidate, 2)
        v1 = bounds.initial_v1_bounds_v[0] + np.ptp(
            bounds.initial_v1_bounds_v
        ) * _van_der_corput(candidate, 3)
        v2 = bounds.initial_v2_bounds_v[0] + np.ptp(
            bounds.initial_v2_bounds_v
        ) * _van_der_corput(candidate, 5)
        previous = bounds.initial_previous_current_bounds_a[0] + np.ptp(
            bounds.initial_previous_current_bounds_a
        ) * _van_der_corput(candidate, 7)
        used_index = candidate
        candidate += 1
        state = Level3State(soc, v1, v2, previous)
        minimum_current = max(
            0.0, previous - level3.constraint.maximum_current_step_a
        )
        if model.terminal_voltage(state, minimum_current) > voltage_limit:
            continue
        feasible = Level3MPC(model).solve(state)
        if not (feasible.optimizer_success and feasible.prediction_feasible):
            continue
        values = np.asarray([soc, v1, v2, previous])
        if np.any(
            np.all(np.isclose(values, reference_values, atol=1.0e-14), axis=1)
        ):
            continue
        records.append(
            {
                "initial_soc": soc,
                "initial_polarization_1_v": v1,
                "initial_polarization_2_v": v2,
                "initial_previous_current_a": previous,
                "design_candidate_index": used_index,
                "design_seed": bounds.design_seed,
            }
        )
    base = pd.DataFrame(records)
    files: dict[str, Any] = {}
    for temperature_c in bounds.temperatures_c:
        token = f"{int(temperature_c)}c"
        frame = base.copy()
        frame.insert(
            0,
            "trajectory_id",
            [f"phase7c_{token}_{i:03d}" for i in range(len(frame))],
        )
        frame.insert(1, "ambient_temperature_c", temperature_c)
        frame.insert(2, "initial_temperature_c", temperature_c)
        path = data_dir / f"initial_states_{token}.csv"
        frame.to_csv(path, index=False)
        files[str(path.relative_to(root))] = {
            "sha256": _sha256(path),
            "trajectory_count": len(frame),
            "temperature_c": temperature_c,
        }
    freeze = {
        "study_name": config.study_name,
        "frozen_before_closed_loop_execution": True,
        "not_teacher_data": True,
        "design_seed": bounds.design_seed,
        "design_start_index": bounds.design_start_index,
        "files": files,
        "source_artifact_verification": verify_frozen_artifacts(config, root),
        "configuration": asdict(config),
    }
    freeze_path = data_dir / "freeze_contract.json"
    freeze_path.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value["sha256"] for key, value in files.items()},
            ensure_ascii=False,
        )
    )
    return freeze


class Chen2020ThermalDFN:
    """环境温度与初温一致、lumped 热模型的 Chen2020 DFN。"""

    def __init__(
        self,
        parameter_set: str,
        temperature_c: float,
        upper_voltage_cutoff_v: float,
        initial_soc: float,
        sample_period_s: float,
        thermal_model: str,
    ) -> None:
        os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
        pybamm.set_logging_level("ERROR")
        model = pybamm.lithium_ion.DFN(options={"thermal": thermal_model})
        parameters = pybamm.ParameterValues(parameter_set)
        parameters.update(
            {
                "Ambient temperature [K]": temperature_c + 273.15,
                "Initial temperature [K]": temperature_c + 273.15,
                "Upper voltage cut-off [V]": upper_voltage_cutoff_v,
                "Current function [A]": pybamm.InputParameter(
                    "phase7c_applied_current_a"
                ),
            }
        )
        self.initial_soc = float(initial_soc)
        self.nominal_capacity_ah = float(
            parameters["Nominal cell capacity [A.h]"]
        )
        self.sample_period_s = float(sample_period_s)
        self.simulation = pybamm.Simulation(
            model,
            parameter_values=parameters,
            solver=pybamm.CasadiSolver(mode="safe"),
        )
        self.simulation.build(initial_soc=self.initial_soc)

    def step(self, charge_current_a: float) -> dict[str, float]:
        solution = self.simulation.step(
            self.sample_period_s,
            inputs={"phase7c_applied_current_a": -float(charge_current_a)},
            save=False,
        )

        def last(name: str) -> float:
            return float(np.asarray(solution[name].entries).reshape(-1)[-1])

        discharge_capacity_ah = last("Discharge capacity [A.h]")
        return {
            "time_s": last("Time [s]"),
            "soc": self.initial_soc
            - discharge_capacity_ah / self.nominal_capacity_ah,
            "terminal_voltage_v": last("Terminal voltage [V]"),
            "average_temperature_c": (
                last("Volume-averaged cell temperature [K]") - 273.15
            ),
        }


def _run_rollout(
    config: Phase7CConfig,
    root: Path,
    controller_kind: str,
    seed: int | None,
    initial: dict[str, Any],
) -> pd.DataFrame:
    phase7b1b = load_phase7b1b_config(root / config.phase7b1b_config)
    level3, inherited, model, networks, phase7b0 = _load_context(
        phase7b1b, root
    )
    state = Level3State(
        float(initial["initial_soc"]),
        float(initial["initial_polarization_1_v"]),
        float(initial["initial_polarization_2_v"]),
        float(initial["initial_previous_current_a"]),
    )
    temperature_c = float(initial["ambient_temperature_c"])
    plant = Chen2020ThermalDFN(
        phase7b0.dfn.parameter_set,
        temperature_c,
        phase7b0.dfn.upper_voltage_cutoff_v,
        state.soc,
        level3.model.sample_period_s,
        config.contract.thermal_model,
    )
    mpc = Level3MPC(model) if controller_kind == "mpc" else None
    network = networks[seed] if seed is not None else None
    lower_bound, upper_bound = inherited.mpc.current_bounds_a
    maximum_step = level3.constraint.maximum_current_step_a
    target = inherited.mpc.target_soc - phase7b0.dfn.target_soc_tolerance
    measured_residual_v = 0.0
    last_voltage_v = float("nan")
    rows: list[dict[str, Any]] = []
    for step in range(phase7b0.dfn.maximum_steps):
        base_started = perf_counter()
        if controller_kind == "mpc":
            assert mpc is not None
            result = mpc.solve(state)
            raw_current = float(result.current_a)
            projected_current = raw_current
            optimizer_success = bool(result.optimizer_success)
            prediction_feasible = bool(result.prediction_feasible)
        else:
            assert network is not None
            raw_current = float(
                network.predict(
                    np.asarray(
                        [
                            state.soc,
                            state.polarization_1_v,
                            state.polarization_2_v,
                            state.previous_current_a,
                        ]
                    )
                )
            )
            projected_current, _, _ = project_current(
                raw_current,
                state.previous_current_a,
                lower_bound,
                upper_bound,
                maximum_step,
            )
            optimizer_success = True
            prediction_feasible = True
        base_time_s = perf_counter() - base_started
        slew_lower = max(
            lower_bound, state.previous_current_a - maximum_step
        )
        slew_upper = min(
            upper_bound, state.previous_current_a + maximum_step
        )
        safety_started = perf_counter()
        correction_v = (
            measured_residual_v + config.contract.residual_growth_guard_v
        )
        voltage_safe_max = _maximum_safe_current(
            state, correction_v, phase7b1b, model
        )
        empty = (
            voltage_safe_max
            < slew_lower - phase7b1b.safety.empty_interval_tolerance_a
        )
        safety_time_s = perf_counter() - safety_started
        common = {
            "temperature_c": temperature_c,
            "controller_kind": controller_kind,
            "seed": -1 if seed is None else seed,
            "trajectory_id": str(initial["trajectory_id"]),
            "step_index": step,
            "time_s": (step + 1) * level3.model.sample_period_s,
            "soc": state.soc,
            "previous_current_a": state.previous_current_a,
            "raw_current_a": raw_current,
            "projected_current_a": projected_current,
            "slew_lower_a": slew_lower,
            "slew_upper_a": slew_upper,
            "voltage_safe_current_max_a": voltage_safe_max,
            "measured_residual_before_v": measured_residual_v,
            "voltage_correction_v": correction_v,
            "base_decision_time_s": base_time_s,
            "safety_layer_time_s": safety_time_s,
            "total_decision_time_s": base_time_s + safety_time_s,
            "optimizer_success": optimizer_success,
            "prediction_feasible": prediction_feasible,
        }
        if empty:
            rows.append(
                {
                    **common,
                    "next_soc": state.soc,
                    "current_a": np.nan,
                    "current_step_a": np.nan,
                    "terminal_voltage_v": last_voltage_v,
                    "average_temperature_c": temperature_c,
                    "predicted_next_voltage_2rc_v": np.nan,
                    "measured_residual_after_v": np.nan,
                    "residual_positive_growth_v": np.nan,
                    "voltage_slew_feasibility_margin_a": (
                        voltage_safe_max - slew_lower
                    ),
                    "empty_voltage_slew_interval": True,
                    "input_projection_intervened": abs(
                        projected_current - raw_current
                    )
                    > phase7b1b.safety.intervention_tolerance_a,
                    "voltage_safety_intervened": False,
                    "voltage_safety_current_correction_a": np.nan,
                }
            )
            break
        safe_current = float(
            np.clip(
                min(projected_current, voltage_safe_max),
                slew_lower,
                slew_upper,
            )
        )
        predicted = model.step(state, safe_current)
        predicted_voltage = model.terminal_voltage(predicted, safe_current)
        measurement = plant.step(safe_current)
        residual_after = (
            float(measurement["terminal_voltage_v"]) - predicted_voltage
        )
        rows.append(
            {
                **common,
                "next_soc": measurement["soc"],
                "current_a": safe_current,
                "current_step_a": abs(
                    safe_current - state.previous_current_a
                ),
                "terminal_voltage_v": measurement["terminal_voltage_v"],
                "average_temperature_c": measurement[
                    "average_temperature_c"
                ],
                "predicted_next_voltage_2rc_v": predicted_voltage,
                "measured_residual_after_v": residual_after,
                "residual_positive_growth_v": max(
                    residual_after - measured_residual_v, 0.0
                ),
                "voltage_slew_feasibility_margin_a": (
                    voltage_safe_max - slew_lower
                ),
                "empty_voltage_slew_interval": False,
                "input_projection_intervened": abs(
                    projected_current - raw_current
                )
                > phase7b1b.safety.intervention_tolerance_a,
                "voltage_safety_intervened": abs(
                    safe_current - projected_current
                )
                > phase7b1b.safety.intervention_tolerance_a,
                "voltage_safety_current_correction_a": (
                    safe_current - projected_current
                ),
            }
        )
        measured_residual_v = residual_after
        last_voltage_v = float(measurement["terminal_voltage_v"])
        state = Level3State(
            float(measurement["soc"]),
            predicted.polarization_1_v,
            predicted.polarization_2_v,
            safe_current,
        )
        if state.soc >= target:
            break
    return pd.DataFrame(rows)


def _worker(
    config: Phase7CConfig,
    root_text: str,
    job: dict[str, Any],
    path_text: str,
) -> str:
    path = Path(path_text)
    frame = _run_rollout(
        config,
        Path(root_text),
        job["controller_kind"],
        job["seed"],
        job["initial"],
    )
    frame.to_csv(path, index=False)
    return path_text


def _cache_name(job: dict[str, Any]) -> str:
    seed = "baseline" if job["seed"] is None else str(job["seed"])
    return (
        f"{job['controller_kind']}_{seed}_"
        f"{job['initial']['trajectory_id']}.csv"
    )


def _run_jobs(
    config: Phase7CConfig,
    root: Path,
    jobs: list[dict[str, Any]],
    resume: bool,
) -> pd.DataFrame:
    run_dir = root / config.data_directory / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    pending: list[tuple[dict[str, Any], Path]] = []
    for job in jobs:
        path = run_dir / _cache_name(job)
        paths.append(path)
        if not (resume and path.exists()):
            pending.append((job, path))
    if pending:
        print(
            f"[Phase 7C] running {len(pending)} trajectories with "
            f"{config.contract.maximum_workers} workers",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=config.contract.maximum_workers
        ) as executor:
            futures = {
                executor.submit(
                    _worker, config, str(root), job, str(path)
                ): path
                for job, path in pending
            }
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed == 1 or completed % 10 == 0:
                    print(
                        f"[Phase 7C] completed {completed}/{len(pending)}",
                        flush=True,
                    )
    return pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )


def _load_initial_states(
    config: Phase7CConfig, root: Path
) -> pd.DataFrame:
    frames = []
    for temperature_c in config.contract.temperatures_c:
        token = f"{int(temperature_c)}c"
        frames.append(
            pd.read_csv(
                root / config.data_directory / f"initial_states_{token}.csv"
            )
        )
    return pd.concat(frames, ignore_index=True)


def _jobs(
    initial: pd.DataFrame,
    controller_kind: str,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    records = initial.to_dict(orient="records")
    if controller_kind == "mpc":
        return [
            {"controller_kind": "mpc", "seed": None, "initial": row}
            for row in records
        ]
    return [
        {"controller_kind": "ann", "seed": seed, "initial": row}
        for seed in seeds
        for row in records
    ]


def _direction_reversals(values: np.ndarray, threshold: float) -> int:
    differences = np.diff(values)
    signs = np.sign(differences[np.abs(differences) >= threshold])
    return int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0


def _continuous_crossing_time(
    group: pd.DataFrame, target: float, sample_period_s: float
) -> float:
    for row in group.sort_values("step_index").itertuples():
        if row.soc >= target:
            return row.step_index * sample_period_s
        if row.next_soc >= target and row.next_soc > row.soc:
            fraction = (target - row.soc) / (row.next_soc - row.soc)
            return (row.step_index + fraction) * sample_period_s
    return float("nan")


def _controller_summary(
    frame: pd.DataFrame,
    target: float,
    maximum_step_a: float,
    config: Phase7CConfig,
) -> dict[str, Any]:
    valid = frame[~frame.current_a.isna()]
    trajectories = list(valid.groupby(["seed", "trajectory_id"]))
    reached = [
        float(group.next_soc.iloc[-1]) >= target for _, group in trajectories
    ]
    interventions = valid[valid.voltage_safety_intervened.astype(bool)]
    return {
        "trajectory_count": len(trajectories),
        "maximum_voltage_v": float(valid.terminal_voltage_v.max()),
        "maximum_average_temperature_c": float(
            valid.average_temperature_c.max()
        ),
        "maximum_current_violation_a": float(
            np.maximum(
                np.maximum(valid.current_a - 10.0, -valid.current_a), 0.0
            ).max()
        ),
        "maximum_slew_violation_a": float(
            np.maximum(valid.current_step_a - maximum_step_a, 0.0).max()
        ),
        "maximum_current_step_a": float(valid.current_step_a.max()),
        "empty_interval_count": int(
            frame.empty_voltage_slew_interval.astype(bool).sum()
        ),
        "target_reach_fraction": float(np.mean(reached)),
        "optimizer_failure_count": int(
            (~frame.optimizer_success.astype(bool)).sum()
        ),
        "prediction_infeasible_count": int(
            (~frame.prediction_feasible.astype(bool)).sum()
        ),
        "maximum_direction_reversals": max(
            (
                _direction_reversals(
                    group.current_a.to_numpy(float),
                    config.contract.oscillation_delta_threshold_a,
                )
                for _, group in trajectories
            ),
            default=0,
        ),
        "maximum_voltage_residual_v": float(
            valid.measured_residual_after_v.max()
        ),
        "p95_voltage_residual_v": float(
            valid.measured_residual_after_v.quantile(0.95)
        ),
        "p99_voltage_residual_v": float(
            valid.measured_residual_after_v.quantile(0.99)
        ),
        "maximum_positive_residual_growth_v": float(
            valid.residual_positive_growth_v.max()
        ),
        "minimum_voltage_slew_feasibility_margin_a": float(
            frame.voltage_slew_feasibility_margin_a.min()
        ),
        "voltage_intervention_fraction": float(
            valid.voltage_safety_intervened.astype(bool).mean()
        ),
        "maximum_current_correction_a": float(
            valid.voltage_safety_current_correction_a.abs().max()
        ),
        "mean_active_current_correction_a": float(
            interventions.voltage_safety_current_correction_a.abs().mean()
        )
        if len(interventions)
        else 0.0,
        "first_intervention_soc": float(interventions.soc.min())
        if len(interventions)
        else float("nan"),
        "mean_base_decision_time_ms": float(
            1000.0 * valid.base_decision_time_s.mean()
        ),
        "mean_safety_layer_time_ms": float(
            1000.0 * valid.safety_layer_time_s.mean()
        ),
        "mean_total_decision_time_ms": float(
            1000.0 * valid.total_decision_time_s.mean()
        ),
    }


def _temperature_metrics(
    config: Phase7CConfig,
    root: Path,
    temperature_c: float,
    mpc: pd.DataFrame,
    ann: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    phase7b1b = load_phase7b1b_config(root / config.phase7b1b_config)
    level3, inherited, _, _, phase7b0 = _load_context(phase7b1b, root)
    target = inherited.mpc.target_soc - phase7b0.dfn.target_soc_tolerance
    mpc_t = mpc[mpc.temperature_c == temperature_c]
    mpc_summary = _controller_summary(
        mpc_t,
        target,
        level3.constraint.maximum_current_step_a,
        config,
    )
    if ann is None:
        checks = {
            "maximum_voltage": (
                mpc_summary["maximum_voltage_v"]
                <= config.gates.maximum_voltage_v
            ),
            "maximum_average_temperature": (
                mpc_summary["maximum_average_temperature_c"]
                <= config.gates.maximum_average_temperature_c
            ),
            "zero_current_violation": (
                mpc_summary["maximum_current_violation_a"]
                <= config.gates.numerical_tolerance
            ),
            "zero_slew_violation": (
                mpc_summary["maximum_slew_violation_a"]
                <= config.gates.numerical_tolerance
            ),
            "zero_empty_interval": mpc_summary["empty_interval_count"] == 0,
            "target_reach_100_percent": (
                mpc_summary["target_reach_fraction"]
                >= config.gates.minimum_target_reach_fraction
            ),
            "zero_oscillation": (
                mpc_summary["maximum_direction_reversals"] == 0
            ),
            "zero_solver_failure": (
                mpc_summary["optimizer_failure_count"] == 0
                and mpc_summary["prediction_infeasible_count"] == 0
            ),
        }
        return pd.DataFrame(), {
            "temperature_c": temperature_c,
            "mpc": mpc_summary,
            "checks": checks,
            "success": bool(all(checks.values())),
        }
    ann_t = ann[ann.temperature_c == temperature_c]
    seed_rows = []
    mpc_groups = {
        trajectory_id: group
        for trajectory_id, group in mpc_t.groupby("trajectory_id")
    }
    for seed in inherited.network.initialization_seeds:
        seed_frame = ann_t[ann_t.seed == seed]
        nrmse_values = []
        time_gaps = []
        discrete_time_differences = []
        continuous_time_differences = []
        charge_differences = []
        for trajectory_id, ann_group in seed_frame.groupby("trajectory_id"):
            mpc_group = mpc_groups[trajectory_id]
            valid_ann = ann_group[~ann_group.current_a.isna()]
            valid_mpc = mpc_group[~mpc_group.current_a.isna()]
            paired = valid_mpc[["step_index", "current_a"]].merge(
                valid_ann[["step_index", "current_a"]],
                on="step_index",
                suffixes=("_mpc", "_ann"),
            )
            nrmse_values.append(
                _regression_metrics(
                    paired.current_a_mpc.to_numpy(float),
                    paired.current_a_ann.to_numpy(float),
                )["nrmse"]
            )
            time_gaps.append(
                abs(len(valid_ann) - len(valid_mpc)) / len(valid_mpc)
            )
            dt = level3.model.sample_period_s
            discrete_time_differences.append(
                (len(valid_ann) - len(valid_mpc)) * dt
            )
            continuous_time_differences.append(
                _continuous_crossing_time(valid_ann, target, dt)
                - _continuous_crossing_time(valid_mpc, target, dt)
            )
            charge_differences.append(
                (
                    valid_ann.current_a.sum()
                    - valid_mpc.current_a.sum()
                )
                * dt
                / 3600.0
            )
        summary = _controller_summary(
            seed_frame,
            target,
            level3.constraint.maximum_current_step_a,
            config,
        )
        seed_rows.append(
            {
                "temperature_c": temperature_c,
                "seed": seed,
                "mean_current_nrmse": float(np.mean(nrmse_values)),
                "maximum_current_nrmse": float(np.max(nrmse_values)),
                "mean_charge_time_gap_fraction": float(np.mean(time_gaps)),
                "mean_discrete_arrival_time_difference_s": float(
                    np.mean(discrete_time_differences)
                ),
                "mean_continuous_arrival_time_difference_s": float(
                    np.mean(continuous_time_differences)
                ),
                "mean_cumulative_charge_difference_ah": float(
                    np.mean(charge_differences)
                ),
                "maximum_voltage_v": summary["maximum_voltage_v"],
                "maximum_average_temperature_c": summary[
                    "maximum_average_temperature_c"
                ],
                "maximum_current_violation_a": summary[
                    "maximum_current_violation_a"
                ],
                "maximum_slew_violation_a": summary[
                    "maximum_slew_violation_a"
                ],
                "empty_interval_count": summary["empty_interval_count"],
                "target_reach_fraction": summary["target_reach_fraction"],
                "maximum_direction_reversals": summary[
                    "maximum_direction_reversals"
                ],
                "voltage_intervention_fraction": summary[
                    "voltage_intervention_fraction"
                ],
                "maximum_current_correction_a": summary[
                    "maximum_current_correction_a"
                ],
                "mean_active_current_correction_a": summary[
                    "mean_active_current_correction_a"
                ],
                "first_intervention_soc": summary["first_intervention_soc"],
                "maximum_voltage_residual_v": summary[
                    "maximum_voltage_residual_v"
                ],
                "p95_voltage_residual_v": summary[
                    "p95_voltage_residual_v"
                ],
                "p99_voltage_residual_v": summary[
                    "p99_voltage_residual_v"
                ],
                "maximum_positive_residual_growth_v": summary[
                    "maximum_positive_residual_growth_v"
                ],
                "minimum_voltage_slew_feasibility_margin_a": summary[
                    "minimum_voltage_slew_feasibility_margin_a"
                ],
                "mean_base_decision_time_ms": summary[
                    "mean_base_decision_time_ms"
                ],
                "mean_safety_layer_time_ms": summary[
                    "mean_safety_layer_time_ms"
                ],
                "speedup": (
                    mpc_summary["mean_total_decision_time_ms"]
                    / summary["mean_total_decision_time_ms"]
                ),
            }
        )
    seeds = pd.DataFrame(seed_rows)
    checks = {
        "mpc_voltage_safe": (
            mpc_summary["maximum_voltage_v"]
            <= config.gates.maximum_voltage_v
        ),
        "ann_voltage_safe": bool(
            (seeds.maximum_voltage_v <= config.gates.maximum_voltage_v).all()
        ),
        "mpc_temperature_safe": (
            mpc_summary["maximum_average_temperature_c"]
            <= config.gates.maximum_average_temperature_c
        ),
        "ann_temperature_safe": bool(
            (
                seeds.maximum_average_temperature_c
                <= config.gates.maximum_average_temperature_c
            ).all()
        ),
        "zero_current_violation": bool(
            (
                seeds.maximum_current_violation_a
                <= config.gates.numerical_tolerance
            ).all()
            and mpc_summary["maximum_current_violation_a"]
            <= config.gates.numerical_tolerance
        ),
        "zero_slew_violation": bool(
            (
                seeds.maximum_slew_violation_a
                <= config.gates.numerical_tolerance
            ).all()
            and mpc_summary["maximum_slew_violation_a"]
            <= config.gates.numerical_tolerance
        ),
        "zero_empty_interval": bool(
            (seeds.empty_interval_count == 0).all()
            and mpc_summary["empty_interval_count"] == 0
        ),
        "current_nrmse_below_1_percent": bool(
            (
                seeds.mean_current_nrmse
                < config.gates.maximum_current_nrmse_percent
            ).all()
        ),
        "charge_time_gap_below_2_percent": bool(
            (
                seeds.mean_charge_time_gap_fraction
                < config.gates.maximum_mean_charge_time_gap_fraction
            ).all()
        ),
        "target_reach_100_percent": bool(
            (
                seeds.target_reach_fraction
                >= config.gates.minimum_target_reach_fraction
            ).all()
            and mpc_summary["target_reach_fraction"]
            >= config.gates.minimum_target_reach_fraction
        ),
        "speedup_above_100": bool(
            (seeds.speedup > config.gates.minimum_speedup).all()
        ),
        "zero_oscillation": bool(
            (seeds.maximum_direction_reversals == 0).all()
            and mpc_summary["maximum_direction_reversals"] == 0
        ),
    }
    return seeds, {
        "temperature_c": temperature_c,
        "mpc": mpc_summary,
        "ann": {
            "current_nrmse_range": [
                float(seeds.mean_current_nrmse.min()),
                float(seeds.mean_current_nrmse.max()),
            ],
            "maximum_mean_charge_time_gap_fraction": float(
                seeds.mean_charge_time_gap_fraction.max()
            ),
            "maximum_voltage_v": float(seeds.maximum_voltage_v.max()),
            "maximum_average_temperature_c": float(
                seeds.maximum_average_temperature_c.max()
            ),
            "minimum_speedup": float(seeds.speedup.min()),
            "minimum_target_reach_fraction": float(
                seeds.target_reach_fraction.min()
            ),
            "intervention_fraction_range": [
                float(seeds.voltage_intervention_fraction.min()),
                float(seeds.voltage_intervention_fraction.max()),
            ],
            "maximum_current_correction_a": float(
                seeds.maximum_current_correction_a.max()
            ),
            "maximum_voltage_residual_v": float(
                seeds.maximum_voltage_residual_v.max()
            ),
            "maximum_positive_residual_growth_v": float(
                seeds.maximum_positive_residual_growth_v.max()
            ),
            "minimum_voltage_slew_feasibility_margin_a": float(
                seeds.minimum_voltage_slew_feasibility_margin_a.min()
            ),
        },
        "checks": checks,
        "success": bool(all(checks.values())),
    }


def _trajectory_diagnostics(
    frame: pd.DataFrame, target: float, sample_period_s: float
) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(
        ["temperature_c", "controller_kind", "seed", "trajectory_id"]
    ):
        valid = group[~group.current_a.isna()]
        interventions = valid[valid.voltage_safety_intervened.astype(bool)]
        rows.append(
            {
                "temperature_c": key[0],
                "controller_kind": key[1],
                "seed": key[2],
                "trajectory_id": key[3],
                "maximum_voltage_v": valid.terminal_voltage_v.max(),
                "maximum_average_temperature_c": (
                    valid.average_temperature_c.max()
                ),
                "maximum_voltage_residual_v": (
                    valid.measured_residual_after_v.max()
                ),
                "p95_voltage_residual_v": (
                    valid.measured_residual_after_v.quantile(0.95)
                ),
                "p99_voltage_residual_v": (
                    valid.measured_residual_after_v.quantile(0.99)
                ),
                "maximum_positive_residual_growth_v": (
                    valid.residual_positive_growth_v.max()
                ),
                "voltage_intervention_fraction": (
                    valid.voltage_safety_intervened.astype(bool).mean()
                ),
                "maximum_current_correction_a": (
                    valid.voltage_safety_current_correction_a.abs().max()
                ),
                "mean_active_current_correction_a": (
                    interventions.voltage_safety_current_correction_a.abs().mean()
                    if len(interventions)
                    else 0.0
                ),
                "first_intervention_soc": (
                    interventions.soc.min() if len(interventions) else np.nan
                ),
                "minimum_voltage_slew_feasibility_margin_a": (
                    group.voltage_slew_feasibility_margin_a.min()
                ),
                "empty_interval_count": (
                    group.empty_voltage_slew_interval.astype(bool).sum()
                ),
                "target_reached": (
                    len(valid) > 0 and valid.next_soc.iloc[-1] >= target
                ),
                "discrete_arrival_time_s": (
                    len(valid) * sample_period_s
                ),
                "continuous_arrival_time_s": _continuous_crossing_time(
                    valid, target, sample_period_s
                ),
                "cumulative_charge_ah": (
                    valid.current_a.sum() * sample_period_s / 3600.0
                ),
                "direction_reversals": _direction_reversals(
                    valid.current_a.to_numpy(float), 0.25
                ),
                "mean_base_decision_time_ms": (
                    1000.0 * valid.base_decision_time_s.mean()
                ),
                "mean_safety_layer_time_ms": (
                    1000.0 * valid.safety_layer_time_s.mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot(
    output: Path, frame: pd.DataFrame, temperature_results: dict[str, Any]
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2, 2, figsize=(13, 8), layout="constrained"
    )
    colors = {15.0: "#2878B5", 30.0: "#D95319"}
    for temperature_c in sorted(frame.temperature_c.unique()):
        subset = frame[
            (frame.temperature_c == temperature_c)
            & (frame.controller_kind == "ann")
            & (frame.seed == 22)
        ]
        if subset.empty:
            subset = frame[
                (frame.temperature_c == temperature_c)
                & (frame.controller_kind == "mpc")
            ]
        example_id = (
            subset.groupby("trajectory_id").average_temperature_c.max().idxmax()
        )
        example = subset[subset.trajectory_id == example_id]
        label = f"{temperature_c:.0f} ℃ worst thermal"
        axes[0, 0].plot(
            example.time_s,
            example.terminal_voltage_v,
            color=colors[temperature_c],
            label=label,
        )
        axes[0, 1].plot(
            example.time_s,
            example.average_temperature_c,
            color=colors[temperature_c],
            label=label,
        )
        axes[1, 0].plot(
            example.time_s,
            example.current_a,
            color=colors[temperature_c],
            label=label,
        )
        axes[1, 1].plot(
            example.time_s,
            example.next_soc,
            color=colors[temperature_c],
            label=label,
        )
    axes[0, 0].axhline(4.2, color="red", linestyle="--")
    axes[0, 1].axhline(35.0, color="red", linestyle="--")
    axes[0, 0].set(ylabel="Terminal voltage [V]")
    axes[0, 1].set(ylabel="Average temperature [℃]")
    axes[1, 0].set(ylabel="Charge current [A]")
    axes[1, 1].set(ylabel="SOC")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend()
    result = "PASS" if all(
        item.get("success", False) for item in temperature_results.values()
    ) else "STOP"
    figure.supxlabel("Time [s]")
    figure.suptitle(f"Phase 7C frozen-controller validation: {result}")
    figure.savefig(output / "phase7c_multitemperature_validation.png", dpi=180)
    plt.close(figure)


def _write_report(
    path: Path,
    payload: dict[str, Any],
) -> None:
    nominal = payload["nominal_25c_reference"]
    lines = [
        "# Phase 7C：冻结控制器的多温度 DFN 外推验证",
        "",
        "## 实验合同",
        "",
        "五个 ANN、MPC、2RC、Level 3P 投影、5 s 控制周期和 "
        "11.3055225 mV 电压残差增长裕量全部冻结。15 ℃与30 ℃各24个"
        "初态在闭环运行前冻结，未加入教师数据，也未重新训练 ANN。",
        "",
        "## 分温度严格结果",
        "",
        "| 温度 | 验证域 | 安全 MPC/架构 | 安全 ANN | 最高电压 | 最高平均温度 | 结论 |",
        "|---:|---|---:|---:|---:|---:|---|",
        (
            f"| 25 ℃ | Phase 7B-1独立确认 | 通过 | 通过 | "
            f"{nominal['maximum_ann_voltage_v']:.6f} V | 等温模型，不适用 | "
            "严格通过 |"
        ),
    ]
    for token, result in payload["temperature_results"].items():
        ann = result.get("ann")
        failed = "、".join(
            key for key, value in result["checks"].items() if not value
        )
        lines.append(
            f"| {result['temperature_c']:.0f} ℃ | Phase 7C独立确认 | "
            f"{'通过' if result.get('success', False) else '触发停止'} | "
            f"{'通过' if result.get('success', False) else '未通过/未运行'} | "
            f"{(ann or result['mpc'])['maximum_voltage_v']:.6f} V | "
            f"{(ann or result['mpc'])['maximum_average_temperature_c']:.3f} ℃ | "
            f"{'严格通过' if result.get('success', False) else failed} |"
        )
    lines.extend(
        [
            "",
            "## 安全 MPC 诊断",
            "",
            "| 温度 | 最大残差 | P95/P99残差 | 最大单步正增长 | 最小电压—斜率余量 | 安全层介入率 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in payload["temperature_results"].values():
        mpc = result["mpc"]
        lines.append(
            f"| {result['temperature_c']:.0f} ℃ | "
            f"{1000 * mpc['maximum_voltage_residual_v']:.3f} mV | "
            f"{1000 * mpc['p95_voltage_residual_v']:.3f}/"
            f"{1000 * mpc['p99_voltage_residual_v']:.3f} mV | "
            f"{1000 * mpc['maximum_positive_residual_growth_v']:.3f} mV | "
            f"{mpc['minimum_voltage_slew_feasibility_margin_a']:.3f} A | "
            f"{100 * mpc['voltage_intervention_fraction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "15 ℃最大单步残差正增长为11.387 mV，已经比冻结的25 ℃裕量"
            "11.306 mV高0.082 mV；本批轨迹仍未越压或出现空区间，但这证明"
            "25 ℃残差增长界本身不能作为全温度上界。",
            "",
            "30 ℃安全 MPC 最高平均温度达到42.264 ℃，超过35 ℃门槛"
            "7.264 ℃。这属于缺少热约束感知，而不是ANN拟合失败。另有1条"
            "30 ℃轨迹出现一次显著电流方向反转；15 ℃有1个控制步返回"
            "optimizer_success=False但prediction_feasible=True，均按预注册"
            "停止规则保留为失败证据。",
            "",
            "## 阶段判定",
            "",
            payload["decision"]["conclusion"],
            "",
            "若任一温度触发停止，本报告不把失败归因于 ANN，除非安全 MPC "
            "通过而安全 ANN 单独失败。确认集不得用于调参后重复宣称独立通过。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_phase7c(
    config: Phase7CConfig,
    root: Path,
    mpc: pd.DataFrame,
    ann: pd.DataFrame | None,
) -> dict[str, Any]:
    phase7b1b = load_phase7b1b_config(root / config.phase7b1b_config)
    level3, inherited, _, _, phase7b0 = _load_context(phase7b1b, root)
    output = root / config.result_directory
    data_dir = root / config.data_directory
    output.mkdir(parents=True, exist_ok=True)
    all_frames = [mpc]
    if ann is not None:
        all_frames.append(ann)
    frame = pd.concat(all_frames, ignore_index=True)
    frame.to_csv(data_dir / "closed_loop_trajectories.csv", index=False)
    seed_tables = []
    temperature_results: dict[str, Any] = {}
    for temperature_c in config.contract.temperatures_c:
        seeds, result = _temperature_metrics(
            config, root, temperature_c, mpc, ann
        )
        if len(seeds):
            seed_tables.append(seeds)
        temperature_results[f"{int(temperature_c)}c"] = result
    seed_metrics = (
        pd.concat(seed_tables, ignore_index=True)
        if seed_tables
        else pd.DataFrame()
    )
    controller_rows = []
    for result in temperature_results.values():
        controller_rows.append(
            {
                "temperature_c": result["temperature_c"],
                "controller_kind": "mpc",
                "seed": -1,
                **result["mpc"],
                "strict_success": result["success"],
            }
        )
    controller_metrics = pd.DataFrame(controller_rows)
    if len(seed_metrics):
        seed_output = seed_metrics.copy()
        seed_output.insert(1, "controller_kind", "ann")
        closed_loop_metrics = pd.concat(
            [controller_metrics, seed_output],
            ignore_index=True,
            sort=False,
        )
    else:
        closed_loop_metrics = controller_metrics
    closed_loop_metrics.to_csv(
        data_dir / "closed_loop_metrics.csv", index=False
    )
    diagnostics = _trajectory_diagnostics(
        frame,
        inherited.mpc.target_soc - phase7b0.dfn.target_soc_tolerance,
        level3.model.sample_period_s,
    )
    diagnostics.to_csv(
        data_dir / "trajectory_diagnostics.csv", index=False
    )
    mpc_pass = all(
        all(
            value
            for key, value in result["checks"].items()
            if key
            in {
                "maximum_voltage",
                "maximum_average_temperature",
                "zero_current_violation",
                "zero_slew_violation",
                "zero_empty_interval",
                "target_reach_100_percent",
                "zero_oscillation",
                "zero_solver_failure",
                "mpc_voltage_safe",
                "mpc_temperature_safe",
            }
        )
        for result in temperature_results.values()
    )
    complete = ann is not None
    success = complete and all(
        result["success"] for result in temperature_results.values()
    )
    failed_checks = {
        token: [key for key, value in result["checks"].items() if not value]
        for token, result in temperature_results.items()
        if not result["success"]
    }
    if not mpc_pass:
        conclusion = (
            "Phase 7C 在安全 MPC 阶段触发严格停止：冻结的25 ℃电压/热安全"
            "合同不能直接外推到全部温度；未运行 ANN 扩大实验。"
        )
    elif success:
        conclusion = (
            "Phase 7C 严格通过：冻结于25 ℃的 ANN＋输入投影＋在线电压"
            "残差安全层在15–30 ℃ DFN上实现零调参外推。"
        )
    elif complete:
        conclusion = (
            "安全 MPC 通过但安全 ANN 未全部通过，温度分布外策略退化需要"
            "单独诊断，禁止立即重训。"
        )
    else:
        conclusion = "安全 MPC 通过，等待五种子安全 ANN 闭环。"
    phase7b1_metrics = json.loads(
        (
            root / "outputs/phase7b1b_voltage_safety/confirmation_metrics.json"
        ).read_text(encoding="utf-8")
    )
    nominal_summary = phase7b1_metrics["residual_guard"]
    payload = {
        "study_name": config.study_name,
        "configuration": asdict(config),
        "frozen_artifact_verification": verify_frozen_artifacts(config, root),
        "temperature_results": temperature_results,
        "nominal_25c_reference": {
            "source": (
                "outputs/phase7b1b_voltage_safety/"
                "confirmation_metrics.json"
            ),
            "maximum_ann_voltage_v": nominal_summary[
                "maximum_ann_voltage_v"
            ],
            "maximum_mpc_voltage_v": nominal_summary[
                "maximum_mpc_voltage_v"
            ],
            "current_nrmse_range": [
                nominal_summary["current_nrmse_min"],
                nominal_summary["current_nrmse_max"],
            ],
            "maximum_charge_time_gap_fraction": nominal_summary[
                "maximum_charge_time_gap_fraction"
            ],
            "minimum_speedup": nominal_summary["minimum_speedup"],
        },
        "decision": {
            "mpc_stage_success": mpc_pass,
            "ann_stage_completed": complete,
            "phase7c_success": success,
            "strict_stop_triggered": (not mpc_pass) or (complete and not success),
            "failed_checks": failed_checks,
            "conclusion": conclusion,
        },
        "status": "completed" if complete or not mpc_pass else "mpc_completed",
        "success": success,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot(output, frame, temperature_results)
    _write_report(output / "PHASE7C_中文实验报告.md", payload)
    return payload


def run_phase7c(
    config: Phase7CConfig,
    root: Path,
    stage: str = "all",
    resume: bool = False,
) -> dict[str, Any]:
    verify_frozen_artifacts(config, root)
    data_dir = root / config.data_directory
    if stage in {"freeze", "all"}:
        freeze_initial_states(config, root)
        if stage == "freeze":
            return {"status": "frozen", "success": True}
    freeze_path = data_dir / "freeze_contract.json"
    if not freeze_path.exists():
        raise RuntimeError("必须先冻结 Phase 7C 初态和合同。")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for relative, record in freeze["files"].items():
        if _sha256(root / relative) != record["sha256"]:
            raise RuntimeError(f"Phase 7C 初态哈希不匹配：{relative}")
    initial = _load_initial_states(config, root)
    phase7b1b = load_phase7b1b_config(root / config.phase7b1b_config)
    _, inherited, _, _, _ = _load_context(phase7b1b, root)
    mpc_jobs = _jobs(
        initial, "mpc", inherited.network.initialization_seeds
    )
    if stage in {"mpc", "all"}:
        mpc = _run_jobs(config, root, mpc_jobs, resume)
        mpc.to_csv(data_dir / "mpc_closed_loop.csv", index=False)
        interim = analyze_phase7c(config, root, mpc, None)
        if not interim["decision"]["mpc_stage_success"]:
            return interim
        if stage == "mpc":
            return interim
    else:
        mpc = pd.read_csv(data_dir / "mpc_closed_loop.csv")
    if stage in {"ann", "all"}:
        ann = _run_jobs(
            config,
            root,
            _jobs(initial, "ann", inherited.network.initialization_seeds),
            resume,
        )
        ann.to_csv(data_dir / "ann_closed_loop.csv", index=False)
    else:
        ann = pd.read_csv(data_dir / "ann_closed_loop.csv")
    return analyze_phase7c(config, root, mpc, ann)
