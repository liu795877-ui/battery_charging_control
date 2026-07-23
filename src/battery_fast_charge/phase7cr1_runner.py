"""Phase 7C-R1：300 s短时域热安全监督层与教师最小修复。"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .phase7a_level3_model import Level3MPC, Level3MPCResult, Level3State
from .phase7b1b_config import load_phase7b1b_config
from .phase7b1b_runner import _load_context, _maximum_safe_current
from .phase7c_runner import Chen2020ThermalDFN, _van_der_corput
from .phase7cr1_config import Phase7CR1Config


def _sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".py", ".yaml", ".yml"}:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def verify_frozen_artifacts(
    config: Phase7CR1Config, root: Path
) -> dict[str, dict[str, Any]]:
    records = {}
    failures = []
    for relative, expected in config.sources["frozen_artifacts"].items():
        actual = _sha256(root / relative)
        matched = actual == expected
        records[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": matched,
        }
        if not matched:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"Phase 7C-R1 冻结工件不匹配：{failures}")
    return records


def _context(config: Phase7CR1Config, root: Path):
    phase7b1b = load_phase7b1b_config(
        root / config.sources["phase7b1b_config"]
    )
    level3, inherited, model, _, phase7b0 = _load_context(phase7b1b, root)
    return phase7b1b, level3, inherited, model, phase7b0


def _surrogate_step(
    temperature_c: float,
    current_a: float,
    ambient_temperature_c: float,
    coefficients: dict[str, float],
) -> float:
    return temperature_c + (
        coefficients["coefficient_i2_c_per_step_a2"] * current_a**2
        + coefficients["coefficient_i_c_per_step_a"] * current_a
        + coefficients["coefficient_temperature_c_per_step_c"]
        * (temperature_c - ambient_temperature_c)
        + coefficients["intercept_c_per_step"]
    )


def _maximum_predicted_temperature(
    current_a: float,
    temperature_c: float,
    config: Phase7CR1Config,
) -> float:
    thermal = config.thermal
    predicted_temperature = float(temperature_c)
    maximum = -np.inf
    if thermal["future_current_policy"] != "constant_candidate":
        raise ValueError("Unsupported R1 future-current policy.")
    for _ in range(int(thermal["prediction_horizon_steps"])):
        predicted_temperature = _surrogate_step(
            predicted_temperature,
            float(current_a),
            thermal["ambient_temperature_c"],
            thermal["surrogate"],
        )
        maximum = max(maximum, predicted_temperature)
    return maximum


def _maximum_braking_temperature(
    current_a: float,
    temperature_c: float,
    config: Phase7CR1Config,
) -> float:
    thermal = config.thermal
    predicted_temperature = float(temperature_c)
    predicted_current = float(current_a)
    maximum = -np.inf
    for step in range(int(thermal["prediction_horizon_steps"])):
        if step > 0:
            predicted_current = max(
                thermal["braking_floor_current_a"],
                predicted_current - 2.0,
            )
        predicted_temperature = _surrogate_step(
            predicted_temperature,
            predicted_current,
            thermal["ambient_temperature_c"],
            thermal["surrogate"],
        )
        maximum = max(maximum, predicted_temperature)
    return maximum


def _maximum_safe_current_for_peak(
    temperature_c: float,
    search_upper_a: float,
    config: Phase7CR1Config,
    peak_function: Any,
) -> tuple[float, float]:
    thermal = config.thermal
    limit = (
        thermal["maximum_average_temperature_c"]
        - thermal["temperature_guard_c"]
    )

    def peak(current_a: float) -> float:
        return peak_function(current_a, temperature_c, config)

    if search_upper_a < 0.0 or peak(0.0) > limit:
        return -1.0, peak(0.0)
    if peak(search_upper_a) <= limit:
        return float(search_upper_a), peak(search_upper_a)
    lower, upper = 0.0, float(search_upper_a)
    while upper - lower > thermal["current_search_tolerance_a"]:
        current = 0.5 * (lower + upper)
        if peak(current) <= limit:
            lower = current
        else:
            upper = current
    return lower, peak(lower)


def maximum_thermal_safe_current(
    temperature_c: float,
    search_upper_a: float,
    config: Phase7CR1Config,
) -> tuple[float, float]:
    return _maximum_safe_current_for_peak(
        temperature_c,
        search_upper_a,
        config,
        _maximum_predicted_temperature,
    )


def maximum_braking_safe_current(
    temperature_c: float,
    search_upper_a: float,
    config: Phase7CR1Config,
) -> tuple[float, float]:
    return _maximum_safe_current_for_peak(
        temperature_c,
        search_upper_a,
        config,
        _maximum_braking_temperature,
    )


def _solve_teacher(
    state: Level3State,
    model: Any,
    config: Phase7CR1Config,
) -> tuple[Level3MPCResult, dict[str, Any]]:
    default = Level3MPC(model).solve(state)
    maximum_step = model.config.constraint.maximum_current_step_a
    lower = max(
        model.inherited.mpc.current_bounds_a[0],
        state.previous_current_a - maximum_step,
    )
    upper = min(
        model.inherited.mpc.current_bounds_a[1],
        state.previous_current_a + maximum_step,
    )
    at_slew_boundary = (
        abs(default.current_a - lower)
        <= config.teacher["slew_boundary_tolerance_a"]
        or abs(default.current_a - upper)
        <= config.teacher["slew_boundary_tolerance_a"]
    )
    retry = (
        (not default.optimizer_success and config.teacher["retry_on_failure"])
        or (
            at_slew_boundary
            and config.teacher["compare_alternative_on_slew_boundary"]
        )
    )
    alternative = None
    selected = default
    if retry:
        alternative_mpc = Level3MPC(model)
        alternative_mpc.set_warm_start(
            np.full(
                alternative_mpc.number_of_blocks,
                state.previous_current_a,
            )
        )
        alternative = alternative_mpc.solve(state)
        candidates = [
            result
            for result in (default, alternative)
            if result.prediction_feasible
        ]
        if candidates:
            selected = min(
                candidates,
                key=lambda result: result.objective_value,
            )
    return selected, {
        "teacher_retry_triggered": retry,
        "default_optimizer_success": default.optimizer_success,
        "default_prediction_feasible": default.prediction_feasible,
        "default_current_a": default.current_a,
        "default_objective": default.objective_value,
        "alternative_optimizer_success": (
            alternative.optimizer_success if alternative is not None else True
        ),
        "alternative_prediction_feasible": (
            alternative.prediction_feasible if alternative is not None else True
        ),
        "alternative_current_a": (
            alternative.current_a if alternative is not None else np.nan
        ),
        "alternative_objective": (
            alternative.objective_value if alternative is not None else np.nan
        ),
        "alternative_selected": (
            alternative is not None and selected is alternative
        ),
        "teacher_branch_objective_improvement": (
            default.objective_value - selected.objective_value
        ),
    }


def freeze_development_states(
    config: Phase7CR1Config, root: Path
) -> dict[str, Any]:
    data_dir = root / config.output["data_directory"]
    data_dir.mkdir(parents=True, exist_ok=True)
    _, level3, _, model, _ = _context(config, root)
    dev = config.development
    records = []
    candidate = int(dev["design_start_index"])
    voltage_limit = (
        model.inherited.mpc.terminal_voltage_max_v
        - dev["initial_voltage_margin_v"]
    )
    while len(records) < int(dev["trajectory_count"]):
        soc = dev["initial_soc_bounds"][0] + np.ptp(
            dev["initial_soc_bounds"]
        ) * _van_der_corput(candidate, 2)
        v1 = dev["initial_v1_bounds_v"][0] + np.ptp(
            dev["initial_v1_bounds_v"]
        ) * _van_der_corput(candidate, 3)
        v2 = dev["initial_v2_bounds_v"][0] + np.ptp(
            dev["initial_v2_bounds_v"]
        ) * _van_der_corput(candidate, 5)
        previous = dev["initial_previous_current_bounds_a"][0] + np.ptp(
            dev["initial_previous_current_bounds_a"]
        ) * _van_der_corput(candidate, 7)
        used = candidate
        candidate += 1
        state = Level3State(soc, v1, v2, previous)
        minimum_current = max(
            0.0, previous - level3.constraint.maximum_current_step_a
        )
        if model.terminal_voltage(state, minimum_current) > voltage_limit:
            continue
        result, _ = _solve_teacher(state, model, config)
        if not result.prediction_feasible:
            continue
        records.append(
            {
                "trajectory_id": f"phase7cr1_dev_{len(records):03d}",
                "ambient_temperature_c": 30.0,
                "initial_temperature_c": 30.0,
                "initial_soc": soc,
                "initial_polarization_1_v": v1,
                "initial_polarization_2_v": v2,
                "initial_previous_current_a": previous,
                "design_candidate_index": used,
                "design_seed": dev["design_seed"],
            }
        )
    frame = pd.DataFrame(records)
    path = data_dir / "development_initial_states.csv"
    frame.to_csv(path, index=False)
    freeze = {
        "development_only": True,
        "not_independent_confirmation": True,
        "not_ann_teacher_data": True,
        "trajectory_count": len(frame),
        "sha256": _sha256(path),
        "source_artifacts": verify_frozen_artifacts(config, root),
    }
    (data_dir / "development_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return freeze


def _rollout(
    config: Phase7CR1Config,
    root: Path,
    initial: dict[str, Any],
) -> pd.DataFrame:
    phase7b1b, level3, inherited, model, phase7b0 = _context(config, root)
    state = Level3State(
        initial["initial_soc"],
        initial["initial_polarization_1_v"],
        initial["initial_polarization_2_v"],
        initial["initial_previous_current_a"],
    )
    plant = Chen2020ThermalDFN(
        phase7b0.dfn.parameter_set,
        initial["ambient_temperature_c"],
        phase7b0.dfn.upper_voltage_cutoff_v,
        state.soc,
        level3.model.sample_period_s,
        "lumped",
    )
    temperature_c = float(initial["initial_temperature_c"])
    voltage_residual_v = 0.0
    lower_bound, upper_bound = inherited.mpc.current_bounds_a
    maximum_step = level3.constraint.maximum_current_step_a
    rows = []
    for step in range(int(config.development["maximum_steps"])):
        started = perf_counter()
        result, teacher = _solve_teacher(state, model, config)
        teacher_time_s = perf_counter() - started
        candidate_current = result.current_a
        slew_lower = max(
            lower_bound, state.previous_current_a - maximum_step
        )
        slew_upper = min(
            upper_bound, state.previous_current_a + maximum_step
        )
        voltage_max = _maximum_safe_current(
            state,
            voltage_residual_v
            + phase7b1b.safety.residual_growth_guard_v,
            phase7b1b,
            model,
        )
        thermal_started = perf_counter()
        search_upper = min(slew_upper, voltage_max)
        constant_thermal_max, constant_predicted_peak = (
            maximum_thermal_safe_current(
                temperature_c, search_upper, config
            )
        )
        braking_thermal_max, braking_predicted_peak = (
            maximum_braking_safe_current(
                temperature_c, search_upper, config
            )
        )
        thermal_braking_active = (
            constant_thermal_max
            < slew_lower - config.thermal["empty_interval_tolerance_a"]
        )
        if thermal_braking_active:
            thermal_max = min(braking_thermal_max, slew_lower)
            predicted_peak = braking_predicted_peak
        else:
            thermal_max = constant_thermal_max
            predicted_peak = constant_predicted_peak
        thermal_time_s = perf_counter() - thermal_started
        final_upper = min(slew_upper, voltage_max, thermal_max)
        empty = (
            final_upper
            < slew_lower - config.thermal["empty_interval_tolerance_a"]
        )
        common = {
            "trajectory_id": initial["trajectory_id"],
            "step_index": step,
            "time_s": (step + 1) * level3.model.sample_period_s,
            "soc": state.soc,
            "temperature_c": temperature_c,
            "previous_current_a": state.previous_current_a,
            "candidate_current_a": candidate_current,
            "slew_lower_a": slew_lower,
            "slew_upper_a": slew_upper,
            "voltage_safe_current_max_a": voltage_max,
            "thermal_safe_current_max_a": thermal_max,
            "constant_thermal_safe_current_max_a": constant_thermal_max,
            "braking_thermal_safe_current_max_a": braking_thermal_max,
            "predicted_300s_peak_temperature_c": predicted_peak,
            "thermal_braking_active": thermal_braking_active,
            "final_upper_a": final_upper,
            "empty_final_interval": empty,
            "teacher_time_s": teacher_time_s,
            "thermal_supervisor_time_s": thermal_time_s,
            **teacher,
        }
        if empty:
            rows.append(
                {
                    **common,
                    "current_a": np.nan,
                    "next_soc": state.soc,
                    "next_temperature_c": temperature_c,
                    "terminal_voltage_v": np.nan,
                    "current_step_a": np.nan,
                    "voltage_residual_v": voltage_residual_v,
                    "thermal_intervened": False,
                    "thermal_current_correction_a": np.nan,
                    "optimizer_success": result.optimizer_success,
                    "prediction_feasible": result.prediction_feasible,
                }
            )
            break
        current = float(
            np.clip(
                min(candidate_current, final_upper),
                slew_lower,
                final_upper,
            )
        )
        predicted = model.step(state, current)
        predicted_voltage = model.terminal_voltage(predicted, current)
        measurement = plant.step(current)
        new_residual = (
            float(measurement["terminal_voltage_v"]) - predicted_voltage
        )
        rows.append(
            {
                **common,
                "current_a": current,
                "next_soc": measurement["soc"],
                "next_temperature_c": measurement[
                    "average_temperature_c"
                ],
                "terminal_voltage_v": measurement["terminal_voltage_v"],
                "current_step_a": abs(
                    current - state.previous_current_a
                ),
                "voltage_residual_v": new_residual,
                "thermal_intervened": (
                    current
                    < min(candidate_current, voltage_max, slew_upper)
                    - 1.0e-9
                ),
                "thermal_current_correction_a": (
                    current - min(candidate_current, voltage_max, slew_upper)
                ),
                "optimizer_success": result.optimizer_success,
                "prediction_feasible": result.prediction_feasible,
            }
        )
        state = Level3State(
            float(measurement["soc"]),
            predicted.polarization_1_v,
            predicted.polarization_2_v,
            current,
        )
        temperature_c = float(measurement["average_temperature_c"])
        voltage_residual_v = new_residual
        if state.soc >= config.development["target_soc"]:
            break
    return pd.DataFrame(rows)


def _worker(
    config: Phase7CR1Config,
    root_text: str,
    initial: dict[str, Any],
    path_text: str,
) -> str:
    frame = _rollout(config, Path(root_text), initial)
    frame.to_csv(path_text, index=False)
    return path_text


def _run_development(
    config: Phase7CR1Config, root: Path, resume: bool
) -> pd.DataFrame:
    data_dir = root / config.output["data_directory"]
    initial = pd.read_csv(data_dir / "development_initial_states.csv")
    run_dir = data_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    pending = []
    for row in initial.to_dict(orient="records"):
        path = run_dir / f"{row['trajectory_id']}.csv"
        paths.append(path)
        if not (resume and path.exists()):
            pending.append((row, path))
    if pending:
        with ProcessPoolExecutor(
            max_workers=int(config.development["maximum_workers"])
        ) as executor:
            futures = {
                executor.submit(
                    _worker, config, str(root), row, str(path)
                ): path
                for row, path in pending
            }
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                print(
                    f"[Phase 7C-R1] completed "
                    f"{completed}/{len(pending)}",
                    flush=True,
                )
    return pd.concat(
        [pd.read_csv(path) for path in paths], ignore_index=True
    )


def _reversal_count(values: np.ndarray, threshold: float = 0.25) -> int:
    differences = np.diff(values)
    signs = np.sign(differences[np.abs(differences) >= threshold])
    return int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0


def _sustained_oscillation_count(values: np.ndarray) -> int:
    differences = np.diff(values)
    indices = np.where(np.abs(differences) >= 0.25)[0]
    signs = np.sign(differences[indices])
    reversals = indices[1:][signs[1:] * signs[:-1] < 0]
    if len(reversals) < 2:
        return 0
    return int(np.sum(np.diff(reversals) <= 4))


def _analyze(
    config: Phase7CR1Config,
    root: Path,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    data_dir = root / config.output["data_directory"]
    result_dir = root / config.output["result_directory"]
    result_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_dir / "development_trajectories.csv", index=False)
    records = []
    for trajectory_id, group in frame.groupby("trajectory_id"):
        valid = group[~group.current_a.isna()]
        interventions = valid[valid.thermal_intervened.astype(bool)]
        records.append(
            {
                "trajectory_id": trajectory_id,
                "target_reached": (
                    len(valid) > 0
                    and valid.next_soc.iloc[-1]
                    >= config.development["target_soc"]
                ),
                "charge_time_s": (
                    float(valid.time_s.iloc[-1]) if len(valid) else np.nan
                ),
                "maximum_voltage_v": float(
                    valid.terminal_voltage_v.max()
                ),
                "maximum_average_temperature_c": float(
                    valid.next_temperature_c.max()
                ),
                "maximum_current_step_a": float(
                    valid.current_step_a.max()
                ),
                "empty_interval_count": int(
                    group.empty_final_interval.astype(bool).sum()
                ),
                "solver_failure_count": int(
                    (~group.optimizer_success.astype(bool)).sum()
                ),
                "prediction_infeasible_count": int(
                    (~group.prediction_feasible.astype(bool)).sum()
                ),
                "teacher_retry_count": int(
                    group.teacher_retry_triggered.astype(bool).sum()
                ),
                "alternative_selected_count": int(
                    group.alternative_selected.astype(bool).sum()
                ),
                "direction_reversal_count": _reversal_count(
                    valid.current_a.to_numpy(float)
                ),
                "sustained_oscillation_count": (
                    _sustained_oscillation_count(
                        valid.current_a.to_numpy(float)
                    )
                ),
                "thermal_intervention_fraction": float(
                    valid.thermal_intervened.astype(bool).mean()
                ),
                "maximum_thermal_current_correction_a": float(
                    valid.thermal_current_correction_a.abs().max()
                ),
                "first_thermal_intervention_time_s": (
                    float(interventions.time_s.iloc[0])
                    if len(interventions)
                    else np.nan
                ),
                "first_thermal_intervention_soc": (
                    float(interventions.soc.iloc[0])
                    if len(interventions)
                    else np.nan
                ),
                "first_thermal_intervention_temperature_c": (
                    float(interventions.temperature_c.iloc[0])
                    if len(interventions)
                    else np.nan
                ),
                "mean_teacher_time_ms": float(
                    1000.0 * valid.teacher_time_s.mean()
                ),
                "mean_thermal_supervisor_time_ms": float(
                    1000.0 * valid.thermal_supervisor_time_s.mean()
                ),
            }
        )
    metrics = pd.DataFrame(records)
    metrics.to_csv(data_dir / "development_metrics.csv", index=False)
    gates = config.gates
    checks = {
        "temperature_safe": bool(
            (
                metrics.maximum_average_temperature_c
                <= gates["maximum_average_temperature_c"]
                + gates["numerical_tolerance"]
            ).all()
        ),
        "voltage_safe": bool(
            (
                metrics.maximum_voltage_v
                <= gates["maximum_voltage_v"]
            ).all()
        ),
        "slew_safe": bool(
            (
                metrics.maximum_current_step_a
                <= gates["maximum_current_step_a"]
                + gates["numerical_tolerance"]
            ).all()
        ),
        "zero_empty_interval": bool(
            metrics.empty_interval_count.sum()
            <= gates["maximum_empty_interval_count"]
        ),
        "zero_solver_failure_after_repair": bool(
            metrics.solver_failure_count.sum()
            <= gates["maximum_solver_failure_count"]
        ),
        "zero_prediction_infeasible": bool(
            metrics.prediction_infeasible_count.sum() == 0
        ),
        "target_reach_100_percent": bool(
            metrics.target_reached.mean()
            >= gates["minimum_target_reach_fraction"]
        ),
        "zero_sustained_oscillation": bool(
            metrics.sustained_oscillation_count.sum()
            <= gates["maximum_sustained_oscillation_count"]
        ),
    }
    success = bool(all(checks.values()))
    payload = {
        "study_name": config.study_name,
        "configuration": config.payload,
        "frozen_artifact_verification": verify_frozen_artifacts(
            config, root
        ),
        "development_set": {
            "trajectory_count": len(metrics),
            "not_independent_confirmation": True,
            "maximum_voltage_v": float(metrics.maximum_voltage_v.max()),
            "maximum_average_temperature_c": float(
                metrics.maximum_average_temperature_c.max()
            ),
            "maximum_current_step_a": float(
                metrics.maximum_current_step_a.max()
            ),
            "target_reach_fraction": float(metrics.target_reached.mean()),
            "empty_interval_count": int(
                metrics.empty_interval_count.sum()
            ),
            "solver_failure_count": int(
                metrics.solver_failure_count.sum()
            ),
            "teacher_retry_count": int(
                metrics.teacher_retry_count.sum()
            ),
            "alternative_selected_count": int(
                metrics.alternative_selected_count.sum()
            ),
            "direction_reversal_count": int(
                metrics.direction_reversal_count.sum()
            ),
            "sustained_oscillation_count": int(
                metrics.sustained_oscillation_count.sum()
            ),
            "thermal_intervention_fraction_range": [
                float(metrics.thermal_intervention_fraction.min()),
                float(metrics.thermal_intervention_fraction.max()),
            ],
            "maximum_thermal_current_correction_a": float(
                metrics.maximum_thermal_current_correction_a.max()
            ),
            "earliest_thermal_intervention_time_s": float(
                metrics.first_thermal_intervention_time_s.min()
            ),
            "earliest_thermal_intervention_temperature_c": float(
                metrics.first_thermal_intervention_temperature_c.min()
            ),
            "mean_thermal_supervisor_time_ms": float(
                metrics.mean_thermal_supervisor_time_ms.mean()
            ),
        },
        "checks": checks,
        "success": success,
        "decision": {
            "freeze_r1_for_new_confirmation": success,
            "run_ann": False,
            "proceed_to_phase7cr2_voltage_guard_development": success,
            "conclusion": (
                "R1开发集严格通过：可以冻结短时域热监督层并进入R2分温度"
                "电压裕量开发；仍禁止运行ANN。"
                if success
                else "R1开发集未通过：停止，不生成新确认集。"
            ),
        },
    }
    payload["development_iterations"] = [
        {
            "name": "无承诺制动预测",
            "target_reach_fraction": 0.25,
            "empty_interval_count": 6,
            "sustained_oscillation_count": None,
            "accepted": False,
            "finding": "滚动时域反复推迟制动。",
        },
        {
            "name": "仅恒流窗口上限",
            "target_reach_fraction": 0.625,
            "empty_interval_count": 3,
            "sustained_oscillation_count": None,
            "accepted": False,
            "finding": "高初始电流使热上限低于斜率下限。",
        },
        {
            "name": "持续制动模式",
            "target_reach_fraction": 1.0,
            "empty_interval_count": 0,
            "sustained_oscillation_count": 1927,
            "accepted": False,
            "finding": "制动解除逻辑产生两步往返。",
        },
        {
            "name": "恒流窗口加瞬态恢复",
            "target_reach_fraction": float(metrics.target_reached.mean()),
            "empty_interval_count": int(
                metrics.empty_interval_count.sum()
            ),
            "sustained_oscillation_count": int(
                metrics.sustained_oscillation_count.sum()
            ),
            "accepted": success,
            "finding": "仅在恒流热上限低于斜率下限时执行瞬态制动。",
        },
    ]
    (result_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot(result_dir, frame)
    _write_report(result_dir / "PHASE7C-R1_中文开发报告.md", payload)
    return payload


def _plot(result_dir: Path, frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(15, 4.5), layout="constrained"
    )
    for trajectory_id, group in frame.groupby("trajectory_id"):
        axes[0].plot(
            group.time_s, group.next_temperature_c, alpha=0.7
        )
        axes[1].plot(group.time_s, group.current_a, alpha=0.7)
    axes[0].axhline(35.0, color="red", linestyle="--")
    axes[0].set(
        xlabel="Time [s]",
        ylabel="Average temperature [℃]",
        title="30 ℃ development trajectories",
    )
    axes[1].set(
        xlabel="Time [s]",
        ylabel="Current [A]",
        title="Thermally supervised MPC",
    )
    active = frame[frame.thermal_intervened.astype(bool)]
    axes[2].scatter(
        active.soc,
        active.temperature_c,
        s=8,
        alpha=0.5,
    )
    axes[2].set(
        xlabel="SOC",
        ylabel="Temperature at intervention [℃]",
        title="Thermal supervisor interventions",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(
        result_dir / "phase7cr1_thermal_supervisor.png", dpi=180
    )
    plt.close(figure)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["development_set"]
    iteration_rows = "\n".join(
        "| {name} | {reach:.1f}% | {empty} | {oscillation} | {accepted} |".format(
            name=item["name"],
            reach=100.0 * item["target_reach_fraction"],
            empty=item["empty_interval_count"],
            oscillation=(
                item["sustained_oscillation_count"]
                if item["sustained_oscillation_count"] is not None
                else "未记录"
            ),
            accepted="接受" if item["accepted"] else "拒绝",
        )
        for item in payload["development_iterations"]
    )
    check_rows = "\n".join(
        f"| {name} | {'通过' if passed else '失败'} |"
        for name, passed in payload["checks"].items()
    )
    report = f"""# Phase 7C-R1：短时域热安全监督层开发报告

## 合同

R1没有重新训练ANN、没有生成ANN教师数据，也没有运行多温度ANN。Phase 7C
原48条确认轨迹没有用于调参，继续保留为永久失败证据。

热监督层使用300 s预测窗口和0.10 ℃开发裕量。正常状态采用恒候选电流预测；
只有当恒流热上限低于斜率下限时，才执行2 A/5 s瞬态制动恢复。若制动轨迹
仍不可行，实验直接失败，禁止突破斜率限制。

## 开发迭代

| 方案 | 到达率 | 空区间 | 持续振荡 | 判定 |
|---|---:|---:|---:|---|
{iteration_rows}

8条开发轨迹用于发现并修复滚动时域拖延、高初始电流斜率冲突和两步往返；
只有最后一版被接受。本表完整保留开发过程，不能把最终结果表述为独立确认。

## 8条新开发轨迹

- 最高DFN电压：{summary['maximum_voltage_v']:.6f} V
- 最高平均温度：{summary['maximum_average_temperature_c']:.6f} ℃
- 最大单步电流变化：{summary['maximum_current_step_a']:.6f} A
- 目标到达率：{100 * summary['target_reach_fraction']:.1f}%
- 空区间：{summary['empty_interval_count']}
- 修复后求解失败：{summary['solver_failure_count']}
- 教师重试/替代分支被选中：{summary['teacher_retry_count']}/
  {summary['alternative_selected_count']}
- 持续振荡：{summary['sustained_oscillation_count']}
- 热层介入率范围：{summary['thermal_intervention_fraction_range']}
- 最大热电流修正：{summary['maximum_thermal_current_correction_a']:.6f} A
- 最早介入时间/温度：{summary['earliest_thermal_intervention_time_s']:.1f} s /
  {summary['earliest_thermal_intervention_temperature_c']:.3f} ℃
- 热监督层平均计算时间：
  {summary['mean_thermal_supervisor_time_ms']:.3f} ms

## 严格门槛

| 检查项 | 结果 |
|---|---|
{check_rows}

## 判定

{payload['decision']['conclusion']}

本结果仅是开发集证据。R1和R2参数全部冻结后必须生成新的独立确认集；
Phase 7C原48条轨迹只用于永久失败回归。
"""
    path.write_text(report, encoding="utf-8")


def run_phase7cr1(
    config: Phase7CR1Config,
    root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    verify_frozen_artifacts(config, root)
    data_dir = root / config.output["data_directory"]
    initial_path = data_dir / "development_initial_states.csv"
    if not initial_path.exists():
        freeze_development_states(config, root)
    freeze = json.loads(
        (data_dir / "development_freeze.json").read_text(encoding="utf-8")
    )
    if _sha256(initial_path) != freeze["sha256"]:
        raise RuntimeError("R1开发初态哈希不匹配。")
    frame = _run_development(config, root, resume)
    return _analyze(config, root, frame)
