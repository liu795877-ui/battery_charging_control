"""Phase 7C-R0：热可行域、15 ℃求解失败和30 ℃方向反转诊断。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

from .phase7a_level3_model import Level3MPC, Level3State
from .phase7b1b_config import load_phase7b1b_config
from .phase7b1b_runner import _load_context, _maximum_safe_current
from .phase7c_runner import Chen2020ThermalDFN
from .phase7cr0_config import Phase7CR0Config


def _sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".py", ".yaml", ".yml"}:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def verify_frozen_artifacts(
    config: Phase7CR0Config, root: Path
) -> dict[str, dict[str, Any]]:
    records = {}
    failures = []
    for relative, expected in config.frozen_artifacts.items():
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
        raise RuntimeError(f"Phase 7C-R0 冻结失败证据不匹配：{failures}")
    return records


def _context(config: Phase7CR0Config, root: Path):
    phase7b1b = load_phase7b1b_config(root / config.phase7b1b_config)
    level3, inherited, model, _, phase7b0 = _load_context(phase7b1b, root)
    return phase7b1b, level3, inherited, model, phase7b0


def _thermal_rollout(
    config: Phase7CR0Config,
    root: Path,
    policy_name: str,
    policy: Callable[[Level3State, float, int], float],
) -> pd.DataFrame:
    phase7b1b, level3, inherited, model, phase7b0 = _context(config, root)
    thermal = config.thermal
    state = Level3State(
        thermal.initial_soc,
        thermal.initial_polarization_1_v,
        thermal.initial_polarization_2_v,
        thermal.initial_previous_current_a,
    )
    plant = Chen2020ThermalDFN(
        phase7b0.dfn.parameter_set,
        thermal.ambient_temperature_c,
        phase7b0.dfn.upper_voltage_cutoff_v,
        state.soc,
        level3.model.sample_period_s,
        "lumped",
    )
    temperature_c = thermal.ambient_temperature_c
    residual_v = 0.0
    lower, upper = inherited.mpc.current_bounds_a
    maximum_step = level3.constraint.maximum_current_step_a
    rows = []
    for step in range(thermal.maximum_steps):
        candidate = float(np.clip(policy(state, temperature_c, step), lower, upper))
        slew_lower = max(lower, state.previous_current_a - maximum_step)
        slew_upper = min(upper, state.previous_current_a + maximum_step)
        projected = float(np.clip(candidate, slew_lower, slew_upper))
        correction = residual_v + phase7b1b.safety.residual_growth_guard_v
        voltage_max = _maximum_safe_current(
            state, correction, phase7b1b, model
        )
        empty = (
            voltage_max
            < slew_lower - phase7b1b.safety.empty_interval_tolerance_a
        )
        if empty:
            rows.append(
                {
                    "policy": policy_name,
                    "step_index": step,
                    "time_s": step * level3.model.sample_period_s,
                    "soc": state.soc,
                    "next_soc": state.soc,
                    "temperature_c": temperature_c,
                    "candidate_current_a": candidate,
                    "current_a": np.nan,
                    "terminal_voltage_v": np.nan,
                    "voltage_residual_v": residual_v,
                    "slew_lower_a": slew_lower,
                    "voltage_safe_current_max_a": voltage_max,
                    "empty_voltage_slew_interval": True,
                }
            )
            break
        current = float(
            np.clip(min(projected, voltage_max), slew_lower, slew_upper)
        )
        predicted = model.step(state, current)
        predicted_voltage = model.terminal_voltage(predicted, current)
        measurement = plant.step(current)
        new_temperature_c = float(measurement["average_temperature_c"])
        new_residual = (
            float(measurement["terminal_voltage_v"]) - predicted_voltage
        )
        rows.append(
            {
                "policy": policy_name,
                "step_index": step,
                "time_s": (step + 1) * level3.model.sample_period_s,
                "soc": state.soc,
                "next_soc": measurement["soc"],
                "temperature_c": new_temperature_c,
                "temperature_increment_c": (
                    new_temperature_c - temperature_c
                ),
                "candidate_current_a": candidate,
                "current_a": current,
                "current_step_a": abs(current - state.previous_current_a),
                "terminal_voltage_v": measurement["terminal_voltage_v"],
                "voltage_residual_v": new_residual,
                "slew_lower_a": slew_lower,
                "voltage_safe_current_max_a": voltage_max,
                "empty_voltage_slew_interval": False,
                "voltage_intervened": abs(current - projected) > 1.0e-12,
            }
        )
        state = Level3State(
            float(measurement["soc"]),
            predicted.polarization_1_v,
            predicted.polarization_2_v,
            current,
        )
        temperature_c = new_temperature_c
        residual_v = new_residual
        if state.soc >= thermal.target_soc:
            break
    return pd.DataFrame(rows)


def _policy_summary(
    frame: pd.DataFrame, config: Phase7CR0Config
) -> dict[str, Any]:
    valid = frame[~frame.current_a.isna()]
    reached = (
        len(valid) > 0
        and float(valid.next_soc.iloc[-1]) >= config.thermal.target_soc
    )
    maximum_temperature = float(valid.temperature_c.max())
    return {
        "policy": str(frame.policy.iloc[0]),
        "reached_target": reached,
        "charge_time_s": float(valid.time_s.iloc[-1]) if reached else np.nan,
        "final_soc": float(valid.next_soc.iloc[-1]),
        "maximum_temperature_c": maximum_temperature,
        "maximum_voltage_v": float(valid.terminal_voltage_v.max()),
        "maximum_current_step_a": float(valid.current_step_a.max()),
        "empty_interval_count": int(
            frame.empty_voltage_slew_interval.astype(bool).sum()
        ),
        "thermally_feasible": bool(
            reached
            and maximum_temperature
            <= config.thermal.maximum_average_temperature_c + 1.0e-9
            and float(valid.terminal_voltage_v.max()) <= 4.200001
            and int(frame.empty_voltage_slew_interval.astype(bool).sum()) == 0
        ),
        "voltage_intervention_fraction": float(
            valid.voltage_intervened.astype(bool).mean()
        ),
    }


def _run_constant_current_sweep(
    config: Phase7CR0Config, root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summaries = []
    for current_a in config.thermal.constant_current_levels_a:
        name = f"cc_{current_a:g}a"
        frame = _thermal_rollout(
            config,
            root,
            name,
            lambda state, temperature, step, value=current_a: value,
        )
        frames.append(frame)
        summaries.append(_policy_summary(frame, config))
        print(
            f"[Phase 7C-R0] {name}: "
            f"Tmax={summaries[-1]['maximum_temperature_c']:.3f} ℃, "
            f"feasible={summaries[-1]['thermally_feasible']}",
            flush=True,
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def _fit_thermal_surrogate(
    frame: pd.DataFrame, ambient_temperature_c: float
) -> dict[str, Any]:
    valid = frame.dropna(
        subset=["temperature_increment_c", "current_a", "temperature_c"]
    ).copy()
    previous_temperature = (
        valid.temperature_c - valid.temperature_increment_c
    )
    current = valid.current_a.to_numpy(float)
    design = np.column_stack(
        [
            current**2,
            current,
            previous_temperature.to_numpy(float) - ambient_temperature_c,
            np.ones(len(valid)),
        ]
    )
    target = valid.temperature_increment_c.to_numpy(float)
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients
    residual = target - predicted
    return {
        "coefficient_i2_c_per_step_a2": float(coefficients[0]),
        "coefficient_i_c_per_step_a": float(coefficients[1]),
        "coefficient_temperature_c_per_step_c": float(coefficients[2]),
        "intercept_c_per_step": float(coefficients[3]),
        "rmse_c_per_step": float(np.sqrt(np.mean(residual**2))),
        "maximum_absolute_error_c_per_step": float(
            np.max(np.abs(residual))
        ),
        "sample_count": len(valid),
    }


def _surrogate_temperature_step(
    temperature_c: float,
    current_a: float,
    ambient_temperature_c: float,
    fit: dict[str, Any],
) -> float:
    increment = (
        fit["coefficient_i2_c_per_step_a2"] * current_a**2
        + fit["coefficient_i_c_per_step_a"] * current_a
        + fit["coefficient_temperature_c_per_step_c"]
        * (temperature_c - ambient_temperature_c)
        + fit["intercept_c_per_step"]
    )
    return temperature_c + increment


def _surrogate_policy(
    config: Phase7CR0Config,
    root: Path,
    fit: dict[str, Any],
    high_current_a: float,
    low_current_a: float,
    switch_temperature_c: float,
    switch_soc: float,
) -> dict[str, Any]:
    phase7b1b, level3, inherited, model, _ = _context(config, root)
    thermal = config.thermal
    state = Level3State(
        thermal.initial_soc,
        thermal.initial_polarization_1_v,
        thermal.initial_polarization_2_v,
        thermal.initial_previous_current_a,
    )
    temperature_c = thermal.ambient_temperature_c
    residual_v = 0.0
    maximum_temperature_c = temperature_c
    switch_step = None
    lower, upper = inherited.mpc.current_bounds_a
    maximum_step = level3.constraint.maximum_current_step_a
    empty_count = 0
    for step in range(thermal.maximum_steps):
        switched = (
            temperature_c >= switch_temperature_c
            or state.soc >= switch_soc
        )
        if switched and switch_step is None:
            switch_step = step
        candidate = low_current_a if switched else high_current_a
        slew_lower = max(lower, state.previous_current_a - maximum_step)
        slew_upper = min(upper, state.previous_current_a + maximum_step)
        projected = float(np.clip(candidate, slew_lower, slew_upper))
        voltage_max = _maximum_safe_current(
            state,
            residual_v + phase7b1b.safety.residual_growth_guard_v,
            phase7b1b,
            model,
        )
        if voltage_max < slew_lower - 1.0e-9:
            empty_count += 1
            break
        current = float(
            np.clip(min(projected, voltage_max), slew_lower, slew_upper)
        )
        predicted = model.step(state, current)
        temperature_c = _surrogate_temperature_step(
            temperature_c,
            current,
            thermal.ambient_temperature_c,
            fit,
        )
        maximum_temperature_c = max(maximum_temperature_c, temperature_c)
        state = predicted
        residual_v = 0.0
        if state.soc >= thermal.target_soc:
            break
    reached = state.soc >= thermal.target_soc
    feasible = (
        reached
        and maximum_temperature_c
        <= thermal.maximum_average_temperature_c - thermal.thermal_guard_c
        and empty_count == 0
    )
    return {
        "high_current_a": high_current_a,
        "low_current_a": low_current_a,
        "switch_temperature_c": switch_temperature_c,
        "switch_soc": switch_soc,
        "predicted_charge_time_s": (
            (step + 1) * level3.model.sample_period_s if reached else np.nan
        ),
        "predicted_maximum_temperature_c": maximum_temperature_c,
        "predicted_switch_time_s": (
            switch_step * level3.model.sample_period_s
            if switch_step is not None
            else np.nan
        ),
        "predicted_final_soc": state.soc,
        "predicted_empty_interval_count": empty_count,
        "predicted_feasible": feasible,
    }


def _oracle_search(
    config: Phase7CR0Config,
    root: Path,
    fit: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for high in config.thermal.oracle_high_currents_a:
        for low in config.thermal.oracle_low_currents_a:
            if low > high:
                continue
            for switch_temperature in (
                config.thermal.oracle_switch_temperatures_c
            ):
                for switch_soc in config.thermal.oracle_switch_socs:
                    rows.append(
                        _surrogate_policy(
                            config,
                            root,
                            fit,
                            high,
                            low,
                            switch_temperature,
                            switch_soc,
                        )
                    )
    return pd.DataFrame(rows).sort_values(
        ["predicted_feasible", "predicted_charge_time_s"],
        ascending=[False, True],
    )


def _validate_oracle_candidates(
    config: Phase7CR0Config,
    root: Path,
    search: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = search[search.predicted_feasible.astype(bool)].head(
        config.thermal.validated_oracle_candidates
    )
    if candidates.empty:
        candidates = search.head(config.thermal.validated_oracle_candidates)
    frames = []
    summaries = []
    for index, row in enumerate(candidates.itertuples(), start=1):
        name = (
            f"oracle_{index:02d}_h{row.high_current_a:g}_"
            f"l{row.low_current_a:g}_t{row.switch_temperature_c:g}_"
            f"s{row.switch_soc:g}"
        )

        def policy(
            state: Level3State,
            temperature_c: float,
            step: int,
            high=row.high_current_a,
            low=row.low_current_a,
            switch_t=row.switch_temperature_c,
            switch_soc=row.switch_soc,
        ) -> float:
            return (
                low
                if temperature_c >= switch_t or state.soc >= switch_soc
                else high
            )

        frame = _thermal_rollout(config, root, name, policy)
        summary = _policy_summary(frame, config)
        summary.update(
            {
                "high_current_a": row.high_current_a,
                "low_current_a": row.low_current_a,
                "switch_temperature_c": row.switch_temperature_c,
                "switch_soc": row.switch_soc,
                "surrogate_charge_time_s": row.predicted_charge_time_s,
                "surrogate_maximum_temperature_c": (
                    row.predicted_maximum_temperature_c
                ),
            }
        )
        frames.append(frame)
        summaries.append(summary)
        print(
            f"[Phase 7C-R0] {name}: Tmax="
            f"{summary['maximum_temperature_c']:.3f} ℃, "
            f"time={summary['charge_time_s']}, "
            f"feasible={summary['thermally_feasible']}",
            flush=True,
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def _thermal_one_step_audit(
    config: Phase7CR0Config,
    phase7c: pd.DataFrame,
    fit: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    subset = phase7c[phase7c.temperature_c == 30.0]
    grid = np.linspace(0.0, 10.0, 1001)
    for source in subset.itertuples():
        allowed = [
            current
            for current in grid
            if _surrogate_temperature_step(
                source.average_temperature_c,
                float(current),
                30.0,
                fit,
            )
            <= config.thermal.maximum_average_temperature_c
        ]
        thermal_max = max(allowed) if allowed else -1.0
        slew_lower = max(0.0, source.previous_current_a - 2.0)
        rows.append(
            {
                "trajectory_id": source.trajectory_id,
                "step_index": source.step_index,
                "time_s": source.time_s,
                "soc": source.soc,
                "temperature_c": source.average_temperature_c,
                "previous_current_a": source.previous_current_a,
                "thermal_safe_current_max_a": thermal_max,
                "slew_lower_a": slew_lower,
                "thermal_slew_margin_a": thermal_max - slew_lower,
                "empty_thermal_slew_interval": thermal_max < slew_lower,
            }
        )
    output = pd.DataFrame(rows)
    braking_horizon_steps = 60
    augmented = []
    for row in output.itertuples():
        temperature_c = float(row.temperature_c)
        current_a = float(row.previous_current_a)
        maximum_temperature_c = temperature_c
        for _ in range(braking_horizon_steps):
            current_a = max(0.0, current_a - 2.0)
            temperature_c = _surrogate_temperature_step(
                temperature_c,
                current_a,
                config.thermal.ambient_temperature_c,
                fit,
            )
            maximum_temperature_c = max(
                maximum_temperature_c, temperature_c
            )
        augmented.append(
            {
                "maximum_temperature_under_300s_max_braking_c": (
                    maximum_temperature_c
                ),
                "max_braking_too_late": (
                    maximum_temperature_c
                    > config.thermal.maximum_average_temperature_c
                ),
            }
        )
    return pd.concat(
        (output.reset_index(drop=True), pd.DataFrame(augmented)), axis=1
    )


def _reconstruct_state(
    initial: pd.Series,
    trajectory: pd.DataFrame,
    target_step: int,
    model: Any,
) -> Level3State:
    state = Level3State(
        float(initial.initial_soc),
        float(initial.initial_polarization_1_v),
        float(initial.initial_polarization_2_v),
        float(initial.initial_previous_current_a),
    )
    for row in trajectory.sort_values("step_index").itertuples():
        if int(row.step_index) >= target_step:
            break
        predicted = model.step(state, float(row.current_a))
        state = Level3State(
            float(row.next_soc),
            predicted.polarization_1_v,
            predicted.polarization_2_v,
            float(row.current_a),
        )
    return state


def _finite_gradient(
    function: Callable[[np.ndarray], float],
    values: np.ndarray,
    step: float,
) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        upper = values.copy()
        lower = values.copy()
        upper[index] += step
        lower[index] -= step
        result[index] = (function(upper) - function(lower)) / (2.0 * step)
    return result


def _constraint_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    values: np.ndarray,
    step: float,
) -> np.ndarray:
    base = function(values)
    result = np.empty((len(base), len(values)))
    for index in range(len(values)):
        upper = values.copy()
        lower = values.copy()
        upper[index] += step
        lower[index] -= step
        result[:, index] = (function(upper) - function(lower)) / (2.0 * step)
    return result


def _optimization_audit(
    state: Level3State,
    model: Any,
    starts: list[tuple[str, np.ndarray]],
    active_tolerance: float,
    finite_step: float,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    cfg = model.inherited.mpc
    number_of_blocks = (
        cfg.prediction_horizon_steps // cfg.control_block_steps
    )

    def objective(values: np.ndarray) -> float:
        soc, _, _ = model.rollout(state, values)
        return float(
            cfg.soc_tracking_weight * np.sum((cfg.target_soc - soc) ** 2)
            + cfg.terminal_soc_weight * (cfg.target_soc - soc[-1]) ** 2
            + cfg.current_smoothness_weight * np.sum(np.diff(values) ** 2)
        )

    def constraints(values: np.ndarray) -> np.ndarray:
        voltage_margin = (
            cfg.terminal_voltage_max_v - model.rollout(state, values)[1]
        )
        return np.concatenate(
            (voltage_margin, model.slew_margins(state, values))
        )

    records = []
    plans = []
    for label, start in starts:
        result = minimize(
            objective,
            np.clip(start, *cfg.current_bounds_a),
            method="SLSQP",
            bounds=[cfg.current_bounds_a] * number_of_blocks,
            constraints={"type": "ineq", "fun": constraints},
            options={
                "maxiter": cfg.optimizer_max_iterations,
                "ftol": cfg.optimizer_ftol,
                "disp": False,
            },
        )
        plan = np.asarray(result.x, dtype=float)
        plans.append(plan)
        constraint_values = constraints(plan)
        lower_margin = plan - cfg.current_bounds_a[0]
        upper_margin = cfg.current_bounds_a[1] - plan
        all_margins = np.concatenate(
            (constraint_values, lower_margin, upper_margin)
        )
        constraint_jac = _constraint_jacobian(
            constraints, plan, finite_step
        )
        all_jac = np.vstack(
            (
                constraint_jac,
                np.eye(number_of_blocks),
                -np.eye(number_of_blocks),
            )
        )
        active = all_margins <= active_tolerance
        gradient = _finite_gradient(objective, plan, finite_step)
        if np.any(active):
            multipliers, _ = nnls(all_jac[active].T, gradient)
            stationarity = gradient - all_jac[active].T @ multipliers
            complementarity = float(
                np.max(np.abs(multipliers * all_margins[active]))
            )
        else:
            multipliers = np.asarray([])
            stationarity = gradient
            complementarity = 0.0
        primal = max(float(-np.min(all_margins)), 0.0)
        kkt_residual = max(
            float(np.max(np.abs(stationarity))),
            primal,
            complementarity,
        )
        predicted_soc, predicted_voltage, expanded = model.rollout(
            state, plan
        )
        records.append(
            {
                "start_label": label,
                "success": bool(result.success),
                "status_code": int(result.status),
                "status_message": str(result.message),
                "iterations": int(result.nit),
                "objective_value": float(result.fun),
                "first_current_a": float(plan[0]),
                "minimum_constraint_margin": float(np.min(all_margins)),
                "minimum_voltage_margin_v": float(
                    np.min(cfg.terminal_voltage_max_v - predicted_voltage)
                ),
                "minimum_slew_margin_a": float(
                    np.min(model.slew_margins(state, plan))
                ),
                "maximum_predicted_voltage_v": float(
                    np.max(predicted_voltage)
                ),
                "terminal_predicted_soc": float(predicted_soc[-1]),
                "active_constraint_count": int(np.sum(active)),
                "kkt_residual": kkt_residual,
                "stationarity_residual": float(
                    np.max(np.abs(stationarity))
                ),
                "primal_residual": primal,
                "complementarity_residual": complementarity,
                "function_evaluations": int(result.nfev),
                "jacobian_evaluations": int(
                    getattr(result, "njev", -1)
                ),
                "expanded_maximum_step_a": float(
                    np.max(
                        np.abs(
                            np.diff(
                                np.concatenate(
                                    ([state.previous_current_a], expanded)
                                )
                            )
                        )
                    )
                ),
            }
        )
    return pd.DataFrame(records), plans


def _starts_for_state(
    state: Level3State,
    model: Any,
    count: int,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    cfg = model.inherited.mpc
    blocks = cfg.prediction_horizon_steps // cfg.control_block_steps
    rng = np.random.default_rng(seed)
    default = Level3MPC(model)._default_start(state)
    starts = [
        ("controller_default", default),
        ("previous_current", np.full(blocks, state.previous_current_a)),
        ("zero", np.zeros(blocks)),
        ("upper", np.full(blocks, cfg.current_bounds_a[1])),
        ("midpoint", np.full(blocks, np.mean(cfg.current_bounds_a))),
    ]
    while len(starts) < count:
        starts.append(
            (
                f"random_{len(starts):02d}",
                rng.uniform(
                    cfg.current_bounds_a[0],
                    cfg.current_bounds_a[1],
                    size=blocks,
                ),
            )
        )
    return starts[:count]


def _solver_failure_audit(
    config: Phase7CR0Config,
    root: Path,
    phase7c: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _, _, _, model, _ = _context(config, root)
    initial_states = pd.read_csv(root / config.phase7c_initial_states_15c)
    initial = initial_states[
        initial_states.trajectory_id
        == config.solver.failure_trajectory_id
    ].iloc[0]
    trajectory = phase7c[
        (phase7c.temperature_c == config.solver.failure_temperature_c)
        & (
            phase7c.trajectory_id
            == config.solver.failure_trajectory_id
        )
    ]
    state = _reconstruct_state(
        initial,
        trajectory,
        config.solver.failure_step_index,
        model,
    )
    repeated_starts = _starts_for_state(state, model, 1, 0)
    repeated = []
    for index in range(config.solver.repeated_runs):
        table, _ = _optimization_audit(
            state,
            model,
            [(f"repeat_{index:02d}", repeated_starts[0][1])],
            config.solver.active_constraint_tolerance,
            config.solver.finite_difference_step,
        )
        repeated.append(table)
    repeated_table = pd.concat(repeated, ignore_index=True)
    multistart_table, _ = _optimization_audit(
        state,
        model,
        _starts_for_state(
            state,
            model,
            config.solver.multistart_count,
            config.solver.multistart_seed,
        ),
        config.solver.active_constraint_tolerance,
        config.solver.finite_difference_step,
    )
    repeated_table.insert(0, "audit_type", "identical_repeat")
    multistart_table.insert(0, "audit_type", "multistart")
    output = pd.concat(
        (repeated_table, multistart_table), ignore_index=True
    )
    original = trajectory[
        trajectory.step_index == config.solver.failure_step_index
    ].iloc[0]
    summary = {
        "reconstructed_state": asdict(state),
        "original_record": {
            "raw_current_a": float(original.raw_current_a),
            "optimizer_success": bool(original.optimizer_success),
            "prediction_feasible": bool(original.prediction_feasible),
            "soc": float(original.soc),
            "previous_current_a": float(original.previous_current_a),
        },
        "identical_repeat_success_count": int(
            repeated_table.success.astype(bool).sum()
        ),
        "identical_repeat_failure_count": int(
            (~repeated_table.success.astype(bool)).sum()
        ),
        "identical_repeat_unique_first_currents": int(
            repeated_table.first_current_a.round(10).nunique()
        ),
        "multistart_success_count": int(
            multistart_table.success.astype(bool).sum()
        ),
        "multistart_failure_count": int(
            (~multistart_table.success.astype(bool)).sum()
        ),
        "multistart_first_current_range_a": [
            float(multistart_table.first_current_a.min()),
            float(multistart_table.first_current_a.max()),
        ],
        "multistart_objective_range": [
            float(multistart_table.objective_value.min()),
            float(multistart_table.objective_value.max()),
        ],
        "all_returned_plans_strictly_feasible": bool(
            (
                multistart_table.minimum_constraint_margin
                >= -model.inherited.mpc.constraint_tolerance
            ).all()
        ),
        "maximum_kkt_residual": float(output.kkt_residual.max()),
        "identical_repeat_status_messages": (
            repeated_table.status_message.value_counts().to_dict()
        ),
        "minimum_returned_constraint_margin": float(
            output.minimum_constraint_margin.min()
        ),
        "best_successful_alternative_start": (
            multistart_table[
                multistart_table.success.astype(bool)
            ].sort_values("objective_value").iloc[0].start_label
            if multistart_table.success.astype(bool).any()
            else None
        ),
    }
    return output, summary


def _reversal_audit(
    config: Phase7CR0Config,
    root: Path,
    phase7c: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _, _, _, model, _ = _context(config, root)
    initial_states = pd.read_csv(root / config.phase7c_initial_states_30c)
    initial = initial_states[
        initial_states.trajectory_id == config.reversal.trajectory_id
    ].iloc[0]
    trajectory = phase7c[
        (phase7c.temperature_c == config.reversal.temperature_c)
        & (phase7c.trajectory_id == config.reversal.trajectory_id)
    ]
    tables = []
    state_summaries = []
    for offset, step_index in enumerate(config.reversal.step_indices):
        state = _reconstruct_state(initial, trajectory, step_index, model)
        table, _ = _optimization_audit(
            state,
            model,
            _starts_for_state(
                state,
                model,
                config.reversal.multistart_count,
                config.reversal.multistart_seed + offset,
            ),
            config.solver.active_constraint_tolerance,
            config.solver.finite_difference_step,
        )
        table.insert(0, "step_index", step_index)
        original = trajectory[trajectory.step_index == step_index].iloc[0]
        table.insert(1, "original_raw_current_a", original.raw_current_a)
        table.insert(2, "original_applied_current_a", original.current_a)
        table.insert(3, "temperature_c", original.average_temperature_c)
        table.insert(4, "soc", original.soc)
        tables.append(table)
        state_summaries.append(
            {
                "step_index": step_index,
                "state": asdict(state),
                "original_raw_current_a": float(original.raw_current_a),
                "original_applied_current_a": float(original.current_a),
                "temperature_c": float(original.average_temperature_c),
                "terminal_voltage_v": float(original.terminal_voltage_v),
                "success_fraction": float(table.success.mean()),
                "first_current_range_a": [
                    float(table.first_current_a.min()),
                    float(table.first_current_a.max()),
                ],
                "objective_range": [
                    float(table.objective_value.min()),
                    float(table.objective_value.max()),
                ],
                "active_constraint_count_range": [
                    int(table.active_constraint_count.min()),
                    int(table.active_constraint_count.max()),
                ],
            }
        )
    output = pd.concat(tables, ignore_index=True)
    original_actions = [
        item["original_raw_current_a"] for item in state_summaries
    ]
    summary = {
        "trajectory_id": config.reversal.trajectory_id,
        "states": state_summaries,
        "original_action_sequence_a": original_actions,
        "single_reversal_amplitudes_a": [
            original_actions[1] - original_actions[0],
            original_actions[2] - original_actions[1],
        ],
        "all_multistart_plans_feasible": bool(
            (
                output.minimum_constraint_margin
                >= -model.inherited.mpc.constraint_tolerance
            ).all()
        ),
        "branch_spread_max_a": float(
            output.groupby("step_index").first_current_a.apply(
                lambda values: values.max() - values.min()
            ).max()
        ),
        "original_middle_action_reproduced": bool(
            np.any(
                np.isclose(
                    output[
                        output.step_index
                        == config.reversal.step_indices[1]
                    ].first_current_a,
                    original_actions[1],
                    atol=1.0e-6,
                )
            )
        ),
    }
    middle = output[
        output.step_index == config.reversal.step_indices[1]
    ]
    default_middle = middle[
        middle.start_label == "controller_default"
    ].iloc[0]
    best_middle = middle.sort_values("objective_value").iloc[0]
    summary["middle_step_default_branch"] = {
        "first_current_a": float(default_middle.first_current_a),
        "objective_value": float(default_middle.objective_value),
        "kkt_residual": float(default_middle.kkt_residual),
        "success_flag": bool(default_middle.success),
    }
    summary["middle_step_best_branch"] = {
        "start_label": str(best_middle.start_label),
        "first_current_a": float(best_middle.first_current_a),
        "objective_value": float(best_middle.objective_value),
        "kkt_residual": float(best_middle.kkt_residual),
        "objective_improvement_fraction": float(
            (
                default_middle.objective_value
                - best_middle.objective_value
            )
            / default_middle.objective_value
        ),
    }
    return output, summary


def _plot(
    result_dir: Path,
    constant_frame: pd.DataFrame,
    oracle_frame: pd.DataFrame,
    one_step: pd.DataFrame,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1, 3, figsize=(15, 4.5), layout="constrained"
    )
    for policy, group in constant_frame.groupby("policy"):
        axes[0].plot(group.time_s, group.temperature_c, label=policy)
    axes[0].axhline(35.0, color="red", linestyle="--")
    axes[0].set(
        xlabel="Time [s]",
        ylabel="Average temperature [℃]",
        title="30 ℃ constant-current sweep",
    )
    axes[0].legend(fontsize=7, ncol=2)
    for policy, group in oracle_frame.groupby("policy"):
        axes[1].plot(group.time_s, group.temperature_c, alpha=0.75)
    axes[1].axhline(35.0, color="red", linestyle="--")
    axes[1].set(
        xlabel="Time [s]",
        ylabel="Average temperature [℃]",
        title="DFN-validated two-stage policies",
    )
    minimum = (
        one_step.groupby("trajectory_id").thermal_slew_margin_a.min()
    )
    axes[2].hist(minimum, bins=20, color="#2878B5")
    axes[2].axvline(0.0, color="red", linestyle="--")
    axes[2].set(
        xlabel="Minimum thermal–slew margin [A]",
        ylabel="Trajectories",
        title="One-step thermal feasibility",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(
        result_dir / "phase7cr0_diagnostics.png", dpi=180
    )
    plt.close(figure)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    thermal = payload["thermal_feasibility"]
    solver = payload["solver_failure_audit"]
    reversal = payload["direction_reversal_audit"]
    fastest = thermal.get("fastest_dfn_feasible_policy")
    fastest_two = thermal.get("fastest_dfn_feasible_two_stage_policy")
    braking = thermal.get("latest_braking_state_in_fastest_policy")
    conservative = thermal.get("conservative_300s_braking_boundary")
    default_branch = reversal["middle_step_default_branch"]
    best_branch = reversal["middle_step_best_branch"]
    report = f"""# Phase 7C-R0：三项独立诊断报告

## 结论

Phase 7C 的原始失败判定保持不变。本阶段没有重新训练 ANN、没有生成
ANN 教师数据，也没有运行240条多温度 ANN 轨迹。

## 1. 30 ℃热可行域

- 35 ℃以下到达80% SOC是否可行：**{'是' if thermal['feasible_strategy_exists'] else '否'}**
- 最快DFN验证可行时间：{fastest['charge_time_s'] if fastest else '无'} s
- 最快策略最高平均温度：{fastest['maximum_temperature_c'] if fastest else '无'} ℃
- 最快两段式策略：{fastest_two['policy'] if fastest_two else '无'}
- 两段式首次降流：{braking if braking else '无'}
- 300 s最大斜率制动的保守最迟边界：{conservative if conservative else '无'}
- 一步热限制空区间总数：{thermal['one_step_empty_interval_count']}
- 出现空区间的轨迹数：{thermal['one_step_empty_trajectory_count']}

本阶段的两段式策略仅用于证明热约束问题是否存在可行解和估计制动提前量，
不是最终监督层，也不构成新的独立确认结果。

恒流3 A以2100 s到达目标，最高34.680 ℃；4 A以1600 s到达但升至
37.052 ℃。离线搜索后由热DFN验证的最快两段式策略为1895 s，最高温度
不超过34.962 ℃。在Phase 7C原失败轨迹上，一步热限制产生大量热—斜率
空区间，说明5 s一步监督发现风险过晚，R1必须采用60–300 s短时域预测。

## 2. 15 ℃求解失败

- 原记录：`optimizer_success=False`、`prediction_feasible=True`
- 完全相同状态重复成功/失败：{solver['identical_repeat_success_count']}/
  {solver['identical_repeat_failure_count']}
- 多起点成功/失败：{solver['multistart_success_count']}/
  {solver['multistart_failure_count']}
- 所有多起点返回计划满足预注册可行容差：
  {solver['all_returned_plans_strictly_feasible']}
- 最大数值KKT残差：{solver['maximum_kkt_residual']:.3e}
- 完全相同起点终止信息：{solver['identical_repeat_status_messages']}
- 可行返回计划最小约束余量：{solver['minimum_returned_constraint_margin']:.3e}
- 建议预注册的单次替代起点：{solver['best_successful_alternative_start']}

完全相同默认起点20次均以SLSQP状态8停止，但返回动作一致、约束残差约
1e-8 A/V量级，并且替代起点14/15成功收敛到同一第一动作。因此可以在
R1开发合同中预注册“一次固定替代起点重试”，但Phase 7C原失败仍保留。

## 3. 30 ℃方向反转

- 原始动作序列：{reversal['original_action_sequence_a']} A
- 相邻动作变化：{reversal['single_reversal_amplitudes_a']} A
- 同状态多起点第一动作最大分支宽度：
  {reversal['branch_spread_max_a']:.6f} A
- 所有多起点计划满足可行容差：
  {reversal['all_multistart_plans_feasible']}
- 中间步默认分支：{default_branch}
- 中间步最佳分支：{best_branch}

第98步默认起点虽然返回`success=True`，但其目标函数和KKT残差均明显差于
替代起点；替代分支把第一动作由3.420 A恢复到5.389 A。该反转主要来自
求解器分支/局部数值停滞，而不是热约束切换。事件继续保留为Phase 7C
异常反转，不回改原门槛。

## 阶段决策

{payload['decision']['conclusion']}
"""
    path.write_text(report, encoding="utf-8")


def run_phase7cr0(
    config: Phase7CR0Config, root: Path
) -> dict[str, Any]:
    verification = verify_frozen_artifacts(config, root)
    data_dir = root / config.data_directory
    result_dir = root / config.result_directory
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    phase7c = pd.read_csv(root / config.phase7c_trajectories)

    constant_frame_path = data_dir / "constant_current_trajectories.csv"
    constant_summary_path = data_dir / "constant_current_metrics.csv"
    if constant_frame_path.exists() and constant_summary_path.exists():
        constant_frame = pd.read_csv(constant_frame_path)
        constant_summary = pd.read_csv(constant_summary_path)
    else:
        constant_frame, constant_summary = _run_constant_current_sweep(
            config, root
        )
    fit = _fit_thermal_surrogate(
        constant_frame, config.thermal.ambient_temperature_c
    )
    search_path = data_dir / "offline_oracle_search.csv"
    if search_path.exists():
        search = pd.read_csv(search_path)
    else:
        search = _oracle_search(config, root, fit)
    oracle_frame_path = data_dir / "validated_oracle_trajectories.csv"
    oracle_summary_path = data_dir / "validated_oracle_metrics.csv"
    if oracle_frame_path.exists() and oracle_summary_path.exists():
        oracle_frame = pd.read_csv(oracle_frame_path)
        oracle_summary = pd.read_csv(oracle_summary_path)
    else:
        oracle_frame, oracle_summary = _validate_oracle_candidates(
            config, root, search
        )
    one_step = _thermal_one_step_audit(config, phase7c, fit)
    solver_table, solver_summary = _solver_failure_audit(
        config, root, phase7c
    )
    reversal_table, reversal_summary = _reversal_audit(
        config, root, phase7c
    )

    constant_frame.to_csv(
        data_dir / "constant_current_trajectories.csv", index=False
    )
    constant_summary.to_csv(
        data_dir / "constant_current_metrics.csv", index=False
    )
    search.to_csv(data_dir / "offline_oracle_search.csv", index=False)
    oracle_frame.to_csv(
        data_dir / "validated_oracle_trajectories.csv", index=False
    )
    oracle_summary.to_csv(
        data_dir / "validated_oracle_metrics.csv", index=False
    )
    one_step.to_csv(
        data_dir / "one_step_thermal_feasibility.csv", index=False
    )
    solver_table.to_csv(
        data_dir / "solver_failure_multistart.csv", index=False
    )
    reversal_table.to_csv(
        data_dir / "direction_reversal_multistart.csv", index=False
    )

    feasible_oracle = oracle_summary[
        oracle_summary.thermally_feasible.astype(bool)
    ].sort_values("charge_time_s")
    feasible_constant = constant_summary[
        constant_summary.thermally_feasible.astype(bool)
    ].sort_values("charge_time_s")
    all_feasible = pd.concat(
        (feasible_constant, feasible_oracle),
        ignore_index=True,
        sort=False,
    ).sort_values("charge_time_s")
    fastest = (
        all_feasible.iloc[0].replace({np.nan: None}).to_dict()
        if len(all_feasible)
        else None
    )
    fastest_two_stage = (
        feasible_oracle.iloc[0].replace({np.nan: None}).to_dict()
        if len(feasible_oracle)
        else None
    )
    empty = one_step[one_step.empty_thermal_slew_interval.astype(bool)]
    first_empty = (
        empty.sort_values("time_s").iloc[0].replace({np.nan: None}).to_dict()
        if len(empty)
        else None
    )
    if fastest_two_stage is not None:
        fastest_frame = pd.concat(
            (constant_frame, oracle_frame), ignore_index=True
        )
        selected = fastest_frame[
            fastest_frame.policy == fastest_two_stage["policy"]
        ]
        switch_rows = selected[
            selected.candidate_current_a
            < selected.candidate_current_a.max() - 1.0e-9
        ]
        if len(switch_rows):
            switch = switch_rows.iloc[0]
            latest_braking = {
                "time_s": float(switch.time_s),
                "soc": float(switch.soc),
                "temperature_c": float(switch.temperature_c),
                "current_a": float(switch.current_a),
            }
        else:
            latest_braking = None
    else:
        latest_braking = None
    braking_boundaries = []
    for trajectory_id, group in one_step.groupby("trajectory_id"):
        ordered = group.sort_values("step_index")
        too_late = ordered[ordered.max_braking_too_late.astype(bool)]
        if too_late.empty:
            continue
        first_index = int(too_late.index[0])
        prior = ordered.loc[ordered.index < first_index].tail(1)
        reference = prior.iloc[0] if len(prior) else too_late.iloc[0]
        braking_boundaries.append(
            {
                "trajectory_id": trajectory_id,
                "latest_safe_time_s": float(reference.time_s),
                "latest_safe_soc": float(reference.soc),
                "latest_safe_temperature_c": float(
                    reference.temperature_c
                ),
                "next_step_too_late": True,
            }
        )
    conservative_braking_boundary = (
        min(braking_boundaries, key=lambda item: item["latest_safe_time_s"])
        if braking_boundaries
        else None
    )
    feasible_strategy_exists = fastest is not None
    repeated_is_deterministic = (
        solver_summary["identical_repeat_unique_first_currents"] == 1
        and (
            solver_summary["identical_repeat_success_count"] == 0
            or solver_summary["identical_repeat_failure_count"] == 0
        )
    )
    retry_rule_candidate = bool(
        solver_summary["all_returned_plans_strictly_feasible"]
        and solver_summary["multistart_success_count"] > 0
    )
    reversal_is_branch_sensitive = (
        reversal_summary["branch_spread_max_a"] > 0.05
    )
    proceed_to_r1 = feasible_strategy_exists
    conclusion = (
        "30 ℃热约束下存在DFN验证可行策略，允许在新的开发集上设计R1短时域"
        "热安全监督层；Phase 7C确认集只作永久失败回归。"
        if proceed_to_r1
        else "30 ℃热约束下尚未找到可行策略，停止R1并优先调整冷却、温度上限、"
        "目标SOC或允许时间。"
    )
    payload = {
        "study_name": config.study_name,
        "configuration": asdict(config),
        "frozen_artifact_verification": verification,
        "thermal_feasibility": {
            "constant_current_metrics": constant_summary.to_dict(
                orient="records"
            ),
            "thermal_surrogate_fit": fit,
            "validated_oracle_metrics": oracle_summary.to_dict(
                orient="records"
            ),
            "feasible_strategy_exists": feasible_strategy_exists,
            "fastest_dfn_feasible_policy": fastest,
            "fastest_dfn_feasible_two_stage_policy": fastest_two_stage,
            "latest_braking_state_in_fastest_policy": latest_braking,
            "conservative_300s_braking_boundary": (
                conservative_braking_boundary
            ),
            "one_step_empty_interval_count": len(empty),
            "one_step_empty_trajectory_count": int(
                empty.trajectory_id.nunique()
            ),
            "first_one_step_empty_interval": first_empty,
        },
        "solver_failure_audit": {
            **solver_summary,
            "identical_replay_is_deterministic": repeated_is_deterministic,
            "finite_retry_rule_is_candidate": retry_rule_candidate,
        },
        "direction_reversal_audit": {
            **reversal_summary,
            "branch_sensitive": reversal_is_branch_sensitive,
        },
        "decision": {
            "proceed_to_phase7cr1_development": proceed_to_r1,
            "phase7c_original_failure_unchanged": True,
            "run_ann": False,
            "conclusion": conclusion,
        },
        "status": "completed",
        "success": True,
    }
    (result_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    _plot(result_dir, constant_frame, oracle_frame, one_step)
    _write_report(result_dir / "PHASE7C-R0_中文诊断报告.md", payload)
    return payload


def refine_low3_candidates(
    config: Phase7CR0Config, root: Path
) -> dict[str, Any]:
    """只补充低段3 A的两段式DFN候选，不重跑恒流基线。"""
    verify_frozen_artifacts(config, root)
    data_dir = root / config.data_directory
    constant_frame = pd.read_csv(
        data_dir / "constant_current_trajectories.csv"
    )
    fit = _fit_thermal_surrogate(
        constant_frame, config.thermal.ambient_temperature_c
    )
    search = _oracle_search(config, root, fit)
    search.to_csv(data_dir / "offline_oracle_search.csv", index=False)
    low3 = search[
        (search.low_current_a == 3.0)
        & search.predicted_feasible.astype(bool)
    ].drop_duplicates(subset=["high_current_a"]).head(4)
    new_frame, new_summary = _validate_oracle_candidates(
        config, root, low3
    )
    frame_path = data_dir / "validated_oracle_trajectories.csv"
    summary_path = data_dir / "validated_oracle_metrics.csv"
    prior_frame = pd.read_csv(frame_path)
    prior_summary = pd.read_csv(summary_path)
    combined_frame = pd.concat(
        (prior_frame, new_frame), ignore_index=True
    ).drop_duplicates(subset=["policy", "step_index"], keep="last")
    combined_summary = pd.concat(
        (prior_summary, new_summary), ignore_index=True
    ).drop_duplicates(subset=["policy"], keep="last")
    combined_frame.to_csv(frame_path, index=False)
    combined_summary.to_csv(summary_path, index=False)
    return {
        "validated_new_candidates": len(new_summary),
        "fastest_new_charge_time_s": float(
            new_summary.charge_time_s.min()
        ),
        "all_new_candidates_feasible": bool(
            new_summary.thermally_feasible.astype(bool).all()
        ),
    }
