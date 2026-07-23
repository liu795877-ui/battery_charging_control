"""Phase 7C-R2：分温度电压裕量与R1热监督层组合验证。"""

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

from .phase7a_level3_model import Level3State
from .phase7b1b_config import load_phase7b1b_config
from .phase7b1b_runner import _load_context, _maximum_safe_current
from .phase7c_runner import Chen2020ThermalDFN, _van_der_corput
from .phase7cr1_config import load_phase7cr1_config
from .phase7cr1_runner import _solve_teacher, _sustained_oscillation_count
from .phase7cr2_config import Phase7CR2Config


def _sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".py", ".yaml", ".yml"}:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def verify_frozen_artifacts(
    config: Phase7CR2Config, root: Path
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
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
        raise RuntimeError(f"Phase 7C-R2 冻结工件不匹配：{failures}")
    r1 = json.loads(
        (
            root
            / "outputs/phase7cr1_thermal_supervisor/metrics.json"
        ).read_text(encoding="utf-8")
    )
    if not r1["success"] or r1["decision"]["run_ann"]:
        raise RuntimeError("R1未严格通过或ANN边界被改变。")
    return records


def _context(config: Phase7CR2Config, root: Path):
    r1 = load_phase7cr1_config(root / config.sources["phase7cr1_config"])
    b1 = load_phase7b1b_config(root / config.sources["phase7b1b_config"])
    level3, inherited, model, _, phase7b0 = _load_context(b1, root)
    return r1, b1, level3, inherited, model, phase7b0


def _thermal_step(
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


def _thermal_peak(
    current_a: float,
    temperature_c: float,
    ambient_temperature_c: float,
    r1: Any,
    braking: bool,
) -> float:
    predicted_temperature = float(temperature_c)
    predicted_current = float(current_a)
    maximum = -np.inf
    for step in range(int(r1.thermal["prediction_horizon_steps"])):
        if braking and step > 0:
            predicted_current = max(
                r1.thermal["braking_floor_current_a"],
                predicted_current - 2.0,
            )
        predicted_temperature = _thermal_step(
            predicted_temperature,
            predicted_current,
            ambient_temperature_c,
            r1.thermal["surrogate"],
        )
        maximum = max(maximum, predicted_temperature)
    return float(maximum)


def _thermal_current_limit(
    temperature_c: float,
    ambient_temperature_c: float,
    search_upper_a: float,
    r1: Any,
    braking: bool,
) -> tuple[float, float]:
    limit = (
        r1.thermal["maximum_average_temperature_c"]
        - r1.thermal["temperature_guard_c"]
    )

    def peak(current_a: float) -> float:
        return _thermal_peak(
            current_a,
            temperature_c,
            ambient_temperature_c,
            r1,
            braking,
        )

    if search_upper_a < 0.0 or peak(0.0) > limit:
        return -1.0, peak(0.0)
    if peak(search_upper_a) <= limit:
        return float(search_upper_a), peak(search_upper_a)
    lower, upper = 0.0, float(search_upper_a)
    tolerance = r1.thermal["current_search_tolerance_a"]
    while upper - lower > tolerance:
        current = 0.5 * (lower + upper)
        if peak(current) <= limit:
            lower = current
        else:
            upper = current
    return lower, peak(lower)


def _existing_state_values(config: Phase7CR2Config, root: Path) -> np.ndarray:
    frames = [
        pd.read_csv(root / path)
        for path in config.sources["legacy_phase7c_states"].values()
    ]
    frames.append(
        pd.read_csv(root / config.sources["legacy_phase7cr1_states"])
    )
    columns = [
        "initial_soc",
        "initial_polarization_1_v",
        "initial_polarization_2_v",
        "initial_previous_current_a",
    ]
    return pd.concat(frames, ignore_index=True)[columns].to_numpy(float)


def _candidate_state(
    index: int, bounds: dict[str, list[float]]
) -> tuple[float, float, float, float]:
    def scale(name: str, base: int) -> float:
        lower, upper = bounds[name]
        return lower + (upper - lower) * _van_der_corput(index, base)

    return (
        scale("soc", 2),
        scale("v1_v", 3),
        scale("v2_v", 5),
        scale("previous_current_a", 7),
    )


def _build_role_states(
    config: Phase7CR2Config,
    root: Path,
    role: str,
    count: int,
    start_index: int,
) -> pd.DataFrame:
    r1, _, level3, _, model, _ = _context(config, root)
    existing = _existing_state_values(config, root)
    general_count = int(round(count * config.datasets["general_fraction"]))
    strata = [
        ("general", general_count, config.datasets["general_bounds"]),
        (
            "high_risk",
            count - general_count,
            config.datasets["high_risk_bounds"],
        ),
    ]
    records = []
    cursor = int(start_index)
    voltage_limit = (
        model.inherited.mpc.terminal_voltage_max_v
        - config.datasets["initial_voltage_margin_v"]
    )
    for stratum, required, bounds in strata:
        accepted = 0
        while accepted < required:
            soc, v1, v2, previous = _candidate_state(cursor, bounds)
            used = cursor
            cursor += 1
            values = np.asarray([soc, v1, v2, previous])
            if np.any(
                np.all(
                    np.isclose(values, existing, atol=1.0e-14),
                    axis=1,
                )
            ):
                continue
            state = Level3State(soc, v1, v2, previous)
            minimum_current = max(
                0.0,
                previous - level3.constraint.maximum_current_step_a,
            )
            if model.terminal_voltage(state, minimum_current) > voltage_limit:
                continue
            result, _ = _solve_teacher(state, model, r1)
            if not result.prediction_feasible:
                continue
            records.append(
                {
                    "role": role,
                    "base_state_id": f"{role}_{len(records):03d}",
                    "risk_stratum": stratum,
                    "initial_soc": soc,
                    "initial_polarization_1_v": v1,
                    "initial_polarization_2_v": v2,
                    "initial_previous_current_a": previous,
                    "design_candidate_index": used,
                    "design_seed": config.datasets["design_seed"],
                }
            )
            existing = np.vstack([existing, values])
            accepted += 1
    base = pd.DataFrame(records)
    expanded = []
    for temperature_c in config.datasets["temperatures_c"]:
        token = int(temperature_c)
        frame = base.copy()
        frame.insert(
            0,
            "trajectory_id",
            [
                f"phase7cr2_{role}_{token}c_{i:03d}"
                for i in range(len(frame))
            ],
        )
        frame.insert(2, "ambient_temperature_c", temperature_c)
        frame.insert(3, "initial_temperature_c", temperature_c)
        expanded.append(frame)
    return pd.concat(expanded, ignore_index=True)


def freeze_new_states(
    config: Phase7CR2Config, root: Path
) -> dict[str, Any]:
    data_dir = root / config.output["data_directory"]
    data_dir.mkdir(parents=True, exist_ok=True)
    development = _build_role_states(
        config,
        root,
        "development",
        int(config.datasets["development_count_per_temperature"]),
        int(config.datasets["development_start_index"]),
    )
    internal = _build_role_states(
        config,
        root,
        "internal_validation",
        int(config.datasets["internal_validation_count_per_temperature"]),
        int(config.datasets["internal_validation_start_index"]),
    )
    files = {}
    for name, frame in (
        ("development_initial_states.csv", development),
        ("internal_validation_initial_states.csv", internal),
    ):
        path = data_dir / name
        frame.to_csv(path, index=False)
        files[name] = {
            "sha256": _sha256(path),
            "trajectory_count": len(frame),
        }
    freeze = {
        "frozen_before_rollout": True,
        "development_only": True,
        "not_r3_confirmation": True,
        "not_ann_teacher_data": True,
        "ann_execution_authorized": False,
        "files": files,
        "frozen_artifact_verification": verify_frozen_artifacts(
            config, root
        ),
    }
    (data_dir / "initial_state_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return freeze


def _rollout(
    config: Phase7CR2Config,
    root: Path,
    initial: dict[str, Any],
    role: str,
    guard_v: float,
) -> pd.DataFrame:
    r1, b1, level3, inherited, model, phase7b0 = _context(config, root)
    ambient = float(initial["ambient_temperature_c"])
    state = Level3State(
        float(initial["initial_soc"]),
        float(initial["initial_polarization_1_v"]),
        float(initial["initial_polarization_2_v"]),
        float(initial["initial_previous_current_a"]),
    )
    plant = Chen2020ThermalDFN(
        phase7b0.dfn.parameter_set,
        ambient,
        phase7b0.dfn.upper_voltage_cutoff_v,
        state.soc,
        level3.model.sample_period_s,
        "lumped",
    )
    temperature_c = float(initial["initial_temperature_c"])
    residual_v = 0.0
    lower_bound, upper_bound = inherited.mpc.current_bounds_a
    maximum_step = level3.constraint.maximum_current_step_a
    rows = []
    for step in range(int(config.datasets["maximum_steps"])):
        teacher_started = perf_counter()
        result, teacher = _solve_teacher(state, model, r1)
        teacher_time_s = perf_counter() - teacher_started
        candidate = float(result.current_a)
        slew_lower = max(lower_bound, state.previous_current_a - maximum_step)
        slew_upper = min(upper_bound, state.previous_current_a + maximum_step)

        supervisor_started = perf_counter()
        voltage_max = _maximum_safe_current(
            state, residual_v + guard_v, b1, model
        )
        thermal_search_upper = min(slew_upper, voltage_max)
        constant_max, constant_peak = _thermal_current_limit(
            temperature_c,
            ambient,
            thermal_search_upper,
            r1,
            braking=False,
        )
        braking_max, braking_peak = _thermal_current_limit(
            temperature_c,
            ambient,
            thermal_search_upper,
            r1,
            braking=True,
        )
        thermal_recovery = (
            constant_max
            < slew_lower - r1.thermal["empty_interval_tolerance_a"]
        )
        if thermal_recovery:
            thermal_max = min(braking_max, slew_lower)
            thermal_peak = braking_peak
        else:
            thermal_max = constant_max
            thermal_peak = constant_peak
        final_upper = min(slew_upper, voltage_max, thermal_max)
        empty_voltage = (
            voltage_max
            < slew_lower - r1.thermal["empty_interval_tolerance_a"]
        )
        empty_thermal = (
            thermal_max
            < slew_lower - r1.thermal["empty_interval_tolerance_a"]
        )
        empty = empty_voltage or empty_thermal
        supervisor_time_s = perf_counter() - supervisor_started

        base = {
            "role": role,
            "trajectory_id": initial["trajectory_id"],
            "ambient_temperature_c": ambient,
            "risk_stratum": initial.get("risk_stratum", "legacy"),
            "step_index": step,
            "time_s": (step + 1) * level3.model.sample_period_s,
            "soc": state.soc,
            "temperature_c": temperature_c,
            "previous_current_a": state.previous_current_a,
            "candidate_current_a": candidate,
            "slew_lower_a": slew_lower,
            "slew_upper_a": slew_upper,
            "guard_v": guard_v,
            "voltage_safe_current_max_a": voltage_max,
            "constant_thermal_safe_current_max_a": constant_max,
            "braking_thermal_safe_current_max_a": braking_max,
            "thermal_safe_current_max_a": thermal_max,
            "predicted_300s_peak_temperature_c": thermal_peak,
            "thermal_recovery_active": thermal_recovery,
            "final_upper_a": final_upper,
            "empty_voltage_slew_interval": empty_voltage,
            "empty_thermal_slew_interval": empty_thermal,
            "empty_final_interval": empty,
            "teacher_time_s": teacher_time_s,
            "supervisor_time_s": supervisor_time_s,
            **teacher,
        }
        if empty:
            rows.append(
                {
                    **base,
                    "current_a": np.nan,
                    "current_step_a": np.nan,
                    "next_soc": state.soc,
                    "next_temperature_c": temperature_c,
                    "terminal_voltage_v": np.nan,
                    "predicted_voltage_v": np.nan,
                    "voltage_residual_before_v": residual_v,
                    "voltage_residual_after_v": np.nan,
                    "positive_residual_growth_v": np.nan,
                    "guard_exceeded": False,
                    "voltage_intervened": False,
                    "thermal_intervened": False,
                    "both_layers_active": False,
                    "voltage_current_correction_a": np.nan,
                    "thermal_current_correction_a": np.nan,
                    "optimizer_success": result.optimizer_success,
                    "prediction_feasible": result.prediction_feasible,
                }
            )
            break

        pre_voltage = min(candidate, slew_upper)
        pre_thermal = min(candidate, slew_upper, voltage_max)
        current = float(
            np.clip(min(candidate, final_upper), slew_lower, final_upper)
        )
        voltage_active = voltage_max < pre_voltage - 1.0e-9
        thermal_active = thermal_max < pre_thermal - 1.0e-9
        predicted = model.step(state, current)
        predicted_voltage = model.terminal_voltage(predicted, current)
        measurement = plant.step(current)
        new_residual = (
            float(measurement["terminal_voltage_v"]) - predicted_voltage
        )
        positive_growth = max(0.0, new_residual - residual_v)
        rows.append(
            {
                **base,
                "current_a": current,
                "current_step_a": abs(current - state.previous_current_a),
                "next_soc": measurement["soc"],
                "next_temperature_c": measurement[
                    "average_temperature_c"
                ],
                "terminal_voltage_v": measurement["terminal_voltage_v"],
                "predicted_voltage_v": predicted_voltage,
                "voltage_residual_before_v": residual_v,
                "voltage_residual_after_v": new_residual,
                "positive_residual_growth_v": positive_growth,
                "guard_exceeded": (
                    positive_growth
                    > guard_v
                    + config.voltage["residual_growth_tolerance_v"]
                ),
                "voltage_intervened": voltage_active,
                "thermal_intervened": thermal_active,
                "both_layers_active": voltage_active and thermal_active,
                "voltage_current_correction_a": (
                    min(candidate, slew_upper, voltage_max) - pre_voltage
                ),
                "thermal_current_correction_a": current - pre_thermal,
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
        residual_v = new_residual
        if state.soc >= config.datasets["target_soc"]:
            break
    return pd.DataFrame(rows)


def _worker(
    config: Phase7CR2Config,
    root_text: str,
    initial: dict[str, Any],
    role: str,
    guard_v: float,
    path_text: str,
) -> str:
    frame = _rollout(
        config, Path(root_text), initial, role, guard_v
    )
    frame.to_csv(path_text, index=False)
    return path_text


def _run_rows(
    config: Phase7CR2Config,
    root: Path,
    rows: list[dict[str, Any]],
    role: str,
    guard_by_temperature: dict[str, float],
    resume: bool,
) -> pd.DataFrame:
    run_dir = (
        root / config.output["data_directory"] / "runs" / role
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    pending = []
    for row in rows:
        temperature = str(int(float(row["ambient_temperature_c"])))
        guard = float(guard_by_temperature[temperature])
        path = run_dir / f"{row['trajectory_id']}.csv"
        paths.append(path)
        if not (resume and path.exists()):
            pending.append((row, guard, path))
    if pending:
        with ProcessPoolExecutor(
            max_workers=int(config.datasets["maximum_workers"])
        ) as executor:
            futures = {
                executor.submit(
                    _worker,
                    config,
                    str(root),
                    row,
                    role,
                    guard,
                    str(path),
                ): path
                for row, guard, path in pending
            }
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                print(
                    f"[Phase 7C-R2:{role}] "
                    f"{completed}/{len(pending)}",
                    flush=True,
                )
    return pd.concat(
        [pd.read_csv(path) for path in paths], ignore_index=True
    )


def _new_state_rows(
    config: Phase7CR2Config, root: Path, role: str
) -> list[dict[str, Any]]:
    name = (
        "development_initial_states.csv"
        if role == "development"
        else "internal_validation_initial_states.csv"
    )
    return pd.read_csv(
        root / config.output["data_directory"] / name
    ).to_dict(orient="records")


def _legacy_rows(
    config: Phase7CR2Config, root: Path, role: str
) -> list[dict[str, Any]]:
    rows = []
    if role == "legacy_phase7c":
        for temperature, relative in config.sources[
            "legacy_phase7c_states"
        ].items():
            frame = pd.read_csv(root / relative)
            frame["ambient_temperature_c"] = float(temperature)
            frame["initial_temperature_c"] = float(temperature)
            frame["trajectory_id"] = [
                f"r2_legacy7c_{temperature}c_{i:03d}"
                for i in range(len(frame))
            ]
            rows.extend(frame.to_dict(orient="records"))
    elif role == "legacy_phase7cr1":
        frame = pd.read_csv(
            root / config.sources["legacy_phase7cr1_states"]
        )
        frame["trajectory_id"] = [
            f"r2_legacy7cr1_30c_{i:03d}" for i in range(len(frame))
        ]
        rows.extend(frame.to_dict(orient="records"))
    else:
        raise ValueError(role)
    return rows


def _derive_guards(
    config: Phase7CR2Config,
    audit: pd.DataFrame,
    root: Path,
) -> dict[str, Any]:
    guards = {
        "25": float(config.voltage["original_25c_guard_v"])
    }
    records = {}
    for temperature in config.datasets["temperatures_c"]:
        token = str(int(temperature))
        group = audit[audit.ambient_temperature_c == temperature]
        observed = float(group.positive_residual_growth_v.max())
        legacy = float(config.voltage["legacy_minimum_v"][int(temperature)])
        guard = (
            max(
                float(config.voltage["original_25c_guard_v"]),
                legacy,
                observed,
            )
            + float(config.voltage["development_margin_v"])
        )
        guards[token] = guard
        records[token] = {
            "temperature_c": temperature,
            "development_maximum_positive_growth_v": observed,
            "legacy_minimum_v": legacy,
            "engineering_margin_v": config.voltage[
                "development_margin_v"
            ],
            "final_guard_v": guard,
        }
    payload = {
        "formula": "max(original_25c, legacy_minimum, development_max) + margin",
        "guards_v": guards,
        "temperature_records": records,
        "only_maintain_or_increase": True,
        "25c_unchanged": True,
    }
    path = (
        root
        / config.output["data_directory"]
        / "derived_voltage_guards.json"
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _trajectory_metrics(
    config: Phase7CR2Config, frame: pd.DataFrame
) -> pd.DataFrame:
    records = []
    for (role, trajectory_id), group in frame.groupby(
        ["role", "trajectory_id"]
    ):
        valid = group[group.current_a.notna()]
        records.append(
            {
                "role": role,
                "trajectory_id": trajectory_id,
                "ambient_temperature_c": float(
                    group.ambient_temperature_c.iloc[0]
                ),
                "target_reached": bool(
                    len(valid)
                    and valid.next_soc.iloc[-1]
                    >= config.datasets["target_soc"]
                ),
                "charge_time_s": (
                    float(valid.time_s.iloc[-1]) if len(valid) else np.nan
                ),
                "maximum_voltage_v": float(
                    valid.terminal_voltage_v.max()
                ),
                "maximum_temperature_c": float(
                    valid.next_temperature_c.max()
                ),
                "minimum_current_a": float(valid.current_a.min()),
                "maximum_current_a": float(valid.current_a.max()),
                "maximum_current_step_a": float(
                    valid.current_step_a.max()
                ),
                "guard_exceedance_count": int(
                    valid.guard_exceeded.astype(bool).sum()
                ),
                "maximum_positive_residual_growth_v": float(
                    valid.positive_residual_growth_v.max()
                ),
                "empty_voltage_slew_count": int(
                    group.empty_voltage_slew_interval.astype(bool).sum()
                ),
                "empty_thermal_slew_count": int(
                    group.empty_thermal_slew_interval.astype(bool).sum()
                ),
                "solver_failure_count": int(
                    (~group.optimizer_success.astype(bool)).sum()
                ),
                "prediction_infeasible_count": int(
                    (~group.prediction_feasible.astype(bool)).sum()
                ),
                "sustained_oscillation_count": (
                    _sustained_oscillation_count(
                        valid.current_a.to_numpy(float)
                    )
                ),
                "voltage_intervention_fraction": float(
                    valid.voltage_intervened.astype(bool).mean()
                ),
                "thermal_intervention_fraction": float(
                    valid.thermal_intervened.astype(bool).mean()
                ),
                "both_layers_fraction": float(
                    valid.both_layers_active.astype(bool).mean()
                ),
                "maximum_voltage_correction_a": float(
                    valid.voltage_current_correction_a.abs().max()
                ),
                "maximum_thermal_correction_a": float(
                    valid.thermal_current_correction_a.abs().max()
                ),
                "teacher_retry_count": int(
                    group.teacher_retry_triggered.astype(bool).sum()
                ),
                "alternative_selected_count": int(
                    group.alternative_selected.astype(bool).sum()
                ),
                "total_thermal_supervisor_time_s": float(
                    valid.supervisor_time_s.sum()
                ),
            }
        )
    return pd.DataFrame(records)


def _checks(
    config: Phase7CR2Config, metrics: pd.DataFrame
) -> dict[str, bool]:
    gates = config.gates
    tolerance = gates["numerical_tolerance"]
    return {
        "maximum_voltage": bool(
            (
                metrics.maximum_voltage_v
                <= gates["maximum_voltage_v"]
            ).all()
        ),
        "maximum_temperature": bool(
            (
                metrics.maximum_temperature_c
                <= gates["maximum_average_temperature_c"] + tolerance
            ).all()
        ),
        "zero_guard_exceedance": bool(
            metrics.guard_exceedance_count.sum()
            <= gates["maximum_guard_exceedance_count"]
        ),
        "current_bounds": bool(
            (
                metrics.minimum_current_a
                >= gates["minimum_current_a"] - tolerance
            ).all()
            and (
                metrics.maximum_current_a
                <= gates["maximum_current_a"] + tolerance
            ).all()
        ),
        "slew_bound": bool(
            (
                metrics.maximum_current_step_a
                <= gates["maximum_current_step_a"] + tolerance
            ).all()
        ),
        "zero_voltage_slew_empty": bool(
            metrics.empty_voltage_slew_count.sum()
            <= gates["maximum_empty_interval_count"]
        ),
        "zero_thermal_slew_empty": bool(
            metrics.empty_thermal_slew_count.sum()
            <= gates["maximum_empty_interval_count"]
        ),
        "zero_solver_failure": bool(
            metrics.solver_failure_count.sum()
            <= gates["maximum_solver_failure_count"]
        ),
        "zero_prediction_infeasible": bool(
            metrics.prediction_infeasible_count.sum()
            <= gates["maximum_prediction_infeasible_count"]
        ),
        "zero_sustained_oscillation": bool(
            metrics.sustained_oscillation_count.sum()
            <= gates["maximum_sustained_oscillation_count"]
        ),
        "target_reach_100_percent": bool(
            metrics.target_reached.mean()
            >= gates["minimum_target_reach_fraction"]
        ),
    }


def _summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    return {
        "trajectory_count": len(group),
        "target_reach_fraction": float(group.target_reached.mean()),
        "maximum_voltage_v": float(group.maximum_voltage_v.max()),
        "maximum_temperature_c": float(group.maximum_temperature_c.max()),
        "maximum_current_step_a": float(
            group.maximum_current_step_a.max()
        ),
        "maximum_positive_residual_growth_v": float(
            group.maximum_positive_residual_growth_v.max()
        ),
        "guard_exceedance_count": int(
            group.guard_exceedance_count.sum()
        ),
        "empty_voltage_slew_count": int(
            group.empty_voltage_slew_count.sum()
        ),
        "empty_thermal_slew_count": int(
            group.empty_thermal_slew_count.sum()
        ),
        "solver_failure_count": int(group.solver_failure_count.sum()),
        "prediction_infeasible_count": int(
            group.prediction_infeasible_count.sum()
        ),
        "sustained_oscillation_count": int(
            group.sustained_oscillation_count.sum()
        ),
        "charge_time_range_s": [
            float(group.charge_time_s.min()),
            float(group.charge_time_s.max()),
        ],
        "voltage_intervention_fraction_range": [
            float(group.voltage_intervention_fraction.min()),
            float(group.voltage_intervention_fraction.max()),
        ],
        "thermal_intervention_fraction_range": [
            float(group.thermal_intervention_fraction.min()),
            float(group.thermal_intervention_fraction.max()),
        ],
        "both_layers_fraction_range": [
            float(group.both_layers_fraction.min()),
            float(group.both_layers_fraction.max()),
        ],
        "maximum_voltage_correction_a": float(
            group.maximum_voltage_correction_a.max()
        ),
        "maximum_thermal_correction_a": float(
            group.maximum_thermal_correction_a.max()
        ),
        "teacher_retry_count": int(group.teacher_retry_count.sum()),
        "alternative_selected_count": int(
            group.alternative_selected_count.sum()
        ),
        "total_thermal_supervisor_time_s": float(
            group.total_thermal_supervisor_time_s.sum()
        ),
    }


def _analyze(
    config: Phase7CR2Config,
    root: Path,
    audit: pd.DataFrame,
    final: pd.DataFrame,
    guard_payload: dict[str, Any],
) -> dict[str, Any]:
    data_dir = root / config.output["data_directory"]
    result_dir = root / config.output["result_directory"]
    result_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(data_dir / "development_guard_audit.csv", index=False)
    final.to_csv(data_dir / "combined_validation_trajectories.csv", index=False)
    metrics = _trajectory_metrics(config, final)
    metrics.to_csv(data_dir / "trajectory_metrics.csv", index=False)
    role_results = {}
    all_success = True
    for role, role_group in metrics.groupby("role"):
        temperature_results = {}
        for temperature, group in role_group.groupby(
            "ambient_temperature_c"
        ):
            checks = _checks(config, group)
            temperature_results[str(int(temperature))] = {
                "summary": _summarize_group(group),
                "checks": checks,
                "success": bool(all(checks.values())),
            }
            all_success = all_success and bool(all(checks.values()))
        role_results[role] = temperature_results
    guard_columns = [
        "role",
        "trajectory_id",
        "ambient_temperature_c",
        "step_index",
        "soc",
        "temperature_c",
        "previous_current_a",
        "current_a",
        "positive_residual_growth_v",
        "guard_v",
    ]
    solver_columns = [
        "role",
        "trajectory_id",
        "ambient_temperature_c",
        "step_index",
        "soc",
        "temperature_c",
        "candidate_current_a",
        "current_a",
        "default_optimizer_success",
        "alternative_optimizer_success",
        "default_prediction_feasible",
        "alternative_prediction_feasible",
        "default_objective",
        "alternative_objective",
        "alternative_selected",
    ]
    guard_events = final[
        final.guard_exceeded.astype(bool)
    ][guard_columns].to_dict(orient="records")
    solver_events = final[
        ~final.optimizer_success.astype(bool)
    ][solver_columns].to_dict(orient="records")
    payload = {
        "study_name": config.study_name,
        "configuration": config.payload,
        "frozen_artifact_verification": verify_frozen_artifacts(
            config, root
        ),
        "derived_voltage_guards": guard_payload,
        "role_results": role_results,
        "global_summary": {
            "final_trajectory_count": len(metrics),
            "development_audit_trajectory_count": int(
                audit.trajectory_id.nunique()
            ),
            "target_reach_fraction": float(
                metrics.target_reached.mean()
            ),
            "maximum_voltage_v": float(
                metrics.maximum_voltage_v.max()
            ),
            "maximum_temperature_c": float(
                metrics.maximum_temperature_c.max()
            ),
            "maximum_current_step_a": float(
                metrics.maximum_current_step_a.max()
            ),
            "guard_exceedance_count": int(
                metrics.guard_exceedance_count.sum()
            ),
            "empty_voltage_slew_count": int(
                metrics.empty_voltage_slew_count.sum()
            ),
            "empty_thermal_slew_count": int(
                metrics.empty_thermal_slew_count.sum()
            ),
            "solver_failure_count": int(
                metrics.solver_failure_count.sum()
            ),
            "prediction_infeasible_count": int(
                metrics.prediction_infeasible_count.sum()
            ),
            "sustained_oscillation_count": int(
                metrics.sustained_oscillation_count.sum()
            ),
            "unified_voltage_thermal_supervisor_total_time_s": float(
                metrics.total_thermal_supervisor_time_s.sum()
            ),
            "thermal_supervisor_total_time_upper_bound_s": float(
                metrics.total_thermal_supervisor_time_s.sum()
            ),
            "timing_note": (
                "R2计时包围统一电压+热上限计算，因此该值是热监督层"
                "总时间的保守上界，不把它误报为纯热层精确计时。"
            ),
        },
        "failure_diagnosis": {
            "guard_exceedance_events": guard_events,
            "solver_failure_events": solver_events,
            "internal_validation_used_for_retuning": False,
            "teacher_selection_issue": (
                "两次失败均为默认分支成功、替代分支状态失败，但冻结的"
                "R1规则仅按预测可行与目标值选择，最终选中了失败替代分支。"
                if solver_events
                else None
            ),
            "required_next_phase": (
                "Phase 7C-R2F：预注册成功状态优先的教师分支选择规则，"
                "并使用全新的开发/内部验证集重新估计30 ℃裕量；本次"
                "内部验证轨迹永久降级为失败回归证据。"
                if not all_success
                else None
            ),
        },
        "success": all_success,
        "decision": {
            "freeze_r1_r2_architecture": all_success,
            "proceed_to_phase7cr3_confirmation_design": all_success,
            "run_ann": False,
            "r3_confirmation_generated": False,
            "conclusion": (
                "R2开发、内部验证和旧证据回归严格通过；可以冻结"
                "R1+R2并单独设计R3确认集，仍禁止运行ANN。"
                if all_success
                else "R2触发停止条件；不得生成R3确认集或运行ANN。"
            ),
        },
    }
    (result_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(result_dir / "PHASE7C-R2_中文实验报告.md", payload)
    _plot(result_dir, final, guard_payload)
    return payload


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    guards = payload["derived_voltage_guards"]["guards_v"]
    global_summary = payload["global_summary"]
    rows = []
    for role, temperatures in payload["role_results"].items():
        for token, result in temperatures.items():
            summary = result["summary"]
            rows.append(
                f"| {role} | {token} ℃ | "
                f"{summary['trajectory_count']} | "
                f"{100 * summary['target_reach_fraction']:.1f}% | "
                f"{summary['maximum_voltage_v']:.6f} | "
                f"{summary['maximum_temperature_c']:.6f} | "
                f"{summary['guard_exceedance_count']} | "
                f"{summary['empty_voltage_slew_count']}/"
                f"{summary['empty_thermal_slew_count']} | "
                f"{summary['solver_failure_count']} | "
                f"{summary['sustained_oscillation_count']} | "
                f"{'通过' if result['success'] else '失败'} |"
            )
    report = f"""# Phase 7C-R2：分温度电压裕量实验报告

## 边界

本阶段冻结R1热监督层、教师修复、MPC、2RC模型、采样周期和全部物理门槛。
只开发15 ℃和30 ℃的电压残差增长裕量；25 ℃裕量保持原值。未训练或运行ANN，
未生成R3独立确认集。Phase 7C旧48条和R1旧8条仅作失败回归。

## 冻结电压裕量

- 15 ℃：{1000 * guards['15']:.6f} mV
- 25 ℃：{1000 * guards['25']:.6f} mV
- 30 ℃：{1000 * guards['30']:.6f} mV

15 ℃和30 ℃按“历史下限、新开发集最大正向增长、25 ℃原值三者最大值，
再加0.5 mV预注册余量”确定；本阶段不允许降低裕量。

## 统一可行区间结果

| 数据角色 | 温度 | 轨迹 | 到达率 | 最高电压/V | 最高温度/℃ | 裕量超越 | 电压/热空区间 | 求解失败 | 持续振荡 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## 判定

{payload['decision']['conclusion']}

## 全局统计

- 最终验证/回归轨迹：{global_summary['final_trajectory_count']}条；
- 目标到达率：{100 * global_summary['target_reach_fraction']:.1f}%；
- 最高DFN电压：{global_summary['maximum_voltage_v']:.6f} V；
- 最高平均温度：{global_summary['maximum_temperature_c']:.6f} ℃；
- 最大单步电流变化：{global_summary['maximum_current_step_a']:.6f} A；
- 统一电压+热监督总计算时间：
  {global_summary['unified_voltage_thermal_supervisor_total_time_s']:.6f} s；
- 热监督层总计算时间保守上界：
  {global_summary['thermal_supervisor_total_time_upper_bound_s']:.6f} s。

计时说明：{global_summary['timing_note']}

## 停止原因

- 冻结裕量超越事件：
  {len(payload['failure_diagnosis']['guard_exceedance_events'])} 次；
- 修复后求解失败：
  {len(payload['failure_diagnosis']['solver_failure_events'])} 次；
- 内部验证结果没有用于再次调参；
- 教师分支诊断：
  {payload['failure_diagnosis']['teacher_selection_issue']};
- 后续要求：
  {payload['failure_diagnosis']['required_next_phase']}

该结论仍不是多温度最终通过。只有R1和R2冻结后生成的全新R3确认集才能提供
独立确认；安全MPC在R3严格通过后，才允许运行240条冻结ANN轨迹。
"""
    path.write_text(report, encoding="utf-8")


def _plot(
    result_dir: Path,
    frame: pd.DataFrame,
    guard_payload: dict[str, Any],
) -> None:
    valid = frame[frame.current_a.notna()]
    figure, axes = plt.subplots(
        1, 3, figsize=(15, 4.5), layout="constrained"
    )
    for (role, temperature), group in valid.groupby(
        ["role", "ambient_temperature_c"]
    ):
        if role not in {"development", "internal_validation"}:
            continue
        label = f"{role}, {int(temperature)} ℃"
        axes[0].scatter(
            group.soc,
            1000 * group.positive_residual_growth_v,
            s=4,
            alpha=0.25,
            label=label,
        )
        axes[1].scatter(
            group.soc,
            group.terminal_voltage_v,
            s=4,
            alpha=0.25,
        )
        axes[2].scatter(
            group.soc,
            group.next_temperature_c,
            s=4,
            alpha=0.25,
        )
    for token, guard in guard_payload["guards_v"].items():
        if token != "25":
            axes[0].axhline(
                1000 * guard,
                linestyle="--",
                label=f"g{token}",
            )
    axes[0].set(
        xlabel="SOC",
        ylabel="Positive residual growth [mV]",
        title="Voltage residual-growth coverage",
    )
    axes[1].axhline(4.2, color="red", linestyle="--")
    axes[1].set(
        xlabel="SOC", ylabel="DFN voltage [V]", title="Voltage safety"
    )
    axes[2].axhline(35.0, color="red", linestyle="--")
    axes[2].set(
        xlabel="SOC",
        ylabel="Average temperature [℃]",
        title="Thermal safety",
    )
    axes[0].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(result_dir / "phase7cr2_summary.png", dpi=180)
    plt.close(figure)


def run_phase7cr2(
    config: Phase7CR2Config,
    root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    verify_frozen_artifacts(config, root)
    data_dir = root / config.output["data_directory"]
    freeze_path = data_dir / "initial_state_freeze.json"
    if not freeze_path.exists():
        freeze_new_states(config, root)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for name, record in freeze["files"].items():
        if _sha256(data_dir / name) != record["sha256"]:
            raise RuntimeError(f"R2初态冻结哈希不匹配：{name}")

    original = float(config.voltage["original_25c_guard_v"])
    audit_guards = {"15": original, "25": original, "30": original}
    audit = _run_rows(
        config,
        root,
        _new_state_rows(config, root, "development"),
        "development_guard_audit",
        audit_guards,
        resume,
    )
    guard_payload = _derive_guards(config, audit, root)
    guards = guard_payload["guards_v"]

    frames = []
    for role, rows in (
        (
            "development",
            _new_state_rows(config, root, "development"),
        ),
        (
            "internal_validation",
            _new_state_rows(config, root, "internal_validation"),
        ),
        (
            "legacy_phase7c",
            _legacy_rows(config, root, "legacy_phase7c"),
        ),
        (
            "legacy_phase7cr1",
            _legacy_rows(config, root, "legacy_phase7cr1"),
        ),
    ):
        frames.append(
            _run_rows(config, root, rows, role, guards, resume)
        )
    final = pd.concat(frames, ignore_index=True)
    return _analyze(config, root, audit, final, guard_payload)
