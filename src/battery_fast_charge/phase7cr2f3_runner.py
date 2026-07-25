"""Phase 7C-R2F3 per-temperature two-stage voltage-guard repair."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .phase7a_level3_model import Level3State
from .phase7b1b_runner import _maximum_safe_current
from .phase7c_runner import Chen2020ThermalDFN
from .phase7cr2_runner import (
    _context,
    _summarize_group,
    _thermal_current_limit,
    _trajectory_metrics,
)
from .phase7cr2f2_config import load_phase7cr2f2_config
from .phase7cr2f2_residual_audit import _artifact_hash
from .phase7cr2f2_runner import (
    _bounded_candidate,
    _historical_roles as _r2f2_historical_roles,
    _initial_measurement,
    _prefix,
    _scrambled_unit,
    _sha256,
    _state_columns,
)
from .phase7cr2f3_config import Phase7CR2F3Config
from .phase7cr2f_config import load_phase7cr2f_config
from .phase7cr2f_runner import (
    _strict_failure_row,
    verify_known_teacher_regressions,
)
from .phase7cr2f_teacher import StrictTeacherSelectionError, solve_teacher_r2f


def verify_frozen_r2f2(
    config: Phase7CR2F3Config, root: Path
) -> dict[str, Any]:
    sources = config.section("sources")
    manifest_path = root / sources["phase7cr2f2_freeze_manifest"]
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_hash != sources["phase7cr2f2_freeze_manifest_sha256"]:
        raise RuntimeError("R2F2 freeze-manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "strict_stop_failed":
        raise RuntimeError("R2F2 strict-stop evidence status changed")
    if manifest["r3_initial_states_generated"]:
        raise RuntimeError("R2F2 evidence unexpectedly contains R3 states")
    if manifest["ann_execution_authorized"]:
        raise RuntimeError("R2F2 evidence unexpectedly authorizes ANN")
    records: dict[str, Any] = {}
    mismatches: list[str] = []
    for relative, expected in manifest["artifacts"].items():
        actual, matched = _artifact_hash(root / relative, expected)
        records[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": matched,
        }
        if not matched:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"R2F2 frozen artifacts changed: {mismatches}")
    guard_path = root / sources["phase7cr2f2_frozen_guards"]
    guard_hash = hashlib.sha256(guard_path.read_bytes()).hexdigest()
    if guard_hash != sources["phase7cr2f2_frozen_guards_sha256"]:
        raise RuntimeError("R2F2 frozen-guard hash mismatch")
    frozen = json.loads(guard_path.read_text(encoding="utf-8"))["guards"]["30"]
    expected_30 = config.section("voltage_guard")["historical_limits"][30]
    if not np.isclose(frozen["boot_v"], expected_30["boot_v"], atol=1.0e-15):
        raise RuntimeError("Frozen 30 C boot guard changed")
    if not np.isclose(
        frozen["running_v"], expected_30["running_v"], atol=1.0e-15
    ):
        raise RuntimeError("Frozen 30 C running guard changed")
    contract = config.section("control_contract")
    if contract["residual_initialization"] != "measured":
        raise RuntimeError("R2F3 requires measured residual initialization")
    if contract["allow_zero_residual_fallback"]:
        raise RuntimeError("Zero residual fallback is forbidden")
    if contract["r3_generation_authorized"] or contract["ann_execution_authorized"]:
        raise RuntimeError("R2F3 cannot authorize R3 or ANN")
    return {
        "manifest_sha256": manifest_hash,
        "frozen_guard_sha256": guard_hash,
        "r2f2_failure_preserved": True,
        "records": records,
    }


def guard_for_step(
    temperature_c: float,
    step_index: int,
    guards: dict[str, Any],
    boot_steps: tuple[int, ...] = (0, 1),
) -> tuple[float, str]:
    token = str(int(round(temperature_c)))
    if step_index in boot_steps:
        return float(guards[token]["boot_v"]), "boot"
    return float(guards[token]["running_v"]), "running"


def derive_temperature_guards(
    voltage: dict[str, Any], development: pd.DataFrame
) -> dict[str, Any]:
    observed: dict[str, dict[str, float]] = {}
    for temperature in (15, 25):
        group = development[
            np.isclose(development.ambient_temperature_c, temperature)
        ]
        boot = group[group.step_index.isin([0, 1])]
        running = group[group.step_index >= 2]
        if boot.empty or running.empty:
            raise RuntimeError(f"Development coverage incomplete at {temperature} C")
        observed[str(temperature)] = {
            "boot_v": float(boot.positive_residual_growth_v.max()),
            "running_v": float(running.positive_residual_growth_v.max()),
        }
    margin = float(voltage["engineering_margin_v"])
    history = voltage["historical_limits"]
    guards = {
        "15": {
            "boot_v": max(
                float(history[15]["boot_v"]), observed["15"]["boot_v"]
            )
            + margin,
            "running_v": max(
                float(history[15]["running_floor_v"]),
                observed["15"]["running_v"] + margin,
            ),
        },
        "25": {
            "boot_v": max(
                float(history[25]["boot_v"]), observed["25"]["boot_v"]
            )
            + margin,
            "running_v": max(
                float(history[25]["previous_guard_v"]),
                float(history[25]["known_running_growth_v"]) + margin,
                observed["25"]["running_v"] + margin,
            ),
        },
        "30": {
            "boot_v": float(history[30]["boot_v"]),
            "running_v": float(history[30]["running_v"]),
        },
    }
    return {
        "guards": guards,
        "development_maxima": observed,
        "engineering_margin_v": margin,
    }


def _source_state_frames(
    config: Phase7CR2F3Config, root: Path
) -> list[pd.DataFrame]:
    sources = config.section("sources")
    paths = list(sources["legacy_phase7c_states"].values()) + [
        sources["legacy_phase7cr1_states"],
        sources["legacy_phase7cr2_development_states"],
        sources["legacy_phase7cr2_internal_states"],
        sources["legacy_phase7cr2f_development_states"],
        sources["legacy_phase7cr2f_internal_states"],
        sources["legacy_phase7b1_25c_states"],
        sources["phase7cr2f2_development_states"],
        sources["phase7cr2f2_internal_states"],
    ]
    return [pd.read_csv(root / relative) for relative in paths]


def _all_historical_values(
    config: Phase7CR2F3Config, root: Path
) -> np.ndarray:
    return pd.concat(_source_state_frames(config, root), ignore_index=True)[
        _state_columns()
    ].to_numpy(float)


def _temperature_centers(
    config: Phase7CR2F3Config, root: Path, temperature_c: int
) -> list[tuple[np.ndarray, str]]:
    records: list[tuple[np.ndarray, str]] = []
    for item in config.section("datasets")["historical_centers"][temperature_c]:
        frame = pd.read_csv(root / item["source"])
        match = frame[frame.trajectory_id == item["trajectory_id"]]
        if len(match) != 1:
            raise RuntimeError(f"Historical center not unique: {item}")
        records.append(
            (match.iloc[0][_state_columns()].to_numpy(float), item["evidence"])
        )
    return records


def _neighborhood_candidate(
    index: int,
    seed: int,
    center: np.ndarray,
    config: Phase7CR2F3Config,
) -> np.ndarray:
    datasets = config.section("datasets")
    widths = datasets["neighborhood_half_width"]
    half = np.asarray(
        [
            widths["soc"],
            widths["v1_v"],
            widths["v2_v"],
            widths["previous_current_a"],
        ],
        dtype=float,
    )
    candidate = center + (2.0 * _scrambled_unit(index, seed) - 1.0) * half
    bounds = datasets["bounds"]["global_coverage"]
    keys = ("soc", "v1_v", "v2_v", "previous_current_a")
    lower = np.asarray([bounds[key][0] for key in keys], dtype=float)
    upper = np.asarray([bounds[key][1] for key in keys], dtype=float)
    return np.clip(candidate, lower, upper)


def _build_state_set(
    config: Phase7CR2F3Config,
    root: Path,
    role: str,
    temperature_c: int,
    existing: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    datasets = config.section("datasets")
    design = datasets["designs"][role][temperature_c]
    counts = datasets["strata"][role]
    cursor = int(design["start_index"])
    seed = int(design["seed"])
    centers = _temperature_centers(config, root, temperature_c)
    r2f = load_phase7cr2f_config(
        root / config.section("sources")["phase7cr2f_config"]
    )
    r1, _, level3, _, model, _ = _context(r2f, root)
    voltage_limit = (
        model.inherited.mpc.terminal_voltage_max_v
        - float(datasets["initial_voltage_margin_v"])
    )
    records: list[dict[str, Any]] = []
    for stratum in (
        "startup_transient_stress",
        "historical_exceedance_neighborhood",
        "general_high_risk",
        "global_coverage",
    ):
        accepted = 0
        attempts = 0
        while accepted < int(counts[stratum]):
            attempts += 1
            if attempts > 10000:
                raise RuntimeError(
                    f"Unable to fill {role}/{temperature_c}C/{stratum}"
                )
            used = cursor
            evidence = "bounded_design"
            center_index = -1
            if stratum == "historical_exceedance_neighborhood":
                center_index = accepted % len(centers)
                center, evidence = centers[center_index]
                values = _neighborhood_candidate(
                    cursor, seed, center, config
                )
            else:
                values = _bounded_candidate(
                    cursor, seed, datasets["bounds"][stratum]
                )
            cursor += 1
            if np.any(
                np.all(np.isclose(values, existing, atol=1.0e-14), axis=1)
            ):
                continue
            state = Level3State(*values)
            minimum_current = max(
                0.0,
                state.previous_current_a
                - level3.constraint.maximum_current_step_a,
            )
            if model.terminal_voltage(state, minimum_current) > voltage_limit:
                continue
            try:
                solve_teacher_r2f(state, model, r1)
            except StrictTeacherSelectionError:
                continue
            initial = {
                "ambient_temperature_c": float(temperature_c),
                "initial_temperature_c": float(temperature_c),
                "initial_soc": values[0],
                "initial_polarization_1_v": values[1],
                "initial_polarization_2_v": values[2],
                "initial_previous_current_a": values[3],
            }
            measured = _initial_measurement(config, root, initial)
            predicted = model.terminal_voltage(state, state.previous_current_a)
            residual = measured - predicted
            records.append(
                {
                    "trajectory_id": (
                        f"phase7cr2f3_{role}_{temperature_c}c_"
                        f"{len(records):03d}"
                    ),
                    "role": role,
                    "ambient_temperature_c": float(temperature_c),
                    "initial_temperature_c": float(temperature_c),
                    "risk_stratum": stratum,
                    "historical_center_index": center_index,
                    "historical_evidence": evidence,
                    "initial_soc": values[0],
                    "initial_polarization_1_v": values[1],
                    "initial_polarization_2_v": values[2],
                    "initial_previous_current_a": values[3],
                    "initial_dfn_voltage_v": measured,
                    "initial_2rc_voltage_v": predicted,
                    "initial_measured_residual_v": residual,
                    "initial_residual_extreme": bool(
                        residual
                        <= float(datasets["initial_residual_extreme_threshold_v"])
                    ),
                    "design_candidate_index": used,
                    "design_seed": seed,
                    "initial_state_history_contract": (
                        "dfn_and_2rc_do_not_share_current_history"
                    ),
                }
            )
            existing = np.vstack([existing, values])
            accepted += 1
    return pd.DataFrame(records), existing


def prepare_and_freeze_states(
    config: Phase7CR2F3Config, root: Path
) -> dict[str, Any]:
    verification = verify_frozen_r2f2(config, root)
    data_dir = root / config.section("output")["data_directory"]
    data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = data_dir / "initial_state_freeze.json"
    if freeze_path.exists():
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
        for name, record in payload["files"].items():
            if _sha256(data_dir / name) != record["sha256"]:
                raise RuntimeError(f"R2F3 state hash mismatch: {name}")
        return payload
    existing = _all_historical_values(config, root)
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for role in ("development", "internal_validation"):
        for temperature_c in (15, 25):
            frame, existing = _build_state_set(
                config, root, role, temperature_c, existing
            )
            frames[(role, temperature_c)] = frame
    files: dict[str, Any] = {}
    for (role, temperature_c), frame in frames.items():
        name = f"{role}_initial_states_{temperature_c}c.csv"
        path = data_dir / name
        frame.to_csv(path, index=False)
        files[name] = {
            "sha256": _sha256(path),
            "trajectory_count": len(frame),
            "strata_counts": frame.risk_stratum.value_counts().to_dict(),
            "design_seed": int(frame.design_seed.iloc[0]),
            "minimum_design_index": int(frame.design_candidate_index.min()),
            "maximum_design_index": int(frame.design_candidate_index.max()),
            "minimum_initial_residual_v": float(
                frame.initial_measured_residual_v.min()
            ),
            "extreme_initial_residual_count": int(
                frame.initial_residual_extreme.astype(bool).sum()
            ),
        }
    payload = {
        "phase": "Phase 7C-R2F3",
        "status": "initial_states_frozen_before_any_closed_loop_rollout",
        "frozen_before_any_closed_loop_rollout": True,
        "development_internal_isolation": True,
        "not_r3_confirmation_data": True,
        "not_ann_teacher_data": True,
        "initial_state_history_contract": (
            "dfn_and_2rc_do_not_share_current_history"
        ),
        "config_sha256": _sha256(
            root / "configs/phase7cr2f3_temperature_two_stage_guards.yaml",
            True,
        ),
        "files": files,
        "frozen_r2f2_verification": verification,
    }
    freeze_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _rollout(
    config: Phase7CR2F3Config,
    root: Path,
    initial: dict[str, Any],
    role: str,
    guards: dict[str, Any],
) -> pd.DataFrame:
    r2f = load_phase7cr2f_config(
        root / config.section("sources")["phase7cr2f_config"]
    )
    r1, b1, level3, inherited, model, phase7b0 = _context(r2f, root)
    ambient = float(initial["ambient_temperature_c"])
    state = Level3State(
        float(initial["initial_soc"]),
        float(initial["initial_polarization_1_v"]),
        float(initial["initial_polarization_2_v"]),
        float(initial["initial_previous_current_a"]),
    )
    measured_initial_voltage = _initial_measurement(config, root, initial)
    predicted_initial_voltage = model.terminal_voltage(
        state, state.previous_current_a
    )
    residual_v = measured_initial_voltage - predicted_initial_voltage
    if not np.isfinite(residual_v):
        raise RuntimeError("Measured residual initialization failed")
    plant = Chen2020ThermalDFN(
        phase7b0.dfn.parameter_set,
        ambient,
        phase7b0.dfn.upper_voltage_cutoff_v,
        state.soc,
        level3.model.sample_period_s,
        "lumped",
    )
    temperature_c = float(initial["initial_temperature_c"])
    lower_bound, upper_bound = inherited.mpc.current_bounds_a
    maximum_step = level3.constraint.maximum_current_step_a
    boot_steps = tuple(config.section("control_contract")["boot_steps"])
    rows: list[dict[str, Any]] = []
    for step in range(int(config.section("datasets")["maximum_steps"])):
        guard_v, guard_stage = guard_for_step(
            ambient, step, guards, boot_steps
        )
        teacher_started = perf_counter()
        try:
            result, teacher = solve_teacher_r2f(state, model, r1)
        except StrictTeacherSelectionError as error:
            failure = _strict_failure_row(
                role, initial, step, state, temperature_c, guard_v, error
            )
            failure["guard_stage"] = guard_stage
            failure["residual_initialization_mode"] = "measured"
            failure["initial_residual_v"] = residual_v
            return pd.concat([pd.DataFrame(rows), failure], ignore_index=True)
        teacher_time_s = perf_counter() - teacher_started
        candidate = float(result.current_a)
        slew_lower = max(lower_bound, state.previous_current_a - maximum_step)
        slew_upper = min(upper_bound, state.previous_current_a + maximum_step)
        supervisor_started = perf_counter()
        voltage_max = _maximum_safe_current(
            state, residual_v + guard_v, b1, model
        )
        search_upper = min(slew_upper, voltage_max)
        constant_max, constant_peak = _thermal_current_limit(
            temperature_c, ambient, search_upper, r1, braking=False
        )
        braking_max, braking_peak = _thermal_current_limit(
            temperature_c, ambient, search_upper, r1, braking=True
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
            "source_trajectory_id": initial.get(
                "source_trajectory_id", initial["trajectory_id"]
            ),
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
            "guard_stage": guard_stage,
            "residual_initialization_mode": "measured",
            "initial_residual_v": measured_initial_voltage
            - predicted_initial_voltage,
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
            "strict_teacher_failure": False,
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
        new_residual = float(measurement["terminal_voltage_v"]) - predicted_voltage
        positive_growth = max(0.0, new_residual - residual_v)
        rows.append(
            {
                **base,
                "current_a": current,
                "current_step_a": abs(current - state.previous_current_a),
                "next_soc": measurement["soc"],
                "next_temperature_c": measurement["average_temperature_c"],
                "terminal_voltage_v": measurement["terminal_voltage_v"],
                "predicted_voltage_v": predicted_voltage,
                "voltage_residual_before_v": residual_v,
                "voltage_residual_after_v": new_residual,
                "positive_residual_growth_v": positive_growth,
                "guard_exceeded": bool(
                    positive_growth
                    > guard_v
                    + float(
                        config.section("voltage_guard")[
                            "residual_growth_tolerance_v"
                        ]
                    )
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
        if state.soc >= float(config.section("datasets")["target_soc"]):
            break
    return pd.DataFrame(rows)


def _worker(
    config: Phase7CR2F3Config,
    root_text: str,
    initial: dict[str, Any],
    role: str,
    guards: dict[str, Any],
    path_text: str,
) -> str:
    frame = _rollout(config, Path(root_text), initial, role, guards)
    frame.to_csv(path_text, index=False)
    return path_text


def _run_rows(
    config: Phase7CR2F3Config,
    root: Path,
    rows: list[dict[str, Any]],
    role: str,
    guards: dict[str, Any],
    run_group: str,
    resume: bool,
) -> pd.DataFrame:
    data_dir = root / config.section("output")["data_directory"]
    run_dir = data_dir / "runs" / run_group / role
    run_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    pending: list[tuple[dict[str, Any], Path]] = []
    for row in rows:
        path = run_dir / f"{row['trajectory_id']}.csv"
        paths.append(path)
        if not (resume and path.exists()):
            pending.append((row, path))
    if pending:
        with ProcessPoolExecutor(
            max_workers=int(config.section("datasets")["maximum_workers"])
        ) as executor:
            futures = {
                executor.submit(
                    _worker,
                    config,
                    str(root),
                    row,
                    role,
                    guards,
                    str(path),
                ): path
                for row, path in pending
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                future.result()
                print(
                    f"[Phase 7C-R2F3:{run_group}:{role}] "
                    f"{completed}/{len(pending)}",
                    flush=True,
                )
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _new_rows(
    config: Phase7CR2F3Config, root: Path, role: str
) -> list[dict[str, Any]]:
    data_dir = root / config.section("output")["data_directory"]
    frames = [
        pd.read_csv(data_dir / f"{role}_initial_states_{temperature}c.csv")
        for temperature in (15, 25)
    ]
    return pd.concat(frames, ignore_index=True).to_dict(orient="records")


def _historical_roles(
    config: Phase7CR2F3Config, root: Path
) -> list[tuple[str, list[dict[str, Any]]]]:
    sources = config.section("sources")
    f2 = load_phase7cr2f2_config(root / sources["phase7cr2f2_config"])
    roles = _r2f2_historical_roles(f2, root)
    for role, key in (
        ("legacy_phase7cr2f2_development", "phase7cr2f2_development_states"),
        ("legacy_phase7cr2f2_internal", "phase7cr2f2_internal_states"),
    ):
        roles.append(
            (
                role,
                _prefix(pd.read_csv(root / sources[key]), role, 30.0),
            )
        )
    count = sum(len(rows) for _, rows in roles)
    expected = int(
        config.section("validation_contract")[
            "expected_historical_regression_trajectory_count"
        ]
    )
    if count != expected:
        raise RuntimeError(
            f"Historical regression count changed: {count} != {expected}"
        )
    return roles


def _verify_state_freeze(
    config: Phase7CR2F3Config, root: Path
) -> dict[str, Any]:
    data_dir = root / config.section("output")["data_directory"]
    path = data_dir / "initial_state_freeze.json"
    if not path.exists():
        raise RuntimeError("R2F3 states must be prepared and frozen first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, record in payload["files"].items():
        if _sha256(data_dir / name) != record["sha256"]:
            raise RuntimeError(f"R2F3 state hash mismatch: {name}")
    return payload


def run_development(
    config: Phase7CR2F3Config, root: Path, resume: bool
) -> dict[str, Any]:
    verify_frozen_r2f2(config, root)
    state_freeze = _verify_state_freeze(config, root)
    data_dir = root / config.section("output")["data_directory"]
    guard_path = data_dir / "frozen_temperature_two_stage_guards.json"
    if guard_path.exists():
        return _verify_guard_freeze(config, root)
    audit_guards = {
        str(temperature): {
            "boot_v": float(values["boot_v"]),
            "running_v": float(values["running_v"]),
        }
        for temperature, values in config.section("voltage_guard")[
            "development_audit_guards"
        ].items()
    }
    audit = _run_rows(
        config,
        root,
        _new_rows(config, root, "development"),
        "development_guard_audit",
        audit_guards,
        "development_audit",
        resume,
    )
    audit_path = data_dir / "development_guard_audit.csv"
    audit.to_csv(audit_path, index=False)
    derived = derive_temperature_guards(
        config.section("voltage_guard"), audit
    )
    payload = {
        "phase": "Phase 7C-R2F3 development",
        "status": "temperature_two_stage_guards_frozen_before_internal_validation",
        "design": "per_temperature_boot_steps_0_1_running_steps_2_plus",
        "residual_initialization": "measured",
        "internal_validation_used_for_tuning": False,
        "guards": derived["guards"],
        "development_maxima": derived["development_maxima"],
        "engineering_margin_v": derived["engineering_margin_v"],
        "state_hashes": {
            name: record["sha256"]
            for name, record in state_freeze["files"].items()
        },
        "development_audit_sha256": _sha256(audit_path),
        "config_sha256": _sha256(
            root / "configs/phase7cr2f3_temperature_two_stage_guards.yaml",
            True,
        ),
        "runner_sha256": _sha256(Path(__file__), True),
    }
    guard_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _verify_guard_freeze(
    config: Phase7CR2F3Config, root: Path
) -> dict[str, Any]:
    data_dir = root / config.section("output")["data_directory"]
    path = data_dir / "frozen_temperature_two_stage_guards.json"
    if not path.exists():
        raise RuntimeError("R2F3 guards must be developed and frozen first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["internal_validation_used_for_tuning"]:
        raise RuntimeError("Internal validation cannot tune R2F3 guards")
    if payload["config_sha256"] != _sha256(
        root / "configs/phase7cr2f3_temperature_two_stage_guards.yaml", True
    ):
        raise RuntimeError("R2F3 config changed after guard freeze")
    if payload["runner_sha256"] != _sha256(Path(__file__), True):
        raise RuntimeError("R2F3 runner changed after guard freeze")
    if payload["development_audit_sha256"] != _sha256(
        data_dir / "development_guard_audit.csv"
    ):
        raise RuntimeError("R2F3 development audit changed after guard freeze")
    expected_30 = config.section("voltage_guard")["historical_limits"][30]
    actual_30 = payload["guards"]["30"]
    if actual_30 != {
        "boot_v": float(expected_30["boot_v"]),
        "running_v": float(expected_30["running_v"]),
    }:
        raise RuntimeError("30 C frozen guard changed during R2F3")
    return payload


def _temperature_stage_counts(rows: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    boot_total = 0
    running_total = 0
    for temperature in (15, 25, 30):
        group = rows[np.isclose(rows.ambient_temperature_c, temperature)]
        boot = int(
            ((group.guard_stage == "boot") & group.guard_exceeded.astype(bool)).sum()
        )
        running = int(
            (
                (group.guard_stage == "running")
                & group.guard_exceeded.astype(bool)
            ).sum()
        )
        boot_total += boot
        running_total += running
        result[str(temperature)] = {
            "boot_exceedance_count": boot,
            "running_exceedance_count": running,
            "total_exceedance_count": boot + running,
        }
    result["all_temperatures"] = {
        "boot_exceedance_count": boot_total,
        "running_exceedance_count": running_total,
        "total_exceedance_count": boot_total + running_total,
    }
    return result


def _checks(
    config: Phase7CR2F3Config,
    trajectory_metrics: pd.DataFrame,
    rows: pd.DataFrame,
) -> dict[str, bool]:
    gates = config.section("gates")
    tolerance = float(gates["numerical_tolerance"])
    counts = _temperature_stage_counts(rows)
    return {
        "maximum_voltage": bool(
            trajectory_metrics.maximum_voltage_v.max()
            <= float(gates["maximum_voltage_v"])
        ),
        "maximum_temperature": bool(
            trajectory_metrics.maximum_temperature_c.max()
            <= float(gates["maximum_average_temperature_c"])
        ),
        "current_bounds": bool(
            trajectory_metrics.minimum_current_a.min()
            >= float(gates["minimum_current_a"]) - tolerance
            and trajectory_metrics.maximum_current_a.max()
            <= float(gates["maximum_current_a"]) + tolerance
        ),
        "slew_bound": bool(
            trajectory_metrics.maximum_current_step_a.max()
            <= float(gates["maximum_current_step_a"]) + tolerance
        ),
        "all_temperature_boot_guard_exceedance_zero": bool(
            counts["all_temperatures"]["boot_exceedance_count"] == 0
        ),
        "all_temperature_running_guard_exceedance_zero": bool(
            counts["all_temperatures"]["running_exceedance_count"] == 0
        ),
        "all_temperature_total_guard_exceedance_zero": bool(
            counts["all_temperatures"]["total_exceedance_count"] == 0
        ),
        "zero_voltage_slew_empty": bool(
            trajectory_metrics.empty_voltage_slew_count.sum() == 0
        ),
        "zero_thermal_slew_empty": bool(
            trajectory_metrics.empty_thermal_slew_count.sum() == 0
        ),
        "zero_solver_failure": bool(
            trajectory_metrics.solver_failure_count.sum() == 0
        ),
        "zero_prediction_infeasible": bool(
            trajectory_metrics.prediction_infeasible_count.sum() == 0
        ),
        "zero_strict_teacher_failure": bool(
            not rows.strict_teacher_failure.astype(bool).any()
        ),
        "zero_sustained_oscillation": bool(
            trajectory_metrics.sustained_oscillation_count.sum() == 0
        ),
        "target_reach_100_percent": bool(
            trajectory_metrics.target_reached.astype(bool).all()
        ),
        "measured_initialization_100_percent": bool(
            (rows.residual_initialization_mode == "measured").all()
            and rows.initial_residual_v.notna().all()
        ),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    guards = payload["frozen_guard_contract"]["guards"]
    counts = payload["guard_exceedance_counts"]
    summary = payload["global_summary"]
    guard_rows = []
    for temperature in (15, 25, 30):
        token = str(temperature)
        guard_rows.append(
            f"| {temperature} | {1000 * guards[token]['boot_v']:.6f} | "
            f"{1000 * guards[token]['running_v']:.6f} | "
            f"{counts[token]['boot_exceedance_count']} | "
            f"{counts[token]['running_exceedance_count']} | "
            f"{counts[token]['total_exceedance_count']} |"
        )
    conclusion = (
        "R2F3严格通过，可以冻结完整多温度安全MPC架构并另行设计R3；当前仍未生成R3初态或运行ANN。"
        if payload["success"]
        else "R2F3触发严格停止；新内部验证永久降级为回归证据，不得原地调参、生成R3或运行ANN。"
    )
    report = f"""# Phase 7C-R2F3 分温度两段裕量实验报告

## 冻结合同

所有温度均在控制步0和1使用启动裕量，从控制步2开始使用运行裕量。残差由时刻0端电压测量初始化，不允许回退到零。30 ℃架构与裕量继承R2F2并保持冻结；R2F3只开发15 ℃和25 ℃。

## 分温度裕量与超越计数

| 温度/℃ | 启动裕量/mV | 运行裕量/mV | 启动超越 | 运行超越 | 总超越 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(guard_rows)}
| 全温度 | — | — | {counts['all_temperatures']['boot_exceedance_count']} | {counts['all_temperatures']['running_exceedance_count']} | {counts['all_temperatures']['total_exceedance_count']} |

## 一次性验证

- 新内部验证：{summary['new_internal_trajectory_count']}条；
- 历史回归：{summary['historical_regression_trajectory_count']}条；
- 总轨迹：{summary['trajectory_count']}条；
- 目标到达率：{100 * summary['target_reach_fraction']:.1f}%；
- 最高DFN电压：{summary['maximum_voltage_v']:.6f} V；
- 最高平均温度：{summary['maximum_temperature_c']:.6f} ℃；
- 最大单步电流变化：{summary['maximum_current_step_a']:.6f} A；
- 严格教师失败/求解失败/预测不可行：{summary['strict_teacher_failure_count']}/{summary['solver_failure_count']}/{summary['prediction_infeasible_count']}；
- 电压—斜率/热—斜率空区间：{summary['empty_voltage_slew_count']}/{summary['empty_thermal_slew_count']}；
- 持续振荡：{summary['sustained_oscillation_count']}；
- 零残差初始化轨迹：{summary['zero_residual_initialization_count']}。

## 判定

{conclusion}
"""
    path.write_text(report, encoding="utf-8")


def run_validation(
    config: Phase7CR2F3Config, root: Path, resume: bool
) -> dict[str, Any]:
    source_verification = verify_frozen_r2f2(config, root)
    state_freeze = _verify_state_freeze(config, root)
    guard_freeze = _verify_guard_freeze(config, root)
    data_dir = root / config.section("output")["data_directory"]
    result_dir = root / config.section("output")["result_directory"]
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    started_path = data_dir / "validation_started.json"
    if started_path.exists() and not resume:
        raise RuntimeError(
            "One-shot R2F3 validation already started; use --resume only"
        )
    if not started_path.exists():
        internal_hashes = {
            name: record["sha256"]
            for name, record in state_freeze["files"].items()
            if name.startswith("internal_validation")
        }
        started_path.write_text(
            json.dumps(
                {
                    "status": "one_shot_validation_started",
                    "guards_sha256": _sha256(
                        data_dir / "frozen_temperature_two_stage_guards.json"
                    ),
                    "internal_state_sha256": internal_hashes,
                    "internal_validation_may_not_modify_guards": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    guards = guard_freeze["guards"]
    frames: list[pd.DataFrame] = []
    for role, rows in _historical_roles(config, root):
        frames.append(
            _run_rows(
                config, root, rows, role, guards, "validation", resume
            )
        )
    frames.append(
        _run_rows(
            config,
            root,
            _new_rows(config, root, "internal_validation"),
            "internal_validation",
            guards,
            "validation",
            resume,
        )
    )
    final = pd.concat(frames, ignore_index=True)
    combined_path = data_dir / "combined_validation_trajectories.csv"
    final.to_csv(combined_path, index=False)
    r2f = load_phase7cr2f_config(
        root / config.section("sources")["phase7cr2f_config"]
    )
    metrics = _trajectory_metrics(r2f, final)
    metrics_path_csv = data_dir / "trajectory_metrics.csv"
    metrics.to_csv(metrics_path_csv, index=False)
    expected = config.section("validation_contract")
    historical_count = int(
        metrics[metrics.role != "internal_validation"].trajectory_id.nunique()
    )
    internal_count = int(
        metrics[metrics.role == "internal_validation"].trajectory_id.nunique()
    )
    total_count = int(metrics.trajectory_id.nunique())
    if historical_count != int(
        expected["expected_historical_regression_trajectory_count"]
    ):
        raise RuntimeError("Historical regression trajectory count mismatch")
    if internal_count != int(expected["expected_new_internal_trajectory_count"]):
        raise RuntimeError("New internal-validation trajectory count mismatch")
    if total_count != int(expected["expected_total_validation_trajectory_count"]):
        raise RuntimeError("Total validation trajectory count mismatch")
    counts = _temperature_stage_counts(final)
    checks = _checks(config, metrics, final)
    teacher_regression = verify_known_teacher_regressions(r2f, root)
    success = bool(all(checks.values()) and teacher_regression["all_passed"])
    zero_initialization_count = int(
        final.groupby("trajectory_id").residual_initialization_mode.first()
        .ne("measured")
        .sum()
    )
    payload = {
        "study_name": config.study_name,
        "configuration": config.payload,
        "frozen_r2f2_verification": source_verification,
        "initial_state_freeze": state_freeze,
        "frozen_guard_contract": guard_freeze,
        "teacher_selection_regression": teacher_regression,
        "guard_exceedance_counts": counts,
        "checks": checks,
        "global_summary": {
            "trajectory_count": total_count,
            "new_internal_trajectory_count": internal_count,
            "historical_regression_trajectory_count": historical_count,
            "target_reach_fraction": float(metrics.target_reached.mean()),
            "maximum_voltage_v": float(metrics.maximum_voltage_v.max()),
            "maximum_temperature_c": float(metrics.maximum_temperature_c.max()),
            "maximum_current_step_a": float(
                metrics.maximum_current_step_a.max()
            ),
            "strict_teacher_failure_count": int(
                final.strict_teacher_failure.astype(bool).sum()
            ),
            "solver_failure_count": int(metrics.solver_failure_count.sum()),
            "prediction_infeasible_count": int(
                metrics.prediction_infeasible_count.sum()
            ),
            "empty_voltage_slew_count": int(
                metrics.empty_voltage_slew_count.sum()
            ),
            "empty_thermal_slew_count": int(
                metrics.empty_thermal_slew_count.sum()
            ),
            "sustained_oscillation_count": int(
                metrics.sustained_oscillation_count.sum()
            ),
            "zero_residual_initialization_count": zero_initialization_count,
        },
        "success": success,
        "decision": {
            "freeze_full_multitemperature_architecture": success,
            "eligible_to_design_r3_separately": success,
            "r3_initial_states_generated": False,
            "ann_run_or_training_performed": False,
            "internal_validation_used_for_retuning": False,
            "internal_validation_becomes_permanent_regression": bool(
                not success
            ),
        },
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = result_dir / "PHASE7C-R2F3_中文实验报告.md"
    _write_report(report_path, payload)
    artifacts = [
        "configs/phase7cr2f3_temperature_two_stage_guards.yaml",
        "src/battery_fast_charge/phase7cr2f3_config.py",
        "src/battery_fast_charge/phase7cr2f3_runner.py",
        "src/battery_fast_charge/phase7cr2f3_cli.py",
        "data/phase7cr2f3_temperature_two_stage_guards/initial_state_freeze.json",
        "data/phase7cr2f3_temperature_two_stage_guards/development_initial_states_15c.csv",
        "data/phase7cr2f3_temperature_two_stage_guards/development_initial_states_25c.csv",
        "data/phase7cr2f3_temperature_two_stage_guards/internal_validation_initial_states_15c.csv",
        "data/phase7cr2f3_temperature_two_stage_guards/internal_validation_initial_states_25c.csv",
        "data/phase7cr2f3_temperature_two_stage_guards/frozen_temperature_two_stage_guards.json",
        "data/phase7cr2f3_temperature_two_stage_guards/development_guard_audit.csv",
        "data/phase7cr2f3_temperature_two_stage_guards/combined_validation_trajectories.csv",
        "data/phase7cr2f3_temperature_two_stage_guards/trajectory_metrics.csv",
        "outputs/phase7cr2f3_temperature_two_stage_guards/metrics.json",
        "outputs/phase7cr2f3_temperature_two_stage_guards/PHASE7C-R2F3_中文实验报告.md",
    ]
    manifest = {
        "phase": "Phase 7C-R2F3",
        "status": "strict_passed" if success else "strict_stop_failed",
        "internal_validation_used_for_retuning": False,
        "r3_initial_states_generated": False,
        "ann_execution_authorized": False,
        "artifacts": {
            relative: _sha256(
                root / relative,
                Path(relative).suffix in {".py", ".yaml", ".md"},
            )
            for relative in artifacts
        },
    }
    (result_dir / "freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def run_phase7cr2f3(
    config: Phase7CR2F3Config,
    root: Path,
    stage: str,
    resume: bool = False,
) -> dict[str, Any]:
    if stage == "prepare":
        return prepare_and_freeze_states(config, root)
    if stage == "develop":
        return run_development(config, root, resume)
    if stage == "validate":
        return run_validation(config, root, resume)
    prepare_and_freeze_states(config, root)
    run_development(config, root, resume)
    return run_validation(config, root, resume)
