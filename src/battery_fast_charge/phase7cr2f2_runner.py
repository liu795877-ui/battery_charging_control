"""Phase 7C-R2F2 two-stage voltage-guard development and validation."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .phase7a_level3_model import Level3State
from .phase7b1b_runner import _maximum_safe_current
from .phase7c_runner import Chen2020ThermalDFN, _van_der_corput
from .phase7cr2_runner import (
    _checks,
    _context,
    _summarize_group,
    _thermal_current_limit,
    _trajectory_metrics,
)
from .phase7cr2f2_config import Phase7CR2F2Config
from .phase7cr2f2_residual_audit import (
    _artifact_hash,
    verify_frozen_r2f,
)
from .phase7cr2f_config import load_phase7cr2f_config
from .phase7cr2f_runner import (
    _strict_failure_row,
    verify_known_teacher_regressions,
)
from .phase7cr2f_teacher import StrictTeacherSelectionError, solve_teacher_r2f


def _sha256(path: Path, normalize_text: bool = False) -> str:
    payload = path.read_bytes()
    if normalize_text:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _verify_manifest(
    root: Path, relative: str, expected_hash: str, expected_status: str
) -> dict[str, Any]:
    path = root / relative
    actual = _sha256(path)
    if actual != expected_hash:
        raise RuntimeError(f"Frozen manifest hash mismatch: {relative}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["status"] != expected_status:
        raise RuntimeError(f"Frozen status changed: {relative}")
    records: dict[str, Any] = {}
    mismatches: list[str] = []
    for artifact, expected in payload["artifacts"].items():
        actual_artifact, matched = _artifact_hash(root / artifact, expected)
        records[artifact] = {
            "expected_sha256": expected,
            "actual_sha256": actual_artifact,
            "matched": matched,
        }
        if not matched:
            mismatches.append(artifact)
    if mismatches:
        raise RuntimeError(f"Frozen artifacts changed: {mismatches}")
    return {"manifest_sha256": actual, "records": records}


def verify_frozen_sources(
    config: Phase7CR2F2Config, root: Path
) -> dict[str, Any]:
    sources = config.section("sources")
    r2f = load_phase7cr2f_config(root / sources["phase7cr2f_config"])
    r2_manifest_path = root / r2f.sources["r2_freeze_manifest"]
    r2_manifest_hash = _sha256(r2_manifest_path)
    if r2_manifest_hash != r2f.sources["r2_freeze_manifest_sha256"]:
        raise RuntimeError("R2 frozen manifest hash mismatch")
    r2_manifest = json.loads(r2_manifest_path.read_text(encoding="utf-8"))
    if r2_manifest["status"] != "strict_stop_failed":
        raise RuntimeError("R2 failure evidence status changed")
    r2_records: dict[str, Any] = {}
    r2_mismatches: list[str] = []
    for artifact, expected in r2_manifest["artifacts"].items():
        actual_artifact, matched = _artifact_hash(root / artifact, expected)
        r2_records[artifact] = {
            "expected_sha256": expected,
            "actual_sha256": actual_artifact,
            "matched": matched,
        }
        if not matched:
            r2_mismatches.append(artifact)
    if r2_mismatches:
        raise RuntimeError(f"R2 frozen artifacts changed: {r2_mismatches}")
    r2_verification = {
        "manifest_sha256": r2_manifest_hash,
        "status_preserved": True,
        "records": r2_records,
    }
    audit_config = yaml.safe_load(
        (root / sources["residual_audit_config"]).read_text(encoding="utf-8")
    )
    r2f_verification = verify_frozen_r2f(audit_config, root)
    audit_verification = _verify_manifest(
        root,
        sources["residual_audit_freeze_manifest"],
        sources["residual_audit_freeze_manifest_sha256"],
        "diagnostic_complete",
    )
    audit_manifest = json.loads(
        (root / sources["residual_audit_freeze_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    decision = audit_manifest["frozen_decision"]
    if decision["residual_initialization"] != "measured":
        raise RuntimeError("Residual initialization contract is not measured")
    if decision["guard_structure"] != "boot_steps_0_1_and_running_steps_2_plus":
        raise RuntimeError("Two-stage residual guard contract changed")
    contract = config.section("control_contract")
    if contract["residual_initialization"] != "measured":
        raise RuntimeError("F2 must use measured residual initialization")
    if contract["allow_zero_residual_fallback"]:
        raise RuntimeError("Zero residual fallback is forbidden")
    if contract["r3_generation_authorized"] or contract["ann_execution_authorized"]:
        raise RuntimeError("F2 cannot authorize R3 or ANN execution")
    return {
        "r2": r2_verification,
        "r2f": r2f_verification,
        "residual_audit": audit_verification,
    }


def guard_for_step(
    temperature_c: float,
    step_index: int,
    guards: dict[str, Any],
    boot_steps: tuple[int, ...] = (0, 1),
) -> tuple[float, str]:
    token = str(int(round(temperature_c)))
    if token != "30":
        return float(guards[token]), "temperature_constant"
    if step_index in boot_steps:
        return float(guards["30"]["boot_v"]), "boot"
    return float(guards["30"]["running_v"]), "running"


def derive_two_stage_guards(
    voltage: dict[str, Any], development: pd.DataFrame
) -> dict[str, Any]:
    boot = development[development.step_index.isin([0, 1])]
    running = development[development.step_index >= 2]
    if boot.empty or running.empty:
        raise RuntimeError("Development trajectories do not cover both guard stages")
    boot_max = float(boot.positive_residual_growth_v.max())
    running_max = float(running.positive_residual_growth_v.max())
    margin = float(voltage["engineering_margin_v"])
    boot_guard = max(
        float(voltage["boot_historical_minimum_30c_v"]), boot_max
    ) + margin
    running_guard = max(
        float(voltage["running_guard_floor_30c_v"]), running_max + margin
    )
    return {
        "15": float(voltage["guard_15c_v"]),
        "25": float(voltage["guard_25c_v"]),
        "30": {"boot_v": boot_guard, "running_v": running_guard},
        "development_boot_max_v": boot_max,
        "development_running_max_v": running_max,
        "engineering_margin_v": margin,
    }


def _state_columns() -> list[str]:
    return [
        "initial_soc",
        "initial_polarization_1_v",
        "initial_polarization_2_v",
        "initial_previous_current_a",
    ]


def _all_historical_states(
    config: Phase7CR2F2Config, root: Path
) -> np.ndarray:
    sources = config.section("sources")
    paths = list(sources["legacy_phase7c_states"].values()) + [
        sources["legacy_phase7cr1_states"],
        sources["legacy_phase7cr2_development_states"],
        sources["legacy_phase7cr2_internal_states"],
        sources["legacy_phase7cr2f_development_states"],
        sources["legacy_phase7cr2f_internal_states"],
        sources["legacy_phase7b1_25c_states"],
    ]
    return pd.concat([pd.read_csv(root / p) for p in paths], ignore_index=True)[
        _state_columns()
    ].to_numpy(float)


def _historical_centers(
    config: Phase7CR2F2Config, root: Path
) -> list[tuple[np.ndarray, str]]:
    records = []
    for item in config.section("datasets")["historical_centers"]:
        frame = pd.read_csv(root / item["source"])
        match = frame[frame.trajectory_id == item["trajectory_id"]]
        if len(match) != 1:
            raise RuntimeError(f"Historical center not unique: {item}")
        records.append(
            (match.iloc[0][_state_columns()].to_numpy(float), item["evidence"])
        )
    return records


def _scrambled_unit(index: int, seed: int) -> np.ndarray:
    phase = np.random.default_rng(seed).random(4)
    return np.mod(
        np.asarray(
            [_van_der_corput(index, base) for base in (2, 3, 5, 7)],
            dtype=float,
        )
        + phase,
        1.0,
    )


def _bounded_candidate(
    index: int, seed: int, bounds: dict[str, list[float]]
) -> np.ndarray:
    unit = _scrambled_unit(index, seed)
    keys = ("soc", "v1_v", "v2_v", "previous_current_a")
    lower = np.asarray([bounds[key][0] for key in keys], dtype=float)
    upper = np.asarray([bounds[key][1] for key in keys], dtype=float)
    return lower + unit * (upper - lower)


def _neighborhood_candidate(
    index: int,
    seed: int,
    center: np.ndarray,
    config: Phase7CR2F2Config,
) -> np.ndarray:
    datasets = config.section("datasets")
    widths = datasets["neighborhood_half_width"]
    half = np.asarray(
        [widths["soc"], widths["v1_v"], widths["v2_v"], widths["previous_current_a"]],
        dtype=float,
    )
    candidate = center + (2.0 * _scrambled_unit(index, seed) - 1.0) * half
    global_bounds = datasets["bounds"]["global_coverage"]
    keys = ("soc", "v1_v", "v2_v", "previous_current_a")
    lower = np.asarray([global_bounds[key][0] for key in keys], dtype=float)
    upper = np.asarray([global_bounds[key][1] for key in keys], dtype=float)
    return np.clip(candidate, lower, upper)


def _initial_measurement(
    config: Phase7CR2F2Config,
    root: Path,
    initial: dict[str, Any],
) -> float:
    r2f = load_phase7cr2f_config(
        root / config.section("sources")["phase7cr2f_config"]
    )
    _, _, level3, _, _, phase7b0 = _context(r2f, root)
    plant = Chen2020ThermalDFN(
        phase7b0.dfn.parameter_set,
        float(initial["ambient_temperature_c"]),
        phase7b0.dfn.upper_voltage_cutoff_v,
        float(initial["initial_soc"]),
        level3.model.sample_period_s,
        "lumped",
    )
    solution = plant.simulation.solve(
        [0.0, float(config.section("control_contract")["initial_measurement_horizon_s"])],
        inputs={
            "phase7c_applied_current_a": -float(
                initial["initial_previous_current_a"]
            )
        },
    )
    values = np.asarray(solution["Terminal voltage [V]"].entries).reshape(-1)
    if not len(values) or not np.isfinite(values[0]):
        raise RuntimeError("Initial DFN voltage measurement unavailable")
    return float(values[0])


def _build_role_states(
    config: Phase7CR2F2Config,
    root: Path,
    role: str,
    existing: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    datasets = config.section("datasets")
    r2f = load_phase7cr2f_config(
        root / config.section("sources")["phase7cr2f_config"]
    )
    r1, _, level3, _, model, _ = _context(r2f, root)
    centers = _historical_centers(config, root)
    counts = datasets["strata"][role]
    cursor = int(datasets[f"{role}_design_start_index"])
    seed = int(datasets[f"{role}_seed"])
    voltage_limit = (
        model.inherited.mpc.terminal_voltage_max_v
        - float(datasets["initial_voltage_margin_v"])
    )
    records: list[dict[str, Any]] = []
    for stratum in (
        "startup_transient_stress",
        "historical_extreme_neighborhood",
        "general_high_risk",
        "global_coverage",
    ):
        accepted = 0
        attempts = 0
        while accepted < int(counts[stratum]):
            attempts += 1
            if attempts > 10000:
                raise RuntimeError(f"Unable to fill F2 stratum: {role}/{stratum}")
            used = cursor
            evidence = "bounded_design"
            center_index = -1
            if stratum == "historical_extreme_neighborhood":
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
                "ambient_temperature_c": float(datasets["temperature_c"]),
                "initial_temperature_c": float(datasets["temperature_c"]),
                "initial_soc": values[0],
                "initial_polarization_1_v": values[1],
                "initial_polarization_2_v": values[2],
                "initial_previous_current_a": values[3],
            }
            measured = _initial_measurement(config, root, initial)
            predicted = model.terminal_voltage(state, state.previous_current_a)
            trajectory_id = f"phase7cr2f2_{role}_30c_{len(records):03d}"
            records.append(
                {
                    "trajectory_id": trajectory_id,
                    "role": role,
                    "ambient_temperature_c": initial["ambient_temperature_c"],
                    "initial_temperature_c": initial["initial_temperature_c"],
                    "risk_stratum": stratum,
                    "historical_center_index": center_index,
                    "historical_evidence": evidence,
                    "initial_soc": values[0],
                    "initial_polarization_1_v": values[1],
                    "initial_polarization_2_v": values[2],
                    "initial_previous_current_a": values[3],
                    "initial_dfn_voltage_v": measured,
                    "initial_2rc_voltage_v": predicted,
                    "initial_measured_residual_v": measured - predicted,
                    "initial_residual_extreme": bool(
                        measured - predicted
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
    config: Phase7CR2F2Config, root: Path
) -> dict[str, Any]:
    verification = verify_frozen_sources(config, root)
    data_dir = root / config.section("output")["data_directory"]
    data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = data_dir / "initial_state_freeze.json"
    if freeze_path.exists():
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
        for name, record in payload["files"].items():
            if _sha256(data_dir / name) != record["sha256"]:
                raise RuntimeError(f"Frozen F2 state hash mismatch: {name}")
        return payload
    for name in ("development_initial_states.csv", "internal_validation_initial_states.csv"):
        if (data_dir / name).exists():
            raise RuntimeError("Unfrozen F2 state file already exists")
    existing = _all_historical_states(config, root)
    development, existing = _build_role_states(
        config, root, "development", existing
    )
    internal, _ = _build_role_states(
        config, root, "internal_validation", existing
    )
    if set(development.trajectory_id) & set(internal.trajectory_id):
        raise RuntimeError("Development and internal validation overlap")
    files: dict[str, Any] = {}
    for name, frame in (
        ("development_initial_states.csv", development),
        ("internal_validation_initial_states.csv", internal),
    ):
        path = data_dir / name
        frame.to_csv(path, index=False)
        files[name] = {
            "sha256": _sha256(path),
            "trajectory_count": len(frame),
            "strata_counts": frame.risk_stratum.value_counts().to_dict(),
            "minimum_initial_residual_v": float(
                frame.initial_measured_residual_v.min()
            ),
            "extreme_initial_residual_count": int(
                frame.initial_residual_extreme.astype(bool).sum()
            ),
            "design_seed": int(frame.design_seed.iloc[0]),
            "minimum_design_index": int(frame.design_candidate_index.min()),
            "maximum_design_index": int(frame.design_candidate_index.max()),
        }
    payload = {
        "phase": "Phase 7C-R2F2",
        "status": "initial_states_frozen_before_any_closed_loop_rollout",
        "frozen_before_any_closed_loop_rollout": True,
        "development_internal_isolation": True,
        "not_r3_confirmation_data": True,
        "not_ann_teacher_data": True,
        "initial_state_history_contract": (
            "dfn_and_2rc_do_not_share_current_history"
        ),
        "config_sha256": _sha256(
            root / "configs/phase7cr2f2_two_stage_voltage_guard.yaml", True
        ),
        "files": files,
        "frozen_source_verification": verification,
    }
    freeze_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _rollout(
    config: Phase7CR2F2Config,
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
    config: Phase7CR2F2Config,
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
    config: Phase7CR2F2Config,
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
                    f"[Phase 7C-R2F2:{run_group}:{role}] "
                    f"{completed}/{len(pending)}",
                    flush=True,
                )
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _load_new_rows(
    config: Phase7CR2F2Config, root: Path, role: str
) -> list[dict[str, Any]]:
    path = (
        root
        / config.section("output")["data_directory"]
        / f"{role}_initial_states.csv"
    )
    return pd.read_csv(path).to_dict(orient="records")


def _prefix(
    frame: pd.DataFrame, role: str, temperature_c: float
) -> list[dict[str, Any]]:
    frame = frame.copy()
    frame["source_trajectory_id"] = frame["trajectory_id"]
    frame["trajectory_id"] = [
        f"f2_{role}_{int(temperature_c)}c_{index:03d}"
        for index in range(len(frame))
    ]
    frame["ambient_temperature_c"] = temperature_c
    frame["initial_temperature_c"] = temperature_c
    return frame.to_dict(orient="records")


def _historical_roles(
    config: Phase7CR2F2Config, root: Path
) -> list[tuple[str, list[dict[str, Any]]]]:
    sources = config.section("sources")
    roles: list[tuple[str, list[dict[str, Any]]]] = []
    phase7c_rows: list[dict[str, Any]] = []
    for temperature, relative in sources["legacy_phase7c_states"].items():
        phase7c_rows.extend(
            _prefix(pd.read_csv(root / relative), "legacy_phase7c", float(temperature))
        )
    roles.append(("legacy_phase7c", phase7c_rows))
    roles.append(
        (
            "legacy_phase7cr1",
            _prefix(
                pd.read_csv(root / sources["legacy_phase7cr1_states"]),
                "legacy_phase7cr1",
                30.0,
            ),
        )
    )
    for role, key in (
        ("legacy_phase7cr2_development", "legacy_phase7cr2_development_states"),
        ("legacy_phase7cr2_internal", "legacy_phase7cr2_internal_states"),
        ("legacy_phase7cr2f_development", "legacy_phase7cr2f_development_states"),
        ("legacy_phase7cr2f_internal", "legacy_phase7cr2f_internal_states"),
    ):
        frame = pd.read_csv(root / sources[key])
        rows: list[dict[str, Any]] = []
        if "ambient_temperature_c" in frame.columns:
            for temperature, group in frame.groupby("ambient_temperature_c"):
                rows.extend(_prefix(group, role, float(temperature)))
        else:
            rows.extend(_prefix(frame, role, 30.0))
        roles.append((role, rows))
    roles.append(
        (
            "legacy_phase7b1_25c",
            _prefix(
                pd.read_csv(root / sources["legacy_phase7b1_25c_states"]),
                "legacy_phase7b1_25c",
                25.0,
            ),
        )
    )
    return roles


def _verify_state_freeze(
    config: Phase7CR2F2Config, root: Path
) -> dict[str, Any]:
    data_dir = root / config.section("output")["data_directory"]
    path = data_dir / "initial_state_freeze.json"
    if not path.exists():
        raise RuntimeError("F2 states must be prepared and frozen first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, record in payload["files"].items():
        if _sha256(data_dir / name) != record["sha256"]:
            raise RuntimeError(f"Frozen F2 state hash mismatch: {name}")
    return payload


def run_development(
    config: Phase7CR2F2Config, root: Path, resume: bool
) -> dict[str, Any]:
    verify_frozen_sources(config, root)
    state_freeze = _verify_state_freeze(config, root)
    data_dir = root / config.section("output")["data_directory"]
    guard_path = data_dir / "frozen_two_stage_voltage_guards.json"
    if guard_path.exists():
        return _verify_guard_freeze(config, root)
    voltage = config.section("voltage_guard")
    audit_guards = {
        "15": float(voltage["guard_15c_v"]),
        "25": float(voltage["guard_25c_v"]),
        "30": {
            "boot_v": float(voltage["development_audit_boot_guard_30c_v"]),
            "running_v": float(
                voltage["development_audit_running_guard_30c_v"]
            ),
        },
    }
    audit = _run_rows(
        config,
        root,
        _load_new_rows(config, root, "development"),
        "development_guard_audit",
        audit_guards,
        "development_audit",
        resume,
    )
    audit.to_csv(data_dir / "development_guard_audit.csv", index=False)
    guards = derive_two_stage_guards(voltage, audit)
    payload = {
        "phase": "Phase 7C-R2F2 development",
        "status": "two_stage_guards_frozen_before_internal_validation",
        "design": "two_stage_boot_steps_0_1_running_steps_2_plus",
        "residual_initialization": "measured",
        "internal_validation_used_for_tuning": False,
        "formulas": {
            "boot": voltage["formula_boot"],
            "running": voltage["formula_running"],
        },
        "guards": guards,
        "state_hashes": {
            name: record["sha256"]
            for name, record in state_freeze["files"].items()
        },
        "development_audit_sha256": _sha256(
            data_dir / "development_guard_audit.csv"
        ),
        "config_sha256": _sha256(
            root / "configs/phase7cr2f2_two_stage_voltage_guard.yaml", True
        ),
        "runner_sha256": _sha256(Path(__file__), True),
    }
    guard_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _verify_guard_freeze(
    config: Phase7CR2F2Config, root: Path
) -> dict[str, Any]:
    data_dir = root / config.section("output")["data_directory"]
    path = data_dir / "frozen_two_stage_voltage_guards.json"
    if not path.exists():
        raise RuntimeError("Two-stage guards must be developed and frozen first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["internal_validation_used_for_tuning"]:
        raise RuntimeError("Internal validation cannot tune F2 guards")
    if payload["config_sha256"] != _sha256(
        root / "configs/phase7cr2f2_two_stage_voltage_guard.yaml", True
    ):
        raise RuntimeError("F2 config changed after guard freeze")
    if payload["runner_sha256"] != _sha256(Path(__file__), True):
        raise RuntimeError("F2 runner changed after guard freeze")
    if payload["development_audit_sha256"] != _sha256(
        data_dir / "development_guard_audit.csv"
    ):
        raise RuntimeError("F2 development audit changed after guard freeze")
    return payload


def _extended_checks(
    config: Phase7CR2F2Config, metrics: pd.DataFrame, rows: pd.DataFrame
) -> dict[str, bool]:
    r2f = load_phase7cr2f_config(
        Path.cwd() / config.section("sources")["phase7cr2f_config"]
    )
    checks = _checks(r2f, metrics)
    gates = config.section("gates")
    checks.update(
        {
            "maximum_voltage_f2": bool(
                metrics.maximum_voltage_v.max() <= gates["maximum_voltage_v"]
            ),
            "maximum_temperature_f2": bool(
                metrics.maximum_temperature_c.max()
                <= gates["maximum_average_temperature_c"]
            ),
            "zero_boot_guard_exceedance": bool(
                not rows.loc[rows.guard_stage == "boot", "guard_exceeded"]
                .astype(bool)
                .any()
            ),
            "zero_running_guard_exceedance": bool(
                not rows.loc[rows.guard_stage == "running", "guard_exceeded"]
                .astype(bool)
                .any()
            ),
            "measured_initialization_only": bool(
                (rows.residual_initialization_mode == "measured").all()
                and rows.initial_residual_v.notna().all()
            ),
            "zero_strict_teacher_failure": bool(
                not rows.strict_teacher_failure.astype(bool).any()
            ),
        }
    )
    return checks


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    guards = payload["frozen_guard_contract"]["guards"]
    summary = payload["global_summary"]
    rows = []
    for role, temperatures in payload["role_results"].items():
        for temperature, result in temperatures.items():
            item = result["summary"]
            rows.append(
                f"| {role} | {temperature} | {item['trajectory_count']} | "
                f"{100 * item['target_reach_fraction']:.1f}% | "
                f"{item['maximum_voltage_v']:.6f} | "
                f"{item['maximum_temperature_c']:.6f} | "
                f"{item['guard_exceedance_count']} | "
                f"{'通过' if result['success'] else '失败'} |"
            )
    conclusion = (
        "F2严格通过；完整架构可以冻结，但本阶段仍未生成R3初态，也未运行ANN。"
        if payload["success"]
        else "F2触发严格停止；新内部验证永久降级为回归证据，不得原地调参、生成R3或运行ANN。"
    )
    report = f"""# Phase 7C-R2F2 两段电压裕量实验报告

## 冻结控制合同

初始残差由时刻0测量初始化。30 ℃在控制步0和1使用启动裕量，从控制步2开始使用运行裕量；控制周期为5 s。DFN与2RC初态继续不共享电流历史。教师选择、R1热监督层以及15 ℃、25 ℃裕量保持冻结。

## 冻结裕量

- 30 ℃启动裕量：{1000 * guards['30']['boot_v']:.6f} mV；
- 30 ℃运行裕量：{1000 * guards['30']['running_v']:.6f} mV；
- 开发集启动期最大正向增长：{1000 * guards['development_boot_max_v']:.6f} mV；
- 开发集运行期最大正向增长：{1000 * guards['development_running_max_v']:.6f} mV；
- 15 ℃冻结裕量：{1000 * guards['15']:.6f} mV；
- 25 ℃冻结裕量：{1000 * guards['25']:.6f} mV。

## 一次性验证与历史回归

| 数据角色 | 温度/℃ | 轨迹数 | 到达率 | 最高电压/V | 最高平均温度/℃ | 裕量超越 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## 全局结果

- 轨迹数：{summary['trajectory_count']}；
- 目标到达率：{100 * summary['target_reach_fraction']:.1f}%；
- 最高DFN电压：{summary['maximum_voltage_v']:.6f} V；
- 最高平均温度：{summary['maximum_temperature_c']:.6f} ℃；
- 启动期/运行期裕量超越：{summary['boot_guard_exceedance_count']}/{summary['running_guard_exceedance_count']}；
- 严格教师失败/求解失败/预测不可行：{summary['strict_teacher_failure_count']}/{summary['solver_failure_count']}/{summary['prediction_infeasible_count']}；
- 电压—斜率/热—斜率空区间：{summary['empty_voltage_slew_count']}/{summary['empty_thermal_slew_count']}；
- 持续振荡：{summary['sustained_oscillation_count']}；
- 零残差初始化轨迹：{summary['zero_residual_initialization_count']}。

## 判定

{conclusion}
"""
    path.write_text(report, encoding="utf-8")


def run_validation(
    config: Phase7CR2F2Config, root: Path, resume: bool
) -> dict[str, Any]:
    source_verification = verify_frozen_sources(config, root)
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
            "One-shot validation already started; use --resume only to finish it"
        )
    if not started_path.exists():
        started = {
            "status": "one_shot_validation_started",
            "guards_sha256": _sha256(
                data_dir / "frozen_two_stage_voltage_guards.json"
            ),
            "internal_state_sha256": state_freeze["files"]
            ["internal_validation_initial_states.csv"]["sha256"],
            "internal_validation_may_not_modify_guards": True,
        }
        started_path.write_text(
            json.dumps(started, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    guards = guard_freeze["guards"]
    frames = [
        _run_rows(
            config,
            root,
            _load_new_rows(config, root, "development"),
            "development_frozen_guard",
            guards,
            "validation",
            resume,
        )
    ]
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
            _load_new_rows(config, root, "internal_validation"),
            "internal_validation",
            guards,
            "validation",
            resume,
        )
    )
    final = pd.concat(frames, ignore_index=True)
    final.to_csv(data_dir / "combined_validation_trajectories.csv", index=False)
    r2f = load_phase7cr2f_config(
        root / config.section("sources")["phase7cr2f_config"]
    )
    trajectory_metrics = _trajectory_metrics(r2f, final)
    trajectory_metrics.to_csv(data_dir / "trajectory_metrics.csv", index=False)
    teacher_regression = verify_known_teacher_regressions(r2f, root)
    role_results: dict[str, Any] = {}
    success = bool(teacher_regression["all_passed"])
    for role, role_metrics in trajectory_metrics.groupby("role"):
        temperatures: dict[str, Any] = {}
        role_rows = final[final.role == role]
        for temperature, group in role_metrics.groupby(
            "ambient_temperature_c"
        ):
            temperature_rows = role_rows[
                np.isclose(role_rows.ambient_temperature_c, temperature)
            ]
            checks = _extended_checks(config, group, temperature_rows)
            result = {
                "summary": _summarize_group(group),
                "checks": checks,
                "success": bool(all(checks.values())),
            }
            temperatures[str(int(temperature))] = result
            success = success and result["success"]
        role_results[role] = temperatures
    boot_exceedances = final[
        (final.guard_stage == "boot") & final.guard_exceeded.astype(bool)
    ]
    running_exceedances = final[
        (final.guard_stage == "running") & final.guard_exceeded.astype(bool)
    ]
    strict_failures = final[final.strict_teacher_failure.astype(bool)]
    zero_initialization_count = int(
        final.groupby("trajectory_id").residual_initialization_mode.first()
        .ne("measured")
        .sum()
    )
    known_r2f = final[
        final.source_trajectory_id.astype(str)
        == "phase7cr2f_internal_validation_30c_014"
    ]
    known_r2f_covered = bool(
        len(known_r2f)
        and not known_r2f.guard_exceeded.astype(bool).any()
        and float(guards["30"]["boot_v"]) >= 0.0250015270785617
    )
    success = success and known_r2f_covered and zero_initialization_count == 0
    global_summary = {
        "trajectory_count": int(trajectory_metrics.trajectory_id.nunique()),
        "target_reach_fraction": float(trajectory_metrics.target_reached.mean()),
        "maximum_voltage_v": float(trajectory_metrics.maximum_voltage_v.max()),
        "maximum_temperature_c": float(
            trajectory_metrics.maximum_temperature_c.max()
        ),
        "maximum_current_step_a": float(
            trajectory_metrics.maximum_current_step_a.max()
        ),
        "boot_guard_exceedance_count": int(len(boot_exceedances)),
        "running_guard_exceedance_count": int(len(running_exceedances)),
        "strict_teacher_failure_count": int(len(strict_failures)),
        "solver_failure_count": int(
            trajectory_metrics.solver_failure_count.sum()
        ),
        "prediction_infeasible_count": int(
            trajectory_metrics.prediction_infeasible_count.sum()
        ),
        "empty_voltage_slew_count": int(
            trajectory_metrics.empty_voltage_slew_count.sum()
        ),
        "empty_thermal_slew_count": int(
            trajectory_metrics.empty_thermal_slew_count.sum()
        ),
        "sustained_oscillation_count": int(
            trajectory_metrics.sustained_oscillation_count.sum()
        ),
        "zero_residual_initialization_count": zero_initialization_count,
    }
    payload = {
        "study_name": config.study_name,
        "configuration": config.payload,
        "frozen_source_verification": source_verification,
        "initial_state_freeze": state_freeze,
        "frozen_guard_contract": guard_freeze,
        "teacher_selection_regression": teacher_regression,
        "r2f_25_001527_mv_regression_covered": known_r2f_covered,
        "role_results": role_results,
        "global_summary": global_summary,
        "success": bool(success),
        "decision": {
            "freeze_full_f2_architecture": bool(success),
            "eligible_to_design_r3_separately": bool(success),
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
    _write_report(result_dir / "PHASE7C-R2F2_中文实验报告.md", payload)
    artifacts = [
        "configs/phase7cr2f2_two_stage_voltage_guard.yaml",
        "src/battery_fast_charge/phase7cr2f2_config.py",
        "src/battery_fast_charge/phase7cr2f2_runner.py",
        "src/battery_fast_charge/phase7cr2f2_cli.py",
        "data/phase7cr2f2_two_stage_voltage_guard/initial_state_freeze.json",
        "data/phase7cr2f2_two_stage_voltage_guard/development_initial_states.csv",
        "data/phase7cr2f2_two_stage_voltage_guard/internal_validation_initial_states.csv",
        "data/phase7cr2f2_two_stage_voltage_guard/frozen_two_stage_voltage_guards.json",
        "data/phase7cr2f2_two_stage_voltage_guard/development_guard_audit.csv",
        "data/phase7cr2f2_two_stage_voltage_guard/combined_validation_trajectories.csv",
        "data/phase7cr2f2_two_stage_voltage_guard/trajectory_metrics.csv",
        "outputs/phase7cr2f2_two_stage_voltage_guard/metrics.json",
        "outputs/phase7cr2f2_two_stage_voltage_guard/PHASE7C-R2F2_中文实验报告.md",
    ]
    manifest = {
        "phase": "Phase 7C-R2F2",
        "status": "strict_passed" if success else "strict_stop_failed",
        "internal_validation_used_for_retuning": False,
        "r3_initial_states_generated": False,
        "ann_execution_authorized": False,
        "artifacts": {
            relative: _sha256(root / relative, Path(relative).suffix in {".py", ".yaml", ".md"})
            for relative in artifacts
        },
    }
    (result_dir / "freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def run_phase7cr2f2(
    config: Phase7CR2F2Config,
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
