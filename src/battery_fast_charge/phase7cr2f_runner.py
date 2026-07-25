"""Phase 7C-R2F：教师选择修复与30 ℃电压裕量再开发。"""

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

from .phase7a_level3_model import Level3MPCResult, Level3State
from .phase7b1b_runner import _maximum_safe_current
from .phase7c_runner import Chen2020ThermalDFN, _van_der_corput
from .phase7cr2_runner import (
    _checks,
    _context,
    _summarize_group,
    _thermal_current_limit,
    _trajectory_metrics,
)
from .phase7cr2f_config import Phase7CR2FConfig
from .phase7cr2f_teacher import (
    StrictTeacherSelectionError,
    select_qualified_teacher_candidate,
    solve_teacher_r2f,
)


def _sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".py", ".yaml", ".yml"}:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def verify_frozen_artifacts(
    config: Phase7CR2FConfig, root: Path
) -> dict[str, Any]:
    manifest_path = root / config.sources["r2_freeze_manifest"]
    manifest_hash = _sha256(manifest_path)
    if manifest_hash != config.sources["r2_freeze_manifest_sha256"]:
        raise RuntimeError("R2冻结清单自身哈希不匹配。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {}
    mismatches = []
    for relative, expected in manifest["artifacts"].items():
        actual = _sha256(root / relative)
        records[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": actual == expected,
        }
        if actual != expected:
            mismatches.append(relative)
    extra = {
        config.sources["teacher_selection_source"]: config.sources[
            "teacher_selection_source_sha256"
        ],
        config.sources["teacher_regression_fixture"]: config.sources[
            "teacher_regression_fixture_sha256"
        ],
    }
    for relative, expected in extra.items():
        actual = _sha256(root / relative)
        records[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": actual == expected,
        }
        if actual != expected:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"R2F冻结工件不匹配：{mismatches}")
    if manifest["status"] != "strict_stop_failed":
        raise RuntimeError("R2证据不得被改写为通过。")
    if not all(config.teacher.values()):
        raise RuntimeError("R2F教师严格资格合同不完整。")
    return {
        "r2_manifest_sha256": manifest_hash,
        "r2_status_preserved": True,
        "records": records,
    }


def _state_columns() -> list[str]:
    return [
        "initial_soc",
        "initial_polarization_1_v",
        "initial_polarization_2_v",
        "initial_previous_current_a",
    ]


def _historical_frames(
    config: Phase7CR2FConfig, root: Path
) -> list[pd.DataFrame]:
    paths = list(config.sources["legacy_phase7c_states"].values()) + [
        config.sources["legacy_phase7cr1_states"],
        config.sources["legacy_phase7cr2_development_states"],
        config.sources["legacy_phase7cr2_internal_states"],
    ]
    return [pd.read_csv(root / path) for path in paths]


def _historical_values(
    config: Phase7CR2FConfig, root: Path
) -> np.ndarray:
    return pd.concat(_historical_frames(config, root), ignore_index=True)[
        _state_columns()
    ].to_numpy(float)


def _historical_centers(
    config: Phase7CR2FConfig, root: Path
) -> list[np.ndarray]:
    centers = []
    for item in config.datasets["historical_centers"]:
        frame = pd.read_csv(root / item["source"])
        match = frame[frame.trajectory_id == item["trajectory_id"]]
        if len(match) != 1:
            raise RuntimeError(f"历史邻域中心不存在或不唯一：{item}")
        centers.append(match.iloc[0][_state_columns()].to_numpy(float))
    return centers


def _scaled_candidate(
    index: int, bounds: dict[str, list[float]]
) -> np.ndarray:
    names = ["soc", "v1_v", "v2_v", "previous_current_a"]
    bases = [2, 3, 5, 7]
    values = []
    for name, base in zip(names, bases, strict=True):
        lower, upper = bounds[name]
        values.append(
            lower + (upper - lower) * _van_der_corput(index, base)
        )
    return np.asarray(values, dtype=float)


def _neighborhood_candidate(
    index: int,
    center: np.ndarray,
    config: Phase7CR2FConfig,
) -> np.ndarray:
    widths = config.datasets["historical_neighborhood_half_width"]
    half_width = np.asarray(
        [
            widths["soc"],
            widths["v1_v"],
            widths["v2_v"],
            widths["previous_current_a"],
        ]
    )
    unit = np.asarray(
        [_van_der_corput(index, base) for base in (2, 3, 5, 7)]
    )
    candidate = center + (2.0 * unit - 1.0) * half_width
    general = config.datasets["general_bounds"]
    lower = np.asarray(
        [
            general["soc"][0],
            general["v1_v"][0],
            general["v2_v"][0],
            6.0,
        ]
    )
    upper = np.asarray(
        [
            general["soc"][1],
            general["v1_v"][1],
            general["v2_v"][1],
            general["previous_current_a"][1],
        ]
    )
    return np.clip(candidate, lower, upper)


def _build_role_states(
    config: Phase7CR2FConfig,
    root: Path,
    role: str,
    counts: dict[str, int],
    start_index: int,
    seed: int,
    existing: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    r1, _, level3, _, model, _ = _context(config, root)
    centers = _historical_centers(config, root)
    voltage_limit = (
        model.inherited.mpc.terminal_voltage_max_v
        - config.datasets["initial_voltage_margin_v"]
    )
    records = []
    cursor = int(start_index)
    for stratum in ("general", "high_risk", "historical_neighborhood"):
        accepted = 0
        while accepted < int(counts[stratum]):
            used = cursor
            if stratum == "historical_neighborhood":
                center_index = accepted % len(centers)
                values = _neighborhood_candidate(
                    cursor, centers[center_index], config
                )
            else:
                center_index = -1
                values = _scaled_candidate(
                    cursor,
                    config.datasets[
                        "general_bounds"
                        if stratum == "general"
                        else "high_risk_bounds"
                    ],
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
            trajectory_id = f"phase7cr2f_{role}_30c_{len(records):03d}"
            records.append(
                {
                    "trajectory_id": trajectory_id,
                    "role": role,
                    "ambient_temperature_c": 30.0,
                    "initial_temperature_c": 30.0,
                    "risk_stratum": stratum,
                    "historical_center_index": center_index,
                    "initial_soc": values[0],
                    "initial_polarization_1_v": values[1],
                    "initial_polarization_2_v": values[2],
                    "initial_previous_current_a": values[3],
                    "design_candidate_index": used,
                    "design_seed": seed,
                }
            )
            existing = np.vstack([existing, values])
            accepted += 1
    return pd.DataFrame(records), existing


def freeze_new_states(
    config: Phase7CR2FConfig, root: Path
) -> dict[str, Any]:
    data_dir = root / config.output["data_directory"]
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = _historical_values(config, root)
    development, existing = _build_role_states(
        config,
        root,
        "development",
        config.datasets["strata"]["development"],
        config.datasets["development_design_start_index"],
        config.datasets["development_seed"],
        existing,
    )
    internal, _ = _build_role_states(
        config,
        root,
        "internal_validation",
        config.datasets["strata"]["internal_validation"],
        config.datasets["internal_design_start_index"],
        config.datasets["internal_validation_seed"],
        existing,
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
            "strata_counts": frame.risk_stratum.value_counts().to_dict(),
        }
    freeze = {
        "frozen_before_any_rollout": True,
        "development_internal_trajectory_isolation": True,
        "temperature_c": 30.0,
        "not_r3_confirmation": True,
        "not_ann_teacher_data": True,
        "ann_execution_authorized": False,
        "guard_design": config.voltage["design"],
        "guard_design_frozen_before_internal_validation": True,
        "files": files,
        "source_verification": verify_frozen_artifacts(config, root),
    }
    (data_dir / "initial_state_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return freeze


def _strict_failure_row(
    role: str,
    initial: dict[str, Any],
    step: int,
    state: Level3State,
    temperature_c: float,
    guard_v: float,
    error: StrictTeacherSelectionError,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "role": role,
                "trajectory_id": initial["trajectory_id"],
                "source_trajectory_id": initial.get(
                    "source_trajectory_id", initial["trajectory_id"]
                ),
                "ambient_temperature_c": initial[
                    "ambient_temperature_c"
                ],
                "risk_stratum": initial.get("risk_stratum", "legacy"),
                "step_index": step,
                "time_s": np.nan,
                "soc": state.soc,
                "temperature_c": temperature_c,
                "previous_current_a": state.previous_current_a,
                "candidate_current_a": np.nan,
                "slew_lower_a": np.nan,
                "slew_upper_a": np.nan,
                "guard_v": guard_v,
                "voltage_safe_current_max_a": np.nan,
                "constant_thermal_safe_current_max_a": np.nan,
                "braking_thermal_safe_current_max_a": np.nan,
                "thermal_safe_current_max_a": np.nan,
                "predicted_300s_peak_temperature_c": np.nan,
                "thermal_recovery_active": False,
                "final_upper_a": np.nan,
                "empty_voltage_slew_interval": False,
                "empty_thermal_slew_interval": False,
                "empty_final_interval": False,
                "teacher_time_s": np.nan,
                "supervisor_time_s": np.nan,
                "teacher_retry_triggered": True,
                "selected_teacher_branch": "none",
                "qualified_teacher_branches": "[]",
                "default_optimizer_success": error.diagnostics.get(
                    "default", {}
                ).get("optimizer_success", False),
                "default_prediction_feasible": error.diagnostics.get(
                    "default", {}
                ).get("prediction_feasible", False),
                "default_current_a": np.nan,
                "default_objective": error.diagnostics.get(
                    "default", {}
                ).get("objective_value", np.nan),
                "alternative_optimizer_success": error.diagnostics.get(
                    "alternative", {}
                ).get("optimizer_success", False),
                "alternative_prediction_feasible": error.diagnostics.get(
                    "alternative", {}
                ).get("prediction_feasible", False),
                "alternative_current_a": np.nan,
                "alternative_objective": error.diagnostics.get(
                    "alternative", {}
                ).get("objective_value", np.nan),
                "alternative_selected": False,
                "teacher_branch_objective_improvement": np.nan,
                "current_a": np.nan,
                "current_step_a": np.nan,
                "next_soc": state.soc,
                "next_temperature_c": temperature_c,
                "terminal_voltage_v": np.nan,
                "predicted_voltage_v": np.nan,
                "voltage_residual_before_v": np.nan,
                "voltage_residual_after_v": np.nan,
                "positive_residual_growth_v": np.nan,
                "guard_exceeded": False,
                "voltage_intervened": False,
                "thermal_intervened": False,
                "both_layers_active": False,
                "voltage_current_correction_a": np.nan,
                "thermal_current_correction_a": np.nan,
                "optimizer_success": False,
                "prediction_feasible": False,
                "strict_teacher_failure": True,
            }
        ]
    )


def _rollout(
    config: Phase7CR2FConfig,
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
        try:
            result, teacher = solve_teacher_r2f(state, model, r1)
        except StrictTeacherSelectionError as error:
            failure = _strict_failure_row(
                role,
                initial,
                step,
                state,
                temperature_c,
                guard_v,
                error,
            )
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
    config: Phase7CR2FConfig,
    root_text: str,
    initial: dict[str, Any],
    role: str,
    guard_v: float,
    path_text: str,
) -> str:
    frame = _rollout(config, Path(root_text), initial, role, guard_v)
    frame.to_csv(path_text, index=False)
    return path_text


def _run_rows(
    config: Phase7CR2FConfig,
    root: Path,
    rows: list[dict[str, Any]],
    role: str,
    guards: dict[str, float],
    resume: bool,
) -> pd.DataFrame:
    run_dir = root / config.output["data_directory"] / "runs" / role
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    pending = []
    for row in rows:
        token = str(int(float(row["ambient_temperature_c"])))
        guard = float(guards[token])
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
                    f"[Phase 7C-R2F:{role}] {completed}/{len(pending)}",
                    flush=True,
                )
    return pd.concat(
        [pd.read_csv(path) for path in paths], ignore_index=True
    )


def _new_rows(
    config: Phase7CR2FConfig, root: Path, role: str
) -> list[dict[str, Any]]:
    return pd.read_csv(
        root
        / config.output["data_directory"]
        / f"{role}_initial_states.csv"
    ).to_dict(orient="records")


def _prefix_legacy(
    frame: pd.DataFrame,
    role: str,
    temperature_c: float,
) -> list[dict[str, Any]]:
    frame = frame.copy()
    frame["source_trajectory_id"] = frame["trajectory_id"]
    frame["trajectory_id"] = [
        f"r2f_{role}_{int(temperature_c)}c_{i:03d}"
        for i in range(len(frame))
    ]
    frame["ambient_temperature_c"] = temperature_c
    frame["initial_temperature_c"] = temperature_c
    return frame.to_dict(orient="records")


def _legacy_roles(
    config: Phase7CR2FConfig, root: Path
) -> list[tuple[str, list[dict[str, Any]]]]:
    roles = []
    phase7c_rows = []
    for temperature, relative in config.sources[
        "legacy_phase7c_states"
    ].items():
        phase7c_rows.extend(
            _prefix_legacy(
                pd.read_csv(root / relative),
                "legacy_phase7c",
                float(temperature),
            )
        )
    roles.append(("legacy_phase7c", phase7c_rows))
    r1 = pd.read_csv(root / config.sources["legacy_phase7cr1_states"])
    roles.append(
        (
            "legacy_phase7cr1",
            _prefix_legacy(r1, "legacy_phase7cr1", 30.0),
        )
    )
    for role, source in (
        (
            "legacy_phase7cr2_development",
            config.sources["legacy_phase7cr2_development_states"],
        ),
        (
            "legacy_phase7cr2_internal",
            config.sources["legacy_phase7cr2_internal_states"],
        ),
    ):
        frame = pd.read_csv(root / source)
        rows = []
        for temperature, group in frame.groupby("ambient_temperature_c"):
            rows.extend(
                _prefix_legacy(group, role, float(temperature))
            )
        roles.append((role, rows))
    return roles


def _derive_and_freeze_guards(
    config: Phase7CR2FConfig,
    root: Path,
    audit: pd.DataFrame,
) -> dict[str, Any]:
    observed = float(audit.positive_residual_growth_v.max())
    history = float(config.voltage["historical_minimum_30c_v"])
    guard_30 = max(history, observed) + float(
        config.voltage["engineering_margin_v"]
    )
    data_dir = root / config.output["data_directory"]
    initial_freeze = json.loads(
        (data_dir / "initial_state_freeze.json").read_text(encoding="utf-8")
    )
    payload = {
        "design": "single_guard",
        "design_selected_before_internal_validation": True,
        "internal_validation_used_for_tuning": False,
        "formula": "max(historical_minimum_30c, new_development_max) + margin",
        "guards_v": {
            "15": float(config.voltage["guard_15c_v"]),
            "25": float(config.voltage["guard_25c_v"]),
            "30": guard_30,
        },
        "historical_minimum_30c_v": history,
        "new_development_maximum_positive_growth_v": observed,
        "engineering_margin_v": float(
            config.voltage["engineering_margin_v"]
        ),
        "new_state_hashes": {
            name: item["sha256"]
            for name, item in initial_freeze["files"].items()
        },
    }
    (data_dir / "frozen_voltage_guards.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _fixture_result(payload: dict[str, Any]) -> Level3MPCResult:
    return Level3MPCResult(
        current_a=float(payload["current_a"]),
        plan_a=np.asarray([payload["current_a"]], dtype=float),
        objective_value=float(payload["objective_value"]),
        optimizer_success=bool(payload["optimizer_success"]),
        prediction_feasible=bool(payload["prediction_feasible"]),
        used_fallback=False,
        status=str(payload["status"]),
        solve_time_s=0.0,
        maximum_voltage_v=4.0,
        minimum_constraint_margin=0.0,
        maximum_current_step_a=0.0,
        minimum_slew_margin_a=0.0,
    )


def verify_known_teacher_regressions(
    config: Phase7CR2FConfig, root: Path
) -> dict[str, Any]:
    cases = json.loads(
        (root / config.sources["teacher_regression_fixture"]).read_text(
            encoding="utf-8"
        )
    )
    records = []
    for case in cases:
        selection = select_qualified_teacher_candidate(
            {
                "default": _fixture_result(case["default"]),
                "alternative": _fixture_result(case["alternative"]),
            }
        )
        records.append(
            {
                "case_id": case["case_id"],
                "selected_branch": selection.label,
                "expected_branch": case["expected_selected_branch"],
                "passed": selection.label
                == case["expected_selected_branch"],
            }
        )
    return {
        "case_count": len(records),
        "records": records,
        "all_passed": all(item["passed"] for item in records),
    }


def _analyze(
    config: Phase7CR2FConfig,
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
    teacher_regression = verify_known_teacher_regressions(config, root)
    role_results = {}
    success = teacher_regression["all_passed"]
    for role, role_group in metrics.groupby("role"):
        temperatures = {}
        for temperature, group in role_group.groupby(
            "ambient_temperature_c"
        ):
            checks = _checks(config, group)
            result = {
                "summary": _summarize_group(group),
                "checks": checks,
                "success": bool(all(checks.values())),
            }
            temperatures[str(int(temperature))] = result
            success = success and result["success"]
        role_results[role] = temperatures
    exceedances = final[final.guard_exceeded.astype(bool)]
    strict_failures = final[final.strict_teacher_failure.astype(bool)]
    known_sources = {
        "phase7cr2_internal_validation_30c_000",
        "phase7c_30c_012",
        "phase7cr1_dev_001",
    }
    known_history = final[
        final.source_trajectory_id.astype(str).isin(known_sources)
    ]
    known_history_passed = bool(
        not known_history.guard_exceeded.astype(bool).any()
    )
    success = success and known_history_passed
    exceedance_columns = [
        "role",
        "trajectory_id",
        "source_trajectory_id",
        "risk_stratum",
        "step_index",
        "soc",
        "temperature_c",
        "previous_current_a",
        "candidate_current_a",
        "current_a",
        "positive_residual_growth_v",
        "guard_v",
        "voltage_residual_before_v",
        "voltage_residual_after_v",
    ]
    exceedance_records = exceedances[exceedance_columns].to_dict(
        orient="records"
    )
    payload = {
        "study_name": config.study_name,
        "configuration": config.payload,
        "frozen_artifact_verification": verify_frozen_artifacts(
            config, root
        ),
        "new_state_freeze": json.loads(
            (data_dir / "initial_state_freeze.json").read_text(
                encoding="utf-8"
            )
        ),
        "frozen_voltage_guards": guard_payload,
        "teacher_selection_regression": teacher_regression,
        "three_known_guard_exceedance_regression": {
            "source_trajectory_ids": sorted(known_sources),
            "all_covered": known_history_passed,
            "remaining_exceedance_count": int(
                known_history.guard_exceeded.astype(bool).sum()
            ),
        },
        "failure_diagnosis": {
            "guard_exceedance_events": exceedance_records,
            "new_internal_validation_used_for_retuning": False,
            "new_internal_validation_downgraded_to_history": bool(
                len(exceedances)
            ),
            "teacher_selection_contract_repaired": bool(
                teacher_regression["all_passed"]
                and not len(strict_failures)
                and metrics.solver_failure_count.sum() == 0
            ),
            "next_historical_minimum_30c_v": (
                float(exceedances.positive_residual_growth_v.max())
                if len(exceedances)
                else None
            ),
            "required_next_phase": (
                "Phase 7C-R2F2：保留R2F新内部验证为永久失败回归，"
                "以25.001527 mV为新的历史下限证据，重新预注册全新"
                "开发/内部验证集；不得在R2F原集合上调参后宣称通过。"
                if len(exceedances)
                else None
            ),
        },
        "role_results": role_results,
        "global_summary": {
            "development_audit_trajectory_count": int(
                audit.trajectory_id.nunique()
            ),
            "final_trajectory_count": len(metrics),
            "target_reach_fraction": float(metrics.target_reached.mean()),
            "maximum_voltage_v": float(metrics.maximum_voltage_v.max()),
            "maximum_temperature_c": float(
                metrics.maximum_temperature_c.max()
            ),
            "maximum_current_step_a": float(
                metrics.maximum_current_step_a.max()
            ),
            "guard_exceedance_count": int(len(exceedances)),
            "strict_teacher_failure_count": int(len(strict_failures)),
            "solver_failure_count": int(
                metrics.solver_failure_count.sum()
            ),
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
            "unified_supervisor_total_time_s": float(
                metrics.total_thermal_supervisor_time_s.sum()
            ),
        },
        "success": bool(success),
        "decision": {
            "freeze_r2f": bool(success),
            "proceed_to_r3_design": bool(success),
            "r3_initial_states_generated": False,
            "run_ann": False,
            "internal_validation_used_for_retuning": False,
            "conclusion": (
                "R2F严格通过；可以冻结R2F并单独设计R3初态，"
                "当前仍未生成R3初态且禁止运行ANN。"
                if success
                else "R2F触发严格停止；内部验证降级为历史回归，"
                "不得原地调参、生成R3初态或运行ANN。"
            ),
        },
    }
    (result_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(result_dir / "PHASE7C-R2F_中文实验报告.md", payload)
    _plot(result_dir, final, guard_payload)
    return payload


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    guard = payload["frozen_voltage_guards"]
    summary = payload["global_summary"]
    next_minimum = payload["failure_diagnosis"][
        "next_historical_minimum_30c_v"
    ]
    next_minimum_text = (
        f"{1000 * next_minimum:.6f} mV"
        if next_minimum is not None
        else "无"
    )
    rows = []
    for role, temperatures in payload["role_results"].items():
        for token, result in temperatures.items():
            item = result["summary"]
            rows.append(
                f"| {role} | {token} ℃ | {item['trajectory_count']} | "
                f"{100 * item['target_reach_fraction']:.1f}% | "
                f"{item['maximum_voltage_v']:.6f} | "
                f"{item['maximum_temperature_c']:.6f} | "
                f"{item['guard_exceedance_count']} | "
                f"{item['solver_failure_count']} | "
                f"{item['empty_voltage_slew_count']}/"
                f"{item['empty_thermal_slew_count']} | "
                f"{'通过' if result['success'] else '失败'} |"
            )
    report = f"""# Phase 7C-R2F：教师选择与30 ℃电压裕量实验报告

## 冻结合同

R2证据保持原样。教师候选必须同时满足优化成功和预测可行；目标函数只在
合格候选之间比较。30 ℃在查看内部验证前预注册为单一裕量，不使用分层设计。
15 ℃不重新开发，只运行历史回归。本阶段未生成R3初态、未训练或运行ANN。

## 冻结裕量

- 15 ℃：{1000 * guard['guards_v']['15']:.6f} mV；
- 25 ℃：{1000 * guard['guards_v']['25']:.6f} mV；
- 30 ℃：{1000 * guard['guards_v']['30']:.6f} mV；
- 30 ℃新开发最大正向增长：
  {1000 * guard['new_development_maximum_positive_growth_v']:.6f} mV；
- 工程余量：{1000 * guard['engineering_margin_v']:.3f} mV。

## 一次性验证

| 数据角色 | 温度 | 轨迹 | 到达率 | 最高电压/V | 最高温度/℃ | 裕量超越 | 求解失败 | 电压/热空区间 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## 全局结果

- 开发审计：{summary['development_audit_trajectory_count']}条；
- 冻结参数后的验证/回归：{summary['final_trajectory_count']}条；
- 目标到达率：{100 * summary['target_reach_fraction']:.1f}%；
- 最高DFN电压：{summary['maximum_voltage_v']:.6f} V；
- 最高平均温度：{summary['maximum_temperature_c']:.6f} ℃；
- 最大单步电流变化：{summary['maximum_current_step_a']:.6f} A；
- 裕量超越：{summary['guard_exceedance_count']}；
- 严格教师失败/求解失败：
  {summary['strict_teacher_failure_count']}/{summary['solver_failure_count']}；
- 电压/热空区间：
  {summary['empty_voltage_slew_count']}/{summary['empty_thermal_slew_count']}；
- 持续振荡：{summary['sustained_oscillation_count']}；
- 统一监督层总计算时间：
  {summary['unified_supervisor_total_time_s']:.6f} s。

## 判定

{payload['decision']['conclusion']}

## 严格停止诊断

- 新内部验证裕量超越：
  {len(payload['failure_diagnosis']['guard_exceedance_events'])}次；
- 教师选择合同已修复：
  {payload['failure_diagnosis']['teacher_selection_contract_repaired']}；
- 新内部验证用于再次调参：
  {payload['failure_diagnosis']['new_internal_validation_used_for_retuning']}；
- 下一阶段历史下限：
  {next_minimum_text}；
- 后续要求：{payload['failure_diagnosis']['required_next_phase']}
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
    for role, group in valid.groupby("role"):
        if role not in {"development", "internal_validation"}:
            continue
        axes[0].scatter(
            group.soc,
            1000 * group.positive_residual_growth_v,
            s=6,
            alpha=0.35,
            label=role,
        )
        axes[1].scatter(
            group.soc, group.terminal_voltage_v, s=6, alpha=0.35
        )
        axes[2].scatter(
            group.soc, group.next_temperature_c, s=6, alpha=0.35
        )
    axes[0].axhline(
        1000 * guard_payload["guards_v"]["30"],
        color="red",
        linestyle="--",
        label="frozen g30",
    )
    axes[1].axhline(4.2, color="red", linestyle="--")
    axes[2].axhline(35.0, color="red", linestyle="--")
    axes[0].set(
        xlabel="SOC",
        ylabel="Positive residual growth [mV]",
        title="30 ℃ residual-growth coverage",
    )
    axes[1].set(
        xlabel="SOC", ylabel="DFN voltage [V]", title="Voltage safety"
    )
    axes[2].set(
        xlabel="SOC",
        ylabel="Average temperature [℃]",
        title="Thermal safety",
    )
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(result_dir / "phase7cr2f_summary.png", dpi=180)
    plt.close(figure)


def run_phase7cr2f(
    config: Phase7CR2FConfig,
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
            raise RuntimeError(f"R2F新初态哈希不匹配：{name}")
    audit_guards = {
        "15": float(config.voltage["guard_15c_v"]),
        "25": float(config.voltage["guard_25c_v"]),
        "30": float(config.voltage["development_audit_guard_30c_v"]),
    }
    audit = _run_rows(
        config,
        root,
        _new_rows(config, root, "development"),
        "development_guard_audit",
        audit_guards,
        resume,
    )
    guard_payload = _derive_and_freeze_guards(config, root, audit)
    guards = guard_payload["guards_v"]
    frames = [
        _run_rows(
            config,
            root,
            _new_rows(config, root, "development"),
            "development",
            guards,
            resume,
        )
    ]
    for role, rows in _legacy_roles(config, root):
        frames.append(
            _run_rows(config, root, rows, role, guards, resume)
        )
    frames.append(
        _run_rows(
            config,
            root,
            _new_rows(config, root, "internal_validation"),
            "internal_validation",
            guards,
            resume,
        )
    )
    final = pd.concat(frames, ignore_index=True)
    return _analyze(config, root, audit, final, guard_payload)
