"""Phase 7C-R2F5 repair of the 25 C boot and running voltage guards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .phase7cr2f2_residual_audit import _artifact_hash
from .phase7cr2f2_runner import _prefix, _sha256, _state_columns
from .phase7cr2f3_runner import (
    _all_historical_values,
    _build_state_set,
    _checks,
    _run_rows,
    _temperature_stage_counts,
)
from .phase7cr2f4_config import load_phase7cr2f4_config
from .phase7cr2f4_runner import _historical_roles as _r2f4_historical_roles
from .phase7cr2f5_config import Phase7CR2F5Config
from .phase7cr2f_config import load_phase7cr2f_config
from .phase7cr2f_runner import verify_known_teacher_regressions
from .phase7cr2_runner import _trajectory_metrics


CONFIG_RELATIVE = "configs/phase7cr2f5_25c_two_stage_guards.yaml"
GUARD_FILENAME = "frozen_25c_two_stage_guards.json"


def verify_frozen_r2f4(
    config: Phase7CR2F5Config, root: Path
) -> dict[str, Any]:
    sources = config.section("sources")
    path = root / sources["phase7cr2f4_freeze_manifest"]
    actual_manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_manifest_hash != sources["phase7cr2f4_freeze_manifest_sha256"]:
        raise RuntimeError("R2F4 freeze-manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["status"] != "strict_stop_failed_during_development":
        raise RuntimeError("R2F4 strict-stop status changed")
    if manifest["internal_validation_started"] or manifest["level4_entered"]:
        raise RuntimeError("R2F4 evidence boundary changed")
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
        raise RuntimeError(f"R2F4 frozen artifacts changed: {mismatches}")
    contract = config.section("control_contract")
    if not contract["only_25c_boot_and_running_guards_are_developed"]:
        raise RuntimeError("R2F5 scope must be limited to both 25 C guards")
    if not contract["guards_15c_and_30c_frozen"]:
        raise RuntimeError("R2F5 must freeze the 15 C and 30 C guards")
    if (
        contract["r3_generation_authorized"]
        or contract["ann_execution_authorized"]
        or contract["level4_entry_authorized"]
    ):
        raise RuntimeError("R2F5 cannot authorize R3, ANN, or Level 4")
    return {
        "manifest_sha256": actual_manifest_hash,
        "r2f4_failure_preserved": True,
        "records": records,
    }


def derive_25c_two_stage_guards(
    voltage: dict[str, Any], development: pd.DataFrame
) -> dict[str, Any]:
    frame = development[np.isclose(development.ambient_temperature_c, 25.0)]
    boot = frame[frame.step_index.isin((0, 1))]
    running = frame[frame.step_index >= 2]
    if boot.empty or running.empty:
        raise RuntimeError("R2F5 development must cover both guard stages")
    boot_observed = float(boot.positive_residual_growth_v.max())
    running_observed = float(running.positive_residual_growth_v.max())
    boot_history = float(voltage["historical_25c_boot_minimum_v"])
    running_history = float(voltage["historical_25c_running_minimum_v"])
    margin = float(voltage["engineering_margin_v"])
    guards = {
        str(temperature): {
            "boot_v": float(values["boot_v"]),
            "running_v": float(values["running_v"]),
        }
        for temperature, values in voltage["frozen_guards"].items()
    }
    guards["25"] = {
        "boot_v": max(boot_history, boot_observed) + margin,
        "running_v": max(running_history, running_observed) + margin,
    }
    return {
        "guards": guards,
        "development_25c_boot_maximum_v": boot_observed,
        "development_25c_running_maximum_v": running_observed,
        "historical_25c_boot_minimum_v": boot_history,
        "historical_25c_running_minimum_v": running_history,
        "engineering_margin_v": margin,
    }


def prepare_and_freeze_states(
    config: Phase7CR2F5Config, root: Path
) -> dict[str, Any]:
    verification = verify_frozen_r2f4(config, root)
    data_dir = root / config.section("output")["data_directory"]
    data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = data_dir / "initial_state_freeze.json"
    if freeze_path.exists():
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
        for name, record in payload["files"].items():
            if _sha256(data_dir / name) != record["sha256"]:
                raise RuntimeError(f"R2F5 state hash mismatch: {name}")
        return payload
    existing = _all_historical_values(config, root)
    extra_sources = (
        config.section("sources")["phase7cr2f4_development_states"],
        config.section("sources")["phase7cr2f4_internal_states"],
    )
    extra = [
        pd.read_csv(root / relative)[_state_columns()].to_numpy(float)
        for relative in extra_sources
    ]
    existing = np.vstack([existing, *extra])
    frames: dict[str, pd.DataFrame] = {}
    for role in ("development", "internal_validation"):
        frame, existing = _build_state_set(config, root, role, 25, existing)
        frame["trajectory_id"] = frame.trajectory_id.str.replace(
            "phase7cr2f3_", "phase7cr2f5_", regex=False
        )
        frames[role] = frame
    files: dict[str, Any] = {}
    for role, frame in frames.items():
        name = f"{role}_initial_states_25c.csv"
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
            "zero_initial_residual_count": int(
                np.isclose(frame.initial_measured_residual_v, 0.0).sum()
            ),
        }
    payload = {
        "phase": "Phase 7C-R2F5",
        "status": "initial_states_frozen_before_any_closed_loop_rollout",
        "frozen_before_any_closed_loop_rollout": True,
        "development_internal_isolation": True,
        "not_r3_confirmation_data": True,
        "not_ann_teacher_data": True,
        "config_sha256": _sha256(root / CONFIG_RELATIVE, True),
        "files": files,
        "frozen_r2f4_verification": verification,
    }
    freeze_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _verify_state_freeze(
    config: Phase7CR2F5Config, root: Path
) -> dict[str, Any]:
    data_dir = root / config.section("output")["data_directory"]
    path = data_dir / "initial_state_freeze.json"
    if not path.exists():
        raise RuntimeError("R2F5 states must be frozen before development")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["config_sha256"] != _sha256(root / CONFIG_RELATIVE, True):
        raise RuntimeError("R2F5 config changed after state freeze")
    for name, record in payload["files"].items():
        if _sha256(data_dir / name) != record["sha256"]:
            raise RuntimeError(f"R2F5 state hash mismatch: {name}")
    return payload


def _new_rows(
    config: Phase7CR2F5Config, root: Path, role: str
) -> list[dict[str, Any]]:
    path = (
        root
        / config.section("output")["data_directory"]
        / f"{role}_initial_states_25c.csv"
    )
    return pd.read_csv(path).to_dict(orient="records")


def run_development(
    config: Phase7CR2F5Config, root: Path, resume: bool
) -> dict[str, Any]:
    verify_frozen_r2f4(config, root)
    state_freeze = _verify_state_freeze(config, root)
    data_dir = root / config.section("output")["data_directory"]
    guard_path = data_dir / GUARD_FILENAME
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
    derived = derive_25c_two_stage_guards(config.section("voltage_guard"), audit)
    payload = {
        "phase": "Phase 7C-R2F5 development",
        "status": "25c_two_stage_guards_frozen_before_internal_validation",
        "only_25c_boot_and_running_guards_developed": True,
        "internal_validation_used_for_tuning": False,
        **derived,
        "state_hashes": {
            name: record["sha256"] for name, record in state_freeze["files"].items()
        },
        "development_audit_sha256": _sha256(audit_path),
        "config_sha256": _sha256(root / CONFIG_RELATIVE, True),
        "runner_sha256": _sha256(Path(__file__), True),
    }
    guard_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _verify_guard_freeze(
    config: Phase7CR2F5Config, root: Path
) -> dict[str, Any]:
    data_dir = root / config.section("output")["data_directory"]
    path = data_dir / GUARD_FILENAME
    if not path.exists():
        raise RuntimeError("R2F5 two-stage guards must be frozen first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["internal_validation_used_for_tuning"]:
        raise RuntimeError("R2F5 internal validation cannot tune guards")
    if payload["config_sha256"] != _sha256(root / CONFIG_RELATIVE, True):
        raise RuntimeError("R2F5 config changed after guard freeze")
    if payload["runner_sha256"] != _sha256(Path(__file__), True):
        raise RuntimeError("R2F5 runner changed after guard freeze")
    if payload["development_audit_sha256"] != _sha256(
        data_dir / "development_guard_audit.csv"
    ):
        raise RuntimeError("R2F5 development audit changed")
    inherited = config.section("voltage_guard")["frozen_guards"]
    for temperature in (15, 30):
        expected = inherited[temperature]
        actual = payload["guards"][str(temperature)]
        if actual != {
            "boot_v": float(expected["boot_v"]),
            "running_v": float(expected["running_v"]),
        }:
            raise RuntimeError("R2F5 changed a frozen temperature guard")
    return payload


def _historical_roles(
    config: Phase7CR2F5Config, root: Path
) -> list[tuple[str, list[dict[str, Any]]]]:
    sources = config.section("sources")
    r2f4 = load_phase7cr2f4_config(root / sources["phase7cr2f4_config"])
    roles = _r2f4_historical_roles(r2f4, root)
    role = "legacy_phase7cr2f4_development"
    rows = _prefix(
        pd.read_csv(root / sources["phase7cr2f4_development_states"]), role, 25.0
    )
    roles.append((role, rows))
    count = sum(len(records) for _, records in roles)
    expected = int(
        config.section("validation_contract")[
            "expected_historical_regression_trajectory_count"
        ]
    )
    if count != expected:
        raise RuntimeError(f"R2F5 historical count mismatch: {count}")
    return roles


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    guards = payload["frozen_guard_contract"]["guards"]
    counts = payload["guard_exceedance_counts"]
    summary = payload["global_summary"]
    result = "严格通过，可单独设计R3。" if payload["success"] else (
        "严格停止；内部验证永久转为回归证据，不得原地调参或进入R3。"
    )
    path.write_text(
        "# Phase 7C-R2F5 25 ℃两段电压裕量联合修复\n\n"
        f"- 25 ℃启动裕量：{1000 * guards['25']['boot_v']:.6f} mV；\n"
        f"- 25 ℃运行裕量：{1000 * guards['25']['running_v']:.6f} mV；\n"
        f"- 全温度启动/运行超越：{counts['all_temperatures']['boot_exceedance_count']}/"
        f"{counts['all_temperatures']['running_exceedance_count']}；\n"
        f"- 验证轨迹：{summary['trajectory_count']}条；\n"
        f"- 目标到达率：{100 * summary['target_reach_fraction']:.1f}%；\n"
        f"- 最高电压：{summary['maximum_voltage_v']:.6f} V；\n"
        f"- 最高平均温度：{summary['maximum_temperature_c']:.6f} ℃。\n\n"
        f"结论：{result}\n",
        encoding="utf-8",
    )


def run_validation(
    config: Phase7CR2F5Config, root: Path, resume: bool
) -> dict[str, Any]:
    source_verification = verify_frozen_r2f4(config, root)
    state_freeze = _verify_state_freeze(config, root)
    guard_freeze = _verify_guard_freeze(config, root)
    data_dir = root / config.section("output")["data_directory"]
    result_dir = root / config.section("output")["result_directory"]
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = result_dir / "metrics.json"
    if metrics_json.exists():
        return json.loads(metrics_json.read_text(encoding="utf-8"))
    started_path = data_dir / "validation_started.json"
    if started_path.exists() and not resume:
        raise RuntimeError("R2F5 one-shot validation already started")
    if not started_path.exists():
        started_path.write_text(
            json.dumps(
                {
                    "status": "one_shot_validation_started",
                    "guards_sha256": _sha256(data_dir / GUARD_FILENAME),
                    "internal_state_sha256": state_freeze["files"][
                        "internal_validation_initial_states_25c.csv"
                    ]["sha256"],
                    "internal_validation_may_not_modify_guards": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    guards = guard_freeze["guards"]
    frames = [
        _run_rows(config, root, rows, role, guards, "validation", resume)
        for role, rows in _historical_roles(config, root)
    ]
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
    trajectory_metrics_path = data_dir / "trajectory_metrics.csv"
    metrics.to_csv(trajectory_metrics_path, index=False)
    historical_count = int(
        metrics[metrics.role != "internal_validation"].trajectory_id.nunique()
    )
    internal_count = int(
        metrics[metrics.role == "internal_validation"].trajectory_id.nunique()
    )
    total_count = int(metrics.trajectory_id.nunique())
    contract = config.section("validation_contract")
    expected = (
        int(contract["expected_historical_regression_trajectory_count"]),
        int(contract["expected_new_internal_trajectory_count"]),
        int(contract["expected_total_validation_trajectory_count"]),
    )
    if (historical_count, internal_count, total_count) != expected:
        raise RuntimeError("R2F5 validation trajectory count mismatch")
    counts = _temperature_stage_counts(final)
    checks = _checks(config, metrics, final)
    teacher_regression = verify_known_teacher_regressions(r2f, root)
    success = bool(all(checks.values()) and teacher_regression["all_passed"])
    summary = {
        "trajectory_count": total_count,
        "historical_regression_trajectory_count": historical_count,
        "new_internal_trajectory_count": internal_count,
        "target_reach_fraction": float(metrics.target_reached.mean()),
        "maximum_voltage_v": float(metrics.maximum_voltage_v.max()),
        "maximum_temperature_c": float(metrics.maximum_temperature_c.max()),
        "maximum_current_step_a": float(metrics.maximum_current_step_a.max()),
        "strict_teacher_failure_count": int(final.strict_teacher_failure.sum()),
        "solver_failure_count": int(metrics.solver_failure_count.sum()),
        "prediction_infeasible_count": int(metrics.prediction_infeasible_count.sum()),
        "empty_voltage_slew_count": int(metrics.empty_voltage_slew_count.sum()),
        "empty_thermal_slew_count": int(metrics.empty_thermal_slew_count.sum()),
        "sustained_oscillation_count": int(metrics.sustained_oscillation_count.sum()),
        "zero_residual_initialization_count": int(
            final.groupby("trajectory_id").residual_initialization_mode.first()
            .ne("measured")
            .sum()
        ),
    }
    payload = {
        "study_name": config.study_name,
        "configuration": config.payload,
        "frozen_r2f4_verification": source_verification,
        "initial_state_freeze": state_freeze,
        "frozen_guard_contract": guard_freeze,
        "teacher_selection_regression": teacher_regression,
        "guard_exceedance_counts": counts,
        "checks": checks,
        "global_summary": summary,
        "success": success,
        "decision": {
            "freeze_full_multitemperature_architecture": success,
            "eligible_to_design_r3_separately": success,
            "r3_initial_states_generated": False,
            "ann_run_or_training_performed": False,
            "level4_entered": False,
            "internal_validation_used_for_retuning": False,
            "internal_validation_becomes_permanent_regression": not success,
        },
    }
    metrics_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = result_dir / "PHASE7C-R2F5_中文实验报告.md"
    _write_report(report_path, payload)
    artifacts = [
        CONFIG_RELATIVE,
        "src/battery_fast_charge/phase7cr2f5_config.py",
        "src/battery_fast_charge/phase7cr2f5_runner.py",
        "src/battery_fast_charge/phase7cr2f5_cli.py",
        "data/phase7cr2f5_25c_two_stage_guards/initial_state_freeze.json",
        "data/phase7cr2f5_25c_two_stage_guards/development_initial_states_25c.csv",
        "data/phase7cr2f5_25c_two_stage_guards/internal_validation_initial_states_25c.csv",
        f"data/phase7cr2f5_25c_two_stage_guards/{GUARD_FILENAME}",
        "data/phase7cr2f5_25c_two_stage_guards/development_guard_audit.csv",
        "data/phase7cr2f5_25c_two_stage_guards/combined_validation_trajectories.csv",
        "data/phase7cr2f5_25c_two_stage_guards/trajectory_metrics.csv",
        "outputs/phase7cr2f5_25c_two_stage_guards/metrics.json",
        "outputs/phase7cr2f5_25c_two_stage_guards/PHASE7C-R2F5_中文实验报告.md",
    ]
    manifest = {
        "phase": "Phase 7C-R2F5",
        "status": "strict_passed" if success else "strict_stop_failed",
        "internal_validation_used_for_retuning": False,
        "r3_initial_states_generated": False,
        "ann_execution_authorized": False,
        "level4_entered": False,
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


def run_phase7cr2f5(
    config: Phase7CR2F5Config,
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
