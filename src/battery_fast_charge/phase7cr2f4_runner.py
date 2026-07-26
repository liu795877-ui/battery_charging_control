"""Phase 7C-R2F4 independent repair of the 25 C running guard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .phase7cr2f2_residual_audit import _artifact_hash
from .phase7cr2f2_runner import _prefix, _sha256, _state_columns
from .phase7cr2f3_config import load_phase7cr2f3_config
from .phase7cr2f3_runner import (
    _all_historical_values,
    _build_state_set,
    _checks,
    _historical_roles as _r2f3_historical_roles,
    _run_rows,
    _temperature_stage_counts,
)
from .phase7cr2f4_config import Phase7CR2F4Config
from .phase7cr2f_config import load_phase7cr2f_config
from .phase7cr2f_runner import verify_known_teacher_regressions
from .phase7cr2_runner import _trajectory_metrics


def verify_frozen_r2f3(
    config: Phase7CR2F4Config, root: Path
) -> dict[str, Any]:
    sources = config.section("sources")
    manifest_path = root / sources["phase7cr2f3_freeze_manifest"]
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_hash != sources["phase7cr2f3_freeze_manifest_sha256"]:
        raise RuntimeError("R2F3 freeze-manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "strict_stop_failed":
        raise RuntimeError("R2F3 strict-stop status changed")
    if manifest["r3_initial_states_generated"] or manifest["ann_execution_authorized"]:
        raise RuntimeError("R2F3 evidence unexpectedly authorizes later stages")
    mismatches: list[str] = []
    records: dict[str, Any] = {}
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
        raise RuntimeError(f"R2F3 frozen artifacts changed: {mismatches}")
    guard_path = root / sources["phase7cr2f3_frozen_guards"]
    guard_hash = hashlib.sha256(guard_path.read_bytes()).hexdigest()
    if guard_hash != sources["phase7cr2f3_frozen_guards_sha256"]:
        raise RuntimeError("R2F3 frozen-guard hash mismatch")
    frozen = json.loads(guard_path.read_text(encoding="utf-8"))["guards"]
    expected_frozen = {
        str(key): {
            "boot_v": float(value["boot_v"]),
            "running_v": float(value["running_v"]),
        }
        for key, value in config.section("voltage_guard")["frozen_guards"].items()
    }
    if frozen != expected_frozen:
        raise RuntimeError("Inherited R2F3 guards changed")
    contract = config.section("control_contract")
    if not contract["only_25c_running_guard_is_developed"]:
        raise RuntimeError("R2F4 scope must be limited to the 25 C running guard")
    if contract["r3_generation_authorized"] or contract["ann_execution_authorized"]:
        raise RuntimeError("R2F4 cannot authorize R3 or ANN")
    if contract["level4_entry_authorized"]:
        raise RuntimeError("R2F4 cannot authorize Level 4")
    return {
        "manifest_sha256": manifest_hash,
        "frozen_guard_sha256": guard_hash,
        "r2f3_failure_preserved": True,
        "records": records,
    }


def derive_25c_running_guard(
    voltage: dict[str, Any], development: pd.DataFrame
) -> dict[str, Any]:
    running = development[
        np.isclose(development.ambient_temperature_c, 25.0)
        & (development.step_index >= 2)
    ]
    if running.empty:
        raise RuntimeError("R2F4 development does not cover the running stage")
    observed = float(running.positive_residual_growth_v.max())
    history = float(voltage["historical_25c_running_minimum_v"])
    margin = float(voltage["engineering_margin_v"])
    guards = {
        str(temperature): {
            "boot_v": float(values["boot_v"]),
            "running_v": float(values["running_v"]),
        }
        for temperature, values in voltage["frozen_guards"].items()
    }
    guards["25"]["running_v"] = max(history, observed) + margin
    return {
        "guards": guards,
        "development_25c_running_maximum_v": observed,
        "historical_25c_running_minimum_v": history,
        "engineering_margin_v": margin,
    }


def prepare_and_freeze_states(
    config: Phase7CR2F4Config, root: Path
) -> dict[str, Any]:
    verification = verify_frozen_r2f3(config, root)
    data_dir = root / config.section("output")["data_directory"]
    data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = data_dir / "initial_state_freeze.json"
    if freeze_path.exists():
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
        for name, record in payload["files"].items():
            if _sha256(data_dir / name) != record["sha256"]:
                raise RuntimeError(f"R2F4 state hash mismatch: {name}")
        return payload
    existing = _all_historical_values(config, root)
    extra = [
        pd.read_csv(root / relative)[_state_columns()].to_numpy(float)
        for relative in config.section("sources")["phase7cr2f3_internal_states"].values()
    ]
    existing = np.vstack([existing, *extra])
    frames: dict[str, pd.DataFrame] = {}
    for role in ("development", "internal_validation"):
        frame, existing = _build_state_set(config, root, role, 25, existing)
        frame["trajectory_id"] = frame.trajectory_id.str.replace(
            "phase7cr2f3_", "phase7cr2f4_", regex=False
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
            "extreme_initial_residual_count": int(
                frame.initial_residual_extreme.astype(bool).sum()
            ),
        }
    payload = {
        "phase": "Phase 7C-R2F4",
        "status": "initial_states_frozen_before_any_closed_loop_rollout",
        "frozen_before_any_closed_loop_rollout": True,
        "development_internal_isolation": True,
        "not_r3_confirmation_data": True,
        "not_ann_teacher_data": True,
        "config_sha256": _sha256(
            root / "configs/phase7cr2f4_25c_running_guard.yaml", True
        ),
        "files": files,
        "frozen_r2f3_verification": verification,
    }
    freeze_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _new_rows(
    config: Phase7CR2F4Config, root: Path, role: str
) -> list[dict[str, Any]]:
    path = (
        root
        / config.section("output")["data_directory"]
        / f"{role}_initial_states_25c.csv"
    )
    return pd.read_csv(path).to_dict(orient="records")


def _verify_state_freeze(
    config: Phase7CR2F4Config, root: Path
) -> dict[str, Any]:
    data_dir = root / config.section("output")["data_directory"]
    path = data_dir / "initial_state_freeze.json"
    if not path.exists():
        raise RuntimeError("R2F4 states must be frozen before development")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, record in payload["files"].items():
        if _sha256(data_dir / name) != record["sha256"]:
            raise RuntimeError(f"R2F4 state hash mismatch: {name}")
    return payload


def run_development(
    config: Phase7CR2F4Config, root: Path, resume: bool
) -> dict[str, Any]:
    verify_frozen_r2f3(config, root)
    state_freeze = _verify_state_freeze(config, root)
    data_dir = root / config.section("output")["data_directory"]
    guard_path = data_dir / "frozen_25c_running_guard.json"
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
    boot_exceedances = int(
        ((audit.guard_stage == "boot") & audit.guard_exceeded.astype(bool)).sum()
    )
    if boot_exceedances:
        raise RuntimeError(
            "R2F4 found a boot-stage exceedance but boot guard is frozen"
        )
    audit_path = data_dir / "development_guard_audit.csv"
    audit.to_csv(audit_path, index=False)
    derived = derive_25c_running_guard(config.section("voltage_guard"), audit)
    payload = {
        "phase": "Phase 7C-R2F4 development",
        "status": "25c_running_guard_frozen_before_internal_validation",
        "only_25c_running_guard_developed": True,
        "internal_validation_used_for_tuning": False,
        **derived,
        "state_hashes": {
            name: record["sha256"]
            for name, record in state_freeze["files"].items()
        },
        "development_audit_sha256": _sha256(audit_path),
        "config_sha256": _sha256(
            root / "configs/phase7cr2f4_25c_running_guard.yaml", True
        ),
        "runner_sha256": _sha256(Path(__file__), True),
    }
    guard_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _verify_guard_freeze(
    config: Phase7CR2F4Config, root: Path
) -> dict[str, Any]:
    data_dir = root / config.section("output")["data_directory"]
    path = data_dir / "frozen_25c_running_guard.json"
    if not path.exists():
        raise RuntimeError("R2F4 running guard must be frozen first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["internal_validation_used_for_tuning"]:
        raise RuntimeError("R2F4 internal validation cannot tune guards")
    if payload["config_sha256"] != _sha256(
        root / "configs/phase7cr2f4_25c_running_guard.yaml", True
    ):
        raise RuntimeError("R2F4 config changed after guard freeze")
    if payload["runner_sha256"] != _sha256(Path(__file__), True):
        raise RuntimeError("R2F4 runner changed after guard freeze")
    if payload["development_audit_sha256"] != _sha256(
        data_dir / "development_guard_audit.csv"
    ):
        raise RuntimeError("R2F4 development audit changed")
    inherited = config.section("voltage_guard")["frozen_guards"]
    guards = payload["guards"]
    for temperature, stage in ((15, "boot_v"), (15, "running_v"), (25, "boot_v"), (30, "boot_v"), (30, "running_v")):
        if guards[str(temperature)][stage] != float(inherited[temperature][stage]):
            raise RuntimeError("R2F4 changed a frozen guard")
    return payload


def _historical_roles(
    config: Phase7CR2F4Config, root: Path
) -> list[tuple[str, list[dict[str, Any]]]]:
    sources = config.section("sources")
    r2f3 = load_phase7cr2f3_config(root / sources["phase7cr2f3_config"])
    roles = _r2f3_historical_roles(r2f3, root)
    for temperature, relative in sources["phase7cr2f3_internal_states"].items():
        role = "legacy_phase7cr2f3_internal"
        rows = _prefix(pd.read_csv(root / relative), role, float(temperature))
        existing = next((item for item in roles if item[0] == role), None)
        if existing is None:
            roles.append((role, rows))
        else:
            existing[1].extend(rows)
    count = sum(len(rows) for _, rows in roles)
    expected = int(
        config.section("validation_contract")[
            "expected_historical_regression_trajectory_count"
        ]
    )
    if count != expected:
        raise RuntimeError(f"R2F4 historical count mismatch: {count}")
    return roles


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    guards = payload["frozen_guard_contract"]["guards"]
    counts = payload["guard_exceedance_counts"]
    summary = payload["global_summary"]
    rows = []
    for temperature in (15, 25, 30):
        token = str(temperature)
        rows.append(
            f"| {temperature} | {1000 * guards[token]['boot_v']:.6f} | "
            f"{1000 * guards[token]['running_v']:.6f} | "
            f"{counts[token]['boot_exceedance_count']} | "
            f"{counts[token]['running_exceedance_count']} | "
            f"{counts[token]['total_exceedance_count']} |"
        )
    conclusion = (
        "R2F4严格通过；可以冻结完整多温度安全MPC架构并进入独立R3设计。当前仍未生成R3初态或运行ANN。"
        if payload["success"]
        else "R2F4严格停止；内部验证永久降级为回归证据，不得原地调参、生成R3、运行ANN或进入Level 4。"
    )
    report = f"""# Phase 7C-R2F4 25 ℃运行裕量修订报告

## 冻结边界

本阶段只开发25 ℃运行裕量。15/30 ℃两段裕量和25 ℃启动裕量均保持R2F3冻结值；残差仍由时刻0测量初始化。

| 温度/℃ | 启动裕量/mV | 运行裕量/mV | 启动超越 | 运行超越 | 总超越 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}
| 全温度 | — | — | {counts['all_temperatures']['boot_exceedance_count']} | {counts['all_temperatures']['running_exceedance_count']} | {counts['all_temperatures']['total_exceedance_count']} |

## 一次性验证

- 历史回归：{summary['historical_regression_trajectory_count']}条；
- 新内部验证：{summary['new_internal_trajectory_count']}条；
- 总轨迹：{summary['trajectory_count']}条；
- 目标到达率：{100 * summary['target_reach_fraction']:.1f}%；
- 最高DFN电压：{summary['maximum_voltage_v']:.6f} V；
- 最高平均温度：{summary['maximum_temperature_c']:.6f} ℃；
- 教师失败/求解失败/预测不可行：{summary['strict_teacher_failure_count']}/{summary['solver_failure_count']}/{summary['prediction_infeasible_count']}；
- 电压—斜率/热—斜率空区间：{summary['empty_voltage_slew_count']}/{summary['empty_thermal_slew_count']}；
- 持续振荡：{summary['sustained_oscillation_count']}；
- 零残差初始化轨迹：{summary['zero_residual_initialization_count']}。

## 判定

{conclusion}
"""
    path.write_text(report, encoding="utf-8")


def run_validation(
    config: Phase7CR2F4Config, root: Path, resume: bool
) -> dict[str, Any]:
    source_verification = verify_frozen_r2f3(config, root)
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
        raise RuntimeError("R2F4 one-shot validation already started")
    if not started_path.exists():
        started_path.write_text(
            json.dumps(
                {
                    "status": "one_shot_validation_started",
                    "guards_sha256": _sha256(
                        data_dir / "frozen_25c_running_guard.json"
                    ),
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
    frames: list[pd.DataFrame] = []
    for role, rows in _historical_roles(config, root):
        frames.append(
            _run_rows(config, root, rows, role, guards, "validation", resume)
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
    if historical_count != int(contract["expected_historical_regression_trajectory_count"]):
        raise RuntimeError("R2F4 historical trajectory count mismatch")
    if internal_count != int(contract["expected_new_internal_trajectory_count"]):
        raise RuntimeError("R2F4 internal trajectory count mismatch")
    if total_count != int(contract["expected_total_validation_trajectory_count"]):
        raise RuntimeError("R2F4 total trajectory count mismatch")
    counts = _temperature_stage_counts(final)
    checks = _checks(config, metrics, final)
    teacher_regression = verify_known_teacher_regressions(r2f, root)
    success = bool(all(checks.values()) and teacher_regression["all_passed"])
    payload = {
        "study_name": config.study_name,
        "configuration": config.payload,
        "frozen_r2f3_verification": source_verification,
        "initial_state_freeze": state_freeze,
        "frozen_guard_contract": guard_freeze,
        "teacher_selection_regression": teacher_regression,
        "guard_exceedance_counts": counts,
        "checks": checks,
        "global_summary": {
            "trajectory_count": total_count,
            "historical_regression_trajectory_count": historical_count,
            "new_internal_trajectory_count": internal_count,
            "target_reach_fraction": float(metrics.target_reached.mean()),
            "maximum_voltage_v": float(metrics.maximum_voltage_v.max()),
            "maximum_temperature_c": float(metrics.maximum_temperature_c.max()),
            "maximum_current_step_a": float(metrics.maximum_current_step_a.max()),
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
            "zero_residual_initialization_count": int(
                final.groupby("trajectory_id").residual_initialization_mode.first()
                .ne("measured")
                .sum()
            ),
        },
        "success": success,
        "decision": {
            "freeze_full_multitemperature_architecture": success,
            "eligible_to_design_r3_separately": success,
            "r3_initial_states_generated": False,
            "ann_run_or_training_performed": False,
            "level4_entered": False,
            "internal_validation_used_for_retuning": False,
            "internal_validation_becomes_permanent_regression": bool(not success),
        },
    }
    metrics_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = result_dir / "PHASE7C-R2F4_中文实验报告.md"
    _write_report(report_path, payload)
    artifacts = [
        "configs/phase7cr2f4_25c_running_guard.yaml",
        "src/battery_fast_charge/phase7cr2f4_config.py",
        "src/battery_fast_charge/phase7cr2f4_runner.py",
        "src/battery_fast_charge/phase7cr2f4_cli.py",
        "data/phase7cr2f4_25c_running_guard/initial_state_freeze.json",
        "data/phase7cr2f4_25c_running_guard/development_initial_states_25c.csv",
        "data/phase7cr2f4_25c_running_guard/internal_validation_initial_states_25c.csv",
        "data/phase7cr2f4_25c_running_guard/frozen_25c_running_guard.json",
        "data/phase7cr2f4_25c_running_guard/development_guard_audit.csv",
        "data/phase7cr2f4_25c_running_guard/combined_validation_trajectories.csv",
        "data/phase7cr2f4_25c_running_guard/trajectory_metrics.csv",
        "outputs/phase7cr2f4_25c_running_guard/metrics.json",
        "outputs/phase7cr2f4_25c_running_guard/PHASE7C-R2F4_中文实验报告.md",
    ]
    manifest = {
        "phase": "Phase 7C-R2F4",
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


def run_phase7cr2f4(
    config: Phase7CR2F4Config,
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
