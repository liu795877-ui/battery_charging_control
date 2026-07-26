"""Phase 7C-R3T2 paired independent-DFN repeatability validation."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import phase7cr2f3_runner as safe_runner
from .phase7cr2_runner import _trajectory_metrics
from .phase7cr2f2_runner import _sha256 as _shared_sha256, _state_columns
from .phase7cr2f3_runner import _all_historical_values, _build_state_set
from .phase7cr2f_config import load_phase7cr2f_config
from .phase7cr3_config import load_phase7cr3_config
from .phase7cr3_runner import _ann_rollout, _model_hashes
from .phase7cr3t2_config import Phase7CR3T2Config
from .phase7cr3t_thermal import optimized_thermal_current_limit


CONFIG_RELATIVE = "configs/phase7cr3t2_dfn_repeatability.yaml"


def _sha256(path: Path, normalize_text: bool = False) -> str:
    payload = path.read_bytes()
    if normalize_text:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def verify_frozen_r3t(
    config: Phase7CR3T2Config, root: Path
) -> dict[str, Any]:
    sources = config.section("sources")
    manifest_path = root / sources["phase7cr3t_manifest"]
    audit_path = root / sources["phase7cr3t_strict_audit"]
    manifest_hash = _sha256(manifest_path)
    audit_hash = _sha256(audit_path)
    if manifest_hash != sources["phase7cr3t_manifest_sha256"]:
        raise RuntimeError("R3T manifest hash mismatch")
    if audit_hash != sources["phase7cr3t_strict_audit_sha256"]:
        raise RuntimeError("R3T strict-audit hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if manifest["status"] != "strict_stop_failed":
        raise RuntimeError("R3T strict-stop evidence changed")
    if audit["failed_checks"] != ["closed_loop_current_exact"]:
        raise RuntimeError("R3T must have only the repeatability failure")
    if audit["same_state_current_limit_difference_a"] != 0.0:
        raise RuntimeError("Same-state supervisor equivalence changed")
    mismatches: list[str] = []
    records: dict[str, Any] = {}
    for relative, expected in manifest["artifacts"].items():
        path = root / relative
        actual = _sha256(
            path, path.suffix in {".py", ".yaml", ".md"}
        )
        records[relative] = {
            "expected": expected,
            "actual": actual,
            "matched": expected == actual,
        }
        if expected != actual:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"Frozen R3T artifacts changed: {mismatches}")
    contract = config.section("control_contract")
    if (
        contract["ann_retraining_authorized"]
        or contract["safety_contract_change_authorized"]
        or contract["level4_entry_authorized"]
    ):
        raise RuntimeError("R3T2 scope or Level 4 boundary changed")
    return {
        "manifest_sha256": manifest_hash,
        "strict_audit_sha256": audit_hash,
        "r3t_failed_checks": audit["failed_checks"],
        "same_state_current_limit_difference_a": audit[
            "same_state_current_limit_difference_a"
        ],
        "records": records,
    }


def prepare_states(config: Phase7CR3T2Config, root: Path) -> dict[str, Any]:
    verification = verify_frozen_r3t(config, root)
    data_dir = root / config.section("output")["data_directory"]
    data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = data_dir / "initial_state_freeze.json"
    if freeze_path.exists():
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
        for name, record in payload["files"].items():
            if _shared_sha256(data_dir / name) != record["sha256"]:
                raise RuntimeError(f"R3T2 state hash mismatch: {name}")
        return payload
    existing = _all_historical_values(config, root)
    extras = [
        pd.read_csv(root / relative)[_state_columns()].to_numpy(float)
        for relative in config.section("sources")["phase7cr3_states"].values()
    ]
    existing = np.vstack([existing, *extras])
    files: dict[str, Any] = {}
    all_frames: dict[str, pd.DataFrame] = {}
    for role in ("development", "confirmation"):
        frames = []
        for temperature in (15, 30):
            frame, existing = _build_state_set(
                config, root, role, temperature, existing
            )
            frame["trajectory_id"] = frame.trajectory_id.str.replace(
                "phase7cr2f3_", "phase7cr3t2_", regex=False
            )
            frames.append(frame)
            name = f"{role}_initial_states_{temperature}c.csv"
            path = data_dir / name
            frame.to_csv(path, index=False)
            files[name] = {
                "sha256": _shared_sha256(path),
                "trajectory_count": len(frame),
                "temperature_c": temperature,
                "design_seed": int(frame.design_seed.iloc[0]),
                "zero_residual_count": int(
                    np.isclose(frame.initial_measured_residual_v, 0.0).sum()
                ),
            }
        all_frames[role] = pd.concat(frames, ignore_index=True)
    state_sets = {
        role: frame[_state_columns()].to_numpy(float)
        for role, frame in all_frames.items()
    }
    if np.any(
        np.all(
            np.isclose(
                state_sets["development"][:, None, :],
                state_sets["confirmation"][None, :, :],
                atol=1.0e-14,
            ),
            axis=2,
        )
    ):
        raise RuntimeError("R3T2 development and confirmation states overlap")
    payload = {
        "phase": "Phase 7C-R3T2",
        "status": "states_frozen_before_any_rollout",
        "development_confirmation_isolated": True,
        "not_teacher_data": True,
        "config_sha256": _sha256(root / CONFIG_RELATIVE, True),
        "files": files,
        "r3t_verification": verification,
        "development_started": False,
        "confirmation_started": False,
        "level4_entered": False,
    }
    freeze_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _state_rows(
    config: Phase7CR3T2Config, root: Path, role: str
) -> list[dict[str, Any]]:
    data_dir = root / config.section("output")["data_directory"]
    frames = [
        pd.read_csv(data_dir / f"{role}_initial_states_{temperature}c.csv")
        for temperature in (15, 30)
    ]
    return pd.concat(frames, ignore_index=True).to_dict(orient="records")


def _repeat_worker(
    r3_config: Any,
    root_text: str,
    initial: dict[str, Any],
    seed: int,
    guards: dict[str, Any],
    repetition: str,
    role: str,
    path_text: str,
) -> str:
    original = safe_runner._thermal_current_limit
    safe_runner._thermal_current_limit = optimized_thermal_current_limit
    try:
        frame = _ann_rollout(
            r3_config, Path(root_text), initial, seed, guards
        )
    finally:
        safe_runner._thermal_current_limit = original
    frame["repetition"] = repetition
    frame["role"] = f"{role}_{repetition}_seed_{seed}"
    frame.to_csv(path_text, index=False)
    return path_text


def _run_repetition(
    config: Phase7CR3T2Config,
    root: Path,
    role: str,
    repetition: str,
    seeds: tuple[int, ...],
    resume: bool,
) -> pd.DataFrame:
    r3_config = load_phase7cr3_config(
        root / config.section("sources")["phase7cr3_config"]
    )
    guards = json.loads(
        (
            root / r3_config.section("sources")["phase7cr2f5_guards"]
        ).read_text(encoding="utf-8")
    )["guards"]
    data_dir = root / config.section("output")["data_directory"]
    run_dir = data_dir / "runs" / role / repetition
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            initial,
            seed,
            run_dir / f"ann_{seed}_{initial['trajectory_id']}.csv",
        )
        for seed in seeds
        for initial in _state_rows(config, root, role)
    ]
    pending = [
        (initial, seed, path)
        for initial, seed, path in jobs
        if not (resume and path.exists())
    ]
    if pending:
        with ProcessPoolExecutor(
            max_workers=int(config.section("datasets")["maximum_workers"])
        ) as executor:
            futures = [
                executor.submit(
                    _repeat_worker,
                    r3_config,
                    str(root),
                    initial,
                    seed,
                    guards,
                    repetition,
                    role,
                    str(path),
                )
                for initial, seed, path in pending
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                future.result()
                if index == 1 or index % 10 == 0:
                    print(
                        f"[R3T2:{role}:{repetition}] {index}/{len(pending)}",
                        flush=True,
                    )
    return pd.concat(
        [pd.read_csv(path) for _, _, path in jobs], ignore_index=True
    )


def _pair_repetitions(
    first: pd.DataFrame, second: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["seed", "trajectory_id", "step_index"]
    columns = keys + [
        "current_a",
        "next_soc",
        "next_temperature_c",
        "terminal_voltage_v",
        "thermal_intervened",
        "voltage_intervened",
        "guard_exceeded",
        "empty_final_interval",
    ]
    paired = first[columns].merge(
        second[columns],
        on=keys,
        suffixes=("_a", "_b"),
        how="outer",
        indicator=True,
    )
    valid = paired.dropna(subset=["current_a_a", "current_a_b"]).copy()
    valid["current_difference_a"] = (
        valid.current_a_a - valid.current_a_b
    ).abs()
    valid["soc_difference"] = (valid.next_soc_a - valid.next_soc_b).abs()
    valid["temperature_difference_c"] = (
        valid.next_temperature_c_a - valid.next_temperature_c_b
    ).abs()
    valid["voltage_difference_v"] = (
        valid.terminal_voltage_v_a - valid.terminal_voltage_v_b
    ).abs()
    decision_columns = (
        "thermal_intervened",
        "voltage_intervened",
        "guard_exceeded",
        "empty_final_interval",
    )
    decision_mismatches = sum(
        int((paired[f"{column}_a"] != paired[f"{column}_b"]).sum())
        for column in decision_columns
    )
    summary = {
        "pair_count": int(
            valid[["seed", "trajectory_id"]].drop_duplicates().shape[0]
        ),
        "paired_step_count": len(valid),
        "all_steps_paired": bool((paired._merge == "both").all()),
        "maximum_current_difference_a": float(
            valid.current_difference_a.max()
        ),
        "maximum_soc_difference": float(valid.soc_difference.max()),
        "maximum_temperature_difference_c": float(
            valid.temperature_difference_c.max()
        ),
        "maximum_voltage_difference_v": float(
            valid.voltage_difference_v.max()
        ),
        "decision_mismatch_count": int(decision_mismatches),
    }
    return valid, summary


def run_development(
    config: Phase7CR3T2Config, root: Path, resume: bool
) -> dict[str, Any]:
    state_freeze = prepare_states(config, root)
    data_dir = root / config.section("output")["data_directory"]
    tolerance_path = data_dir / "frozen_repeatability_tolerances.json"
    if tolerance_path.exists():
        return json.loads(tolerance_path.read_text(encoding="utf-8"))
    seed = int(config.section("repeatability_contract")["development_seed"])
    first = _run_repetition(config, root, "development", "a", (seed,), resume)
    second = _run_repetition(config, root, "development", "b", (seed,), resume)
    first.to_csv(data_dir / "development_repetition_a.csv", index=False)
    second.to_csv(data_dir / "development_repetition_b.csv", index=False)
    paired, summary = _pair_repetitions(first, second)
    paired.to_csv(data_dir / "development_paired_steps.csv", index=False)
    contract = config.section("repeatability_contract")
    tolerances = {
        "current_difference_a": max(
            float(contract["historical_current_difference_a"]),
            summary["maximum_current_difference_a"],
        )
        + float(contract["current_engineering_margin_a"]),
        "soc_difference": max(
            float(contract["historical_soc_difference"]),
            summary["maximum_soc_difference"],
        )
        + float(contract["soc_engineering_margin"]),
        "temperature_difference_c": max(
            float(contract["historical_temperature_difference_c"]),
            summary["maximum_temperature_difference_c"],
        )
        + float(contract["temperature_engineering_margin_c"]),
    }
    checks = {
        "expected_pair_count": summary["pair_count"]
        == int(contract["expected_development_pair_count"]),
        "all_steps_paired": summary["all_steps_paired"],
        "decision_signatures_identical": summary["decision_mismatch_count"]
        == 0,
    }
    success = bool(all(checks.values()))
    payload = {
        "phase": "Phase 7C-R3T2 development",
        "status": "tolerances_frozen_before_confirmation"
        if success
        else "strict_stop_failed",
        "development_summary": summary,
        "historical_limits": {
            "current_difference_a": float(
                contract["historical_current_difference_a"]
            ),
            "soc_difference": float(contract["historical_soc_difference"]),
            "temperature_difference_c": float(
                contract["historical_temperature_difference_c"]
            ),
        },
        "engineering_margins": {
            "current_difference_a": float(
                contract["current_engineering_margin_a"]
            ),
            "soc_difference": float(contract["soc_engineering_margin"]),
            "temperature_difference_c": float(
                contract["temperature_engineering_margin_c"]
            ),
        },
        "frozen_tolerances": tolerances,
        "checks": checks,
        "success": success,
        "state_hashes": {
            name: record["sha256"]
            for name, record in state_freeze["files"].items()
        },
        "config_sha256": _sha256(root / CONFIG_RELATIVE, True),
        "runner_sha256": _sha256(Path(__file__), True),
        "confirmation_started": False,
        "confirmation_used_for_retuning": False,
        "level4_entered": False,
    }
    tolerance_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _verify_tolerance_freeze(
    config: Phase7CR3T2Config, root: Path
) -> dict[str, Any]:
    path = (
        root
        / config.section("output")["data_directory"]
        / "frozen_repeatability_tolerances.json"
    )
    if not path.exists():
        raise RuntimeError("R3T2 tolerances are not frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload["success"] or payload["confirmation_used_for_retuning"]:
        raise RuntimeError("R3T2 tolerance freeze is invalid")
    if payload["config_sha256"] != _sha256(root / CONFIG_RELATIVE, True):
        raise RuntimeError("R3T2 config changed after tolerance freeze")
    if payload["runner_sha256"] != _sha256(Path(__file__), True):
        raise RuntimeError("R3T2 runner changed after tolerance freeze")
    return payload


def run_confirmation(
    config: Phase7CR3T2Config, root: Path, resume: bool
) -> dict[str, Any]:
    tolerance_freeze = _verify_tolerance_freeze(config, root)
    output = config.section("output")
    data_dir = root / output["data_directory"]
    result_dir = root / output["result_directory"]
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    seeds = tuple(
        int(seed)
        for seed in config.section("repeatability_contract")[
            "confirmation_seeds"
        ]
    )
    first = _run_repetition(
        config, root, "confirmation", "a", seeds, resume
    )
    second = _run_repetition(
        config, root, "confirmation", "b", seeds, resume
    )
    first_path = data_dir / "confirmation_repetition_a.csv"
    second_path = data_dir / "confirmation_repetition_b.csv"
    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)
    paired, summary = _pair_repetitions(first, second)
    paired_path = data_dir / "confirmation_paired_steps.csv"
    paired.to_csv(paired_path, index=False)
    r3_config = load_phase7cr3_config(
        root / config.section("sources")["phase7cr3_config"]
    )
    base_config = load_phase7cr2f_config(
        root / r3_config.section("sources")["phase7cr2f_config"]
    )
    combined = pd.concat([first, second], ignore_index=True)
    trajectory = _trajectory_metrics(base_config, combined)
    trajectory_path = data_dir / "trajectory_metrics.csv"
    trajectory.to_csv(trajectory_path, index=False)
    frozen_mpc_ms = float(
        config.section("repeatability_contract")["frozen_mpc_total_mean_ms"]
    )
    timing_rows = []
    for (repetition, seed), group in combined[
        combined.current_a.notna()
    ].groupby(["repetition", "seed"]):
        total_ms = 1000.0 * float(
            (group.teacher_time_s + group.supervisor_time_s).mean()
        )
        timing_rows.append(
            {
                "repetition": repetition,
                "seed": int(seed),
                "ann_total_mean_ms": total_ms,
                "end_to_end_speedup": frozen_mpc_ms / total_ms,
            }
        )
    timing = pd.DataFrame(timing_rows)
    timing_path = data_dir / "timing_metrics.csv"
    timing.to_csv(timing_path, index=False)
    tolerances = tolerance_freeze["frozen_tolerances"]
    physical = {
        "trajectory_count": int(len(trajectory)),
        "target_reach_fraction": float(trajectory.target_reached.mean()),
        "maximum_voltage_v": float(trajectory.maximum_voltage_v.max()),
        "maximum_temperature_c": float(trajectory.maximum_temperature_c.max()),
        "minimum_current_a": float(trajectory.minimum_current_a.min()),
        "maximum_current_a": float(trajectory.maximum_current_a.max()),
        "maximum_current_step_a": float(
            trajectory.maximum_current_step_a.max()
        ),
        "guard_exceedance_count": int(
            trajectory.guard_exceedance_count.sum()
        ),
        "empty_voltage_slew_count": int(
            trajectory.empty_voltage_slew_count.sum()
        ),
        "empty_thermal_slew_count": int(
            trajectory.empty_thermal_slew_count.sum()
        ),
        "prediction_infeasible_count": int(
            trajectory.prediction_infeasible_count.sum()
        ),
        "sustained_oscillation_count": int(
            trajectory.sustained_oscillation_count.sum()
        ),
    }
    gates = config.section("gates")
    contract = config.section("repeatability_contract")
    checks = {
        "expected_pair_count": summary["pair_count"]
        == int(contract["expected_confirmation_pair_count"]),
        "all_steps_paired": summary["all_steps_paired"],
        "current_within_frozen_tolerance": summary[
            "maximum_current_difference_a"
        ]
        <= tolerances["current_difference_a"],
        "soc_within_frozen_tolerance": summary["maximum_soc_difference"]
        <= tolerances["soc_difference"],
        "temperature_within_frozen_tolerance": summary[
            "maximum_temperature_difference_c"
        ]
        <= tolerances["temperature_difference_c"],
        "decision_signatures_identical": summary["decision_mismatch_count"]
        == 0,
        "target_reach_100_percent": physical["target_reach_fraction"]
        == 1.0,
        "voltage_safe": physical["maximum_voltage_v"]
        <= float(gates["maximum_voltage_v"]),
        "temperature_safe": physical["maximum_temperature_c"]
        <= float(gates["maximum_average_temperature_c"]),
        "current_bounds_safe": physical["minimum_current_a"]
        >= float(gates["minimum_current_a"])
        - float(gates["numerical_tolerance"])
        and physical["maximum_current_a"]
        <= float(gates["maximum_current_a"])
        + float(gates["numerical_tolerance"]),
        "slew_safe": physical["maximum_current_step_a"]
        <= float(gates["maximum_current_step_a"])
        + float(gates["numerical_tolerance"]),
        "zero_guard_exceedance": physical["guard_exceedance_count"] == 0,
        "zero_empty_intervals": physical["empty_voltage_slew_count"] == 0
        and physical["empty_thermal_slew_count"] == 0,
        "zero_prediction_infeasible": physical[
            "prediction_infeasible_count"
        ]
        == 0,
        "zero_sustained_oscillation": physical[
            "sustained_oscillation_count"
        ]
        == 0,
        "all_speedups_above_100": bool(
            (
                timing.end_to_end_speedup
                > float(contract["minimum_end_to_end_speedup"])
            ).all()
        ),
    }
    success = bool(all(checks.values()))
    payload = {
        "phase": "Phase 7C-R3T2",
        "status": "strict_passed" if success else "strict_stop_failed",
        "tolerance_freeze": tolerance_freeze,
        "confirmation_summary": summary,
        "physical_summary": physical,
        "timing_summary": {
            "minimum_end_to_end_speedup": float(
                timing.end_to_end_speedup.min()
            ),
            "maximum_end_to_end_speedup": float(
                timing.end_to_end_speedup.max()
            ),
        },
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "frozen_model_hashes": _model_hashes(root),
        "success": success,
        "decision": {
            "level4_authorized": success,
            "level4_entered": False,
            "ann_retrained": False,
            "safety_contract_changed": False,
            "confirmation_used_for_retuning": False,
        },
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = result_dir / "PHASE7C-R3T2_中文实验报告.md"
    report_path.write_text(
        "# Phase 7C-R3T2 DFN数值重复性验证\n\n"
        f"- 状态：{payload['status']}；\n"
        f"- 确认配对：{summary['pair_count']}组；\n"
        f"- 最大电流差异：{summary['maximum_current_difference_a']:.12g} A；\n"
        f"- 冻结电流容差：{tolerances['current_difference_a']:.12g} A；\n"
        f"- 最大SOC差异：{summary['maximum_soc_difference']:.12g}；\n"
        f"- 最大温度差异：{summary['maximum_temperature_difference_c']:.12g} ℃；\n"
        f"- 端到端加速：{payload['timing_summary']['minimum_end_to_end_speedup']:.2f}×–"
        f"{payload['timing_summary']['maximum_end_to_end_speedup']:.2f}×。\n\n"
        + (
            "全部重复性、物理安全和速度门槛通过；Level 4已获准但尚未进入。\n"
            if success
            else "至少一项预注册门槛失败；不得进入Level 4。\n"
        ),
        encoding="utf-8",
    )
    artifacts = [
        CONFIG_RELATIVE,
        "src/battery_fast_charge/phase7cr3t2_config.py",
        "src/battery_fast_charge/phase7cr3t2_runner.py",
        "data/phase7cr3t2_dfn_repeatability/initial_state_freeze.json",
        "data/phase7cr3t2_dfn_repeatability/frozen_repeatability_tolerances.json",
        "data/phase7cr3t2_dfn_repeatability/confirmation_repetition_a.csv",
        "data/phase7cr3t2_dfn_repeatability/confirmation_repetition_b.csv",
        "data/phase7cr3t2_dfn_repeatability/confirmation_paired_steps.csv",
        "data/phase7cr3t2_dfn_repeatability/trajectory_metrics.csv",
        "data/phase7cr3t2_dfn_repeatability/timing_metrics.csv",
        "outputs/phase7cr3t2_dfn_repeatability/metrics.json",
        "outputs/phase7cr3t2_dfn_repeatability/PHASE7C-R3T2_中文实验报告.md",
    ]
    manifest = {
        "phase": "Phase 7C-R3T2",
        "status": payload["status"],
        "level4_authorized": success,
        "level4_entered": False,
        "ann_retrained": False,
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


def run_phase7cr3t2(
    config: Phase7CR3T2Config,
    root: Path,
    stage: str,
    resume: bool = False,
) -> dict[str, Any]:
    if stage == "prepare":
        return prepare_states(config, root)
    if stage == "develop":
        return run_development(config, root, resume)
    if stage == "confirm":
        return run_confirmation(config, root, resume)
    prepare_states(config, root)
    run_development(config, root, resume)
    return run_confirmation(config, root, resume)
