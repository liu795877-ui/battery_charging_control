"""Phase 7C-R3T equivalence audit and optimized runtime confirmation."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from . import phase7cr2f3_runner as safe_runner
from .phase7cr1_config import load_phase7cr1_config
from .phase7cr2_runner import _thermal_current_limit, _trajectory_metrics
from .phase7cr2f_config import load_phase7cr2f_config
from .phase7cr3_config import load_phase7cr3_config
from .phase7cr3_runner import _ann_rollout, _confirmation_rows, _model_hashes
from .phase7cr3t_config import Phase7CR3TConfig
from .phase7cr3t_thermal import optimized_thermal_current_limit


CONFIG_RELATIVE = "configs/phase7cr3t_supervisor_runtime.yaml"


def _sha256(path: Path, normalize_text: bool = False) -> str:
    payload = path.read_bytes()
    if normalize_text:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def verify_frozen_r3(config: Phase7CR3TConfig, root: Path) -> dict[str, Any]:
    sources = config.section("sources")
    records: dict[str, Any] = {}
    for path_key, hash_key in (
        ("phase7cr3_freeze_manifest", "phase7cr3_freeze_manifest_sha256"),
        ("phase7cr3_strict_audit", "phase7cr3_strict_audit_sha256"),
        ("phase7cr3_ann_traces", "phase7cr3_ann_traces_sha256"),
    ):
        relative = sources[path_key]
        expected = sources[hash_key]
        actual = _sha256(root / relative)
        records[relative] = {"expected": expected, "actual": actual, "matched": expected == actual}
        if expected != actual:
            raise RuntimeError(f"Frozen R3 artifact changed: {relative}")
    audit = json.loads((root / sources["phase7cr3_strict_audit"]).read_text(encoding="utf-8"))
    if audit["status"] != "strict_stop_failed" or audit["failed_checks"] != ["speedup_above_100"]:
        raise RuntimeError("R3T requires the frozen speed-only R3 failure")
    contract = config.section("optimization_contract")
    forbidden = (
        "ann_retraining_authorized",
        "voltage_guard_change_authorized",
        "thermal_model_change_authorized",
        "control_output_change_authorized",
        "level4_entry_authorized",
    )
    if any(contract[key] for key in forbidden):
        raise RuntimeError("R3T scope or Level 4 boundary changed")
    return {"records": records, "r3_failed_checks": audit["failed_checks"]}


def run_equivalence_audit(config: Phase7CR3TConfig, root: Path) -> dict[str, Any]:
    source_verification = verify_frozen_r3(config, root)
    data_dir = root / config.section("output")["data_directory"]
    result_dir = root / config.section("output")["result_directory"]
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = result_dir / "equivalence_freeze.json"
    if freeze_path.exists():
        return json.loads(freeze_path.read_text(encoding="utf-8"))
    traces = pd.read_csv(root / config.section("sources")["phase7cr3_ann_traces"])
    r1 = load_phase7cr1_config(root / config.section("sources")["phase7cr1_config"])
    maximum_current_difference = 0.0
    maximum_peak_difference = 0.0
    current_mismatch_count = 0
    evaluated_calls = 0
    old_seconds = 0.0
    optimized_seconds = 0.0
    for row in traces.itertuples(index=False):
        search_upper = min(float(row.slew_upper_a), float(row.voltage_safe_current_max_a))
        for braking in (False, True):
            started = perf_counter()
            old = _thermal_current_limit(
                float(row.temperature_c),
                float(row.ambient_temperature_c),
                search_upper,
                r1,
                braking,
            )
            old_seconds += perf_counter() - started
            started = perf_counter()
            optimized = optimized_thermal_current_limit(
                float(row.temperature_c),
                float(row.ambient_temperature_c),
                search_upper,
                r1,
                braking,
            )
            optimized_seconds += perf_counter() - started
            current_difference = abs(old[0] - optimized[0])
            peak_difference = abs(old[1] - optimized[1])
            maximum_current_difference = max(maximum_current_difference, current_difference)
            maximum_peak_difference = max(maximum_peak_difference, peak_difference)
            current_mismatch_count += int(current_difference != 0.0)
            evaluated_calls += 1
    contract = config.section("optimization_contract")
    checks = {
        "all_r3_steps_and_both_modes_audited": evaluated_calls == 2 * len(traces),
        "current_limit_exact": maximum_current_difference <= float(contract["maximum_current_limit_difference_a"]),
        "peak_temperature_equivalent": maximum_peak_difference <= float(contract["maximum_peak_temperature_difference_c"]),
        "optimized_microbenchmark_faster": optimized_seconds < old_seconds,
    }
    success = bool(all(checks.values()))
    payload = {
        "phase": "Phase 7C-R3T equivalence development",
        "status": "optimization_frozen_before_confirmation" if success else "strict_stop_failed",
        "evaluated_trace_rows": len(traces),
        "evaluated_supervisor_calls": evaluated_calls,
        "maximum_current_limit_difference_a": maximum_current_difference,
        "maximum_peak_temperature_difference_c": maximum_peak_difference,
        "current_mismatch_count": current_mismatch_count,
        "old_supervisor_benchmark_s": old_seconds,
        "optimized_supervisor_benchmark_s": optimized_seconds,
        "microbenchmark_speedup": old_seconds / optimized_seconds,
        "checks": checks,
        "success": success,
        "source_verification": source_verification,
        "config_sha256": _sha256(root / CONFIG_RELATIVE, True),
        "implementation_sha256": _sha256(Path(__file__).with_name("phase7cr3t_thermal.py"), True),
        "confirmation_started": False,
        "level4_entered": False,
    }
    freeze_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _optimized_ann_worker(
    r3_config: Any,
    root_text: str,
    initial: dict[str, Any],
    seed: int,
    guards: dict[str, Any],
    path_text: str,
) -> str:
    original = safe_runner._thermal_current_limit
    safe_runner._thermal_current_limit = optimized_thermal_current_limit
    try:
        frame = _ann_rollout(r3_config, Path(root_text), initial, seed, guards)
    finally:
        safe_runner._thermal_current_limit = original
    frame.to_csv(path_text, index=False)
    return path_text


def run_confirmation(config: Phase7CR3TConfig, root: Path, resume: bool) -> dict[str, Any]:
    freeze = run_equivalence_audit(config, root)
    if not freeze["success"]:
        raise RuntimeError("R3T equivalence audit did not pass")
    if freeze["config_sha256"] != _sha256(root / CONFIG_RELATIVE, True):
        raise RuntimeError("R3T config changed after optimization freeze")
    if freeze["implementation_sha256"] != _sha256(Path(__file__).with_name("phase7cr3t_thermal.py"), True):
        raise RuntimeError("R3T implementation changed after optimization freeze")
    output = config.section("output")
    data_dir = root / output["data_directory"]
    result_dir = root / output["result_directory"]
    metrics_path = result_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    r3_config = load_phase7cr3_config(root / config.section("sources")["phase7cr3_config"])
    guards = json.loads((root / r3_config.section("sources")["phase7cr2f5_guards"]).read_text(encoding="utf-8"))["guards"]
    seeds = tuple(int(seed) for seed in config.section("execution")["seeds"])
    run_dir = data_dir / "runs" / "optimized_confirmation"
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (initial, seed, run_dir / f"ann_{seed}_{initial['trajectory_id']}.csv")
        for seed in seeds
        for initial in _confirmation_rows(r3_config, root)
    ]
    pending = [(initial, seed, path) for initial, seed, path in jobs if not (resume and path.exists())]
    if pending:
        with ProcessPoolExecutor(max_workers=int(config.section("execution")["maximum_workers"])) as executor:
            futures = [
                executor.submit(
                    _optimized_ann_worker,
                    r3_config,
                    str(root),
                    initial,
                    seed,
                    guards,
                    str(path),
                )
                for initial, seed, path in pending
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                future.result()
                if index == 1 or index % 10 == 0:
                    print(f"[R3T] {index}/{len(pending)}", flush=True)
    optimized = pd.concat([pd.read_csv(path) for _, _, path in jobs], ignore_index=True)
    optimized_path = data_dir / "optimized_confirmation_trajectories.csv"
    optimized.to_csv(optimized_path, index=False)
    baseline = pd.read_csv(root / config.section("sources")["phase7cr3_ann_traces"])
    paired = baseline[["seed", "trajectory_id", "step_index", "current_a"]].merge(
        optimized[["seed", "trajectory_id", "step_index", "current_a"]],
        on=["seed", "trajectory_id", "step_index"],
        suffixes=("_baseline", "_optimized"),
        how="outer",
        indicator=True,
    )
    paired_valid = paired.dropna(subset=["current_a_baseline", "current_a_optimized"])
    maximum_current_difference = float(
        (paired_valid.current_a_baseline - paired_valid.current_a_optimized).abs().max()
    )
    base_config = load_phase7cr2f_config(root / r3_config.section("sources")["phase7cr2f_config"])
    trajectory = _trajectory_metrics(base_config, optimized)
    trajectory.to_csv(data_dir / "trajectory_metrics.csv", index=False)
    valid = optimized[optimized.current_a.notna()]
    frozen_mpc_ms = float(config.section("optimization_contract")["frozen_mpc_total_mean_ms"])
    seed_rows = []
    for seed, group in valid.groupby("seed"):
        total_mean_ms = 1000.0 * float((group.teacher_time_s + group.supervisor_time_s).mean())
        seed_rows.append(
            {
                "seed": int(seed),
                "ann_candidate_mean_ms": 1000.0 * float(group.teacher_time_s.mean()),
                "supervisor_mean_ms": 1000.0 * float(group.supervisor_time_s.mean()),
                "ann_total_mean_ms": total_mean_ms,
                "end_to_end_speedup": frozen_mpc_ms / total_mean_ms,
            }
        )
    seed_metrics = pd.DataFrame(seed_rows)
    seed_metrics.to_csv(data_dir / "seed_timing_metrics.csv", index=False)
    contract = config.section("optimization_contract")
    physical = {
        "trajectory_count": int(len(trajectory)),
        "target_reach_fraction": float(trajectory.target_reached.mean()),
        "maximum_voltage_v": float(trajectory.maximum_voltage_v.max()),
        "maximum_temperature_c": float(trajectory.maximum_temperature_c.max()),
        "minimum_current_a": float(trajectory.minimum_current_a.min()),
        "maximum_current_a": float(trajectory.maximum_current_a.max()),
        "maximum_current_step_a": float(trajectory.maximum_current_step_a.max()),
        "guard_exceedance_count": int(trajectory.guard_exceedance_count.sum()),
        "empty_voltage_slew_count": int(trajectory.empty_voltage_slew_count.sum()),
        "empty_thermal_slew_count": int(trajectory.empty_thermal_slew_count.sum()),
        "prediction_infeasible_count": int(trajectory.prediction_infeasible_count.sum()),
        "sustained_oscillation_count": int(trajectory.sustained_oscillation_count.sum()),
    }
    checks = {
        "trajectory_count_240": physical["trajectory_count"] == int(config.section("execution")["expected_trajectory_count"]),
        "all_steps_paired": bool((paired._merge == "both").all()),
        "closed_loop_current_exact": maximum_current_difference <= float(contract["maximum_closed_loop_current_difference_a"]),
        "target_reach_100_percent": physical["target_reach_fraction"] == 1.0,
        "voltage_safe": physical["maximum_voltage_v"] <= 4.200001,
        "temperature_safe": physical["maximum_temperature_c"] <= 35.0,
        "current_bounds_safe": physical["minimum_current_a"] >= -1e-9 and physical["maximum_current_a"] <= 10.0 + 1e-9,
        "slew_safe": physical["maximum_current_step_a"] <= 2.0 + 1e-9,
        "zero_guard_exceedance": physical["guard_exceedance_count"] == 0,
        "zero_empty_intervals": physical["empty_voltage_slew_count"] == 0 and physical["empty_thermal_slew_count"] == 0,
        "zero_prediction_infeasible": physical["prediction_infeasible_count"] == 0,
        "zero_sustained_oscillation": physical["sustained_oscillation_count"] == 0,
        "all_seed_speedups_above_100": bool((seed_metrics.end_to_end_speedup > float(contract["minimum_end_to_end_speedup"])).all()),
        "mean_total_runtime_below_contract": float(seed_metrics.ann_total_mean_ms.mean()) <= float(contract["maximum_ann_total_mean_ms"]),
    }
    success = bool(all(checks.values()))
    payload = {
        "phase": "Phase 7C-R3T",
        "status": "strict_passed" if success else "strict_stop_failed",
        "equivalence_freeze": freeze,
        "physical_summary": physical,
        "maximum_closed_loop_current_difference_a": maximum_current_difference,
        "seed_timing_metrics": seed_rows,
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
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_phase7cr3t(
    config: Phase7CR3TConfig,
    root: Path,
    stage: str,
    resume: bool = False,
) -> dict[str, Any]:
    if stage == "equivalence":
        return run_equivalence_audit(config, root)
    if stage == "confirm":
        return run_confirmation(config, root, resume)
    run_equivalence_audit(config, root)
    return run_confirmation(config, root, resume)
