"""Level 4-2 thermal-margin development and one-shot validation."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from . import phase7cr2f3_runner as safe_runner
from .phase7cr2_runner import _trajectory_metrics
from .phase7cr2f_config import load_phase7cr2f_config
from .phase7cr3_runner import _ann_rollout, _model_hashes
from .phase7cr3t_thermal import optimized_thermal_current_limit
from .phase7d1_config import load_phase7d1_config
from .phase7d1_runner import prepare_phase7d1_states
from .phase7d2_config import Phase7D2Config


CONFIG_RELATIVE = "configs/phase7d2_thermal_performance_development.yaml"
RUNNER_RELATIVE = "src/battery_fast_charge/phase7d2_runner.py"


def _sha256(path: Path, normalize_text: bool = False) -> str:
    payload = path.read_bytes()
    if normalize_text:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _verify_sources(config: Phase7D2Config, root: Path) -> dict[str, Any]:
    sources = config.section("sources")
    state_freeze_path = root / sources["phase7d1_state_freeze"]
    guard_path = root / sources["frozen_voltage_guards"]
    if _sha256(state_freeze_path) != sources["phase7d1_state_freeze_sha256"]:
        raise RuntimeError("Level 4-1 state freeze changed")
    if _sha256(guard_path) != sources["frozen_voltage_guards_sha256"]:
        raise RuntimeError("Frozen voltage guards changed")
    freeze = json.loads(state_freeze_path.read_text(encoding="utf-8"))
    state_dir = root / sources["state_directory"]
    for name, record in freeze["files"].items():
        if _sha256(state_dir / name) != record["sha256"]:
            raise RuntimeError(f"Frozen Level 4 state changed: {name}")
    guards = json.loads(guard_path.read_text(encoding="utf-8"))["guards"]
    return {
        "state_freeze_sha256": _sha256(state_freeze_path),
        "voltage_guard_sha256": _sha256(guard_path),
        "state_hashes": {name: record["sha256"] for name, record in freeze["files"].items()},
        "guards": guards,
    }


def _state_rows(config: Phase7D2Config, root: Path, role: str, temperature: int | None = None) -> list[dict[str, Any]]:
    state_dir = root / config.section("sources")["state_directory"]
    frames = []
    temperatures = (temperature,) if temperature is not None else tuple(config.section("datasets")["temperatures_c"])
    for value in temperatures:
        frame = pd.read_csv(state_dir / f"{role}_initial_states_{int(value)}c.csv")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).to_dict(orient="records")


def _thermal_limit_with_guard(
    temperature_c: float,
    ambient_temperature_c: float,
    search_upper_a: float,
    r1: Any,
    braking: bool,
    temperature_guard_c: float,
    prediction_horizon_steps: int,
) -> tuple[float, float]:
    thermal = dict(r1.thermal)
    thermal["temperature_guard_c"] = float(temperature_guard_c)
    thermal["prediction_horizon_steps"] = int(prediction_horizon_steps)
    proxy = SimpleNamespace(thermal=thermal)
    return optimized_thermal_current_limit(
        temperature_c, ambient_temperature_c, search_upper_a, proxy, braking
    )


def _worker(
    config: Phase7D2Config,
    root_text: str,
    initial: dict[str, Any],
    seed: int,
    guards: dict[str, Any],
    variant: str,
    temperature_guard_c: float,
    horizon_steps: int,
    path_text: str,
) -> str:
    original = safe_runner._thermal_current_limit

    def parameterized_limit(
        temperature_c: float,
        ambient_temperature_c: float,
        search_upper_a: float,
        r1: Any,
        braking: bool,
    ) -> tuple[float, float]:
        return _thermal_limit_with_guard(
            temperature_c,
            ambient_temperature_c,
            search_upper_a,
            r1,
            braking,
            temperature_guard_c,
            horizon_steps,
        )

    safe_runner._thermal_current_limit = parameterized_limit
    try:
        frame = _ann_rollout(config, Path(root_text), initial, seed, guards)
    finally:
        safe_runner._thermal_current_limit = original
    frame["role"] = f"{variant}_seed_{seed}"
    frame["level4_variant"] = variant
    frame["thermal_temperature_guard_c"] = temperature_guard_c
    frame["thermal_prediction_horizon_steps"] = horizon_steps
    frame.to_csv(path_text, index=False)
    return path_text


def _run_variant(
    config: Phase7D2Config,
    root: Path,
    role: str,
    variant: str,
    temperature_guard_c: float,
    rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
    guards: dict[str, Any],
    resume: bool,
) -> pd.DataFrame:
    output = config.section("output")
    run_dir = root / output["data_directory"] / "runs" / role / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    horizon = int(config.section("thermal_candidates")["prediction_horizon_steps"])
    jobs = [
        (
            initial,
            seed,
            run_dir / f"ann_{seed}_{initial['trajectory_id']}.csv",
        )
        for seed in seeds
        for initial in rows
    ]
    pending = [job for job in jobs if not (resume and job[2].exists())]
    if pending:
        with ProcessPoolExecutor(max_workers=int(config.section("datasets")["maximum_workers"])) as executor:
            futures = [
                executor.submit(
                    _worker,
                    config,
                    str(root),
                    initial,
                    seed,
                    guards,
                    variant,
                    temperature_guard_c,
                    horizon,
                    str(path),
                )
                for initial, seed, path in pending
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                future.result()
                if index == 1 or index % 10 == 0:
                    print(f"[Level4 {role}/{variant}] {index}/{len(pending)}", flush=True)
    return pd.concat([pd.read_csv(path) for _, _, path in jobs], ignore_index=True)


def _metric_table(config: Phase7D2Config, root: Path, traces: pd.DataFrame) -> pd.DataFrame:
    base = load_phase7cr2f_config(root / config.section("sources")["phase7cr2f_config"])
    metrics = _trajectory_metrics(base, traces)
    role_parts = metrics.role.str.extract(r"^(?P<variant>.+)_seed_(?P<seed>\d+)$")
    metrics["level4_variant"] = role_parts.variant
    metrics["seed"] = role_parts.seed.astype(int)
    return metrics


def _safety_checks(config: Phase7D2Config, metrics: pd.DataFrame) -> dict[str, bool]:
    gates = config.section("gates")
    tol = float(gates["numerical_tolerance"])
    violation_columns = (
        "guard_exceedance_count",
        "empty_voltage_slew_count",
        "empty_thermal_slew_count",
        "solver_failure_count",
        "prediction_infeasible_count",
        "sustained_oscillation_count",
    )
    return {
        "target_reach_100_percent": bool(metrics.target_reached.all()),
        "voltage_safe": bool((metrics.maximum_voltage_v <= gates["maximum_voltage_v"]).all()),
        "temperature_safe": bool((metrics.maximum_temperature_c <= gates["maximum_average_temperature_c"]).all()),
        "current_bounds_safe": bool(
            (metrics.minimum_current_a >= gates["minimum_current_a"] - tol).all()
            and (metrics.maximum_current_a <= gates["maximum_current_a"] + tol).all()
        ),
        "slew_safe": bool((metrics.maximum_current_step_a <= gates["maximum_current_step_a"] + tol).all()),
        "all_violation_counts_zero": bool(all(int(metrics[column].sum()) == 0 for column in violation_columns)),
    }


def _summary_row(
    config: Phase7D2Config,
    traces: pd.DataFrame,
    metrics: pd.DataFrame,
    variant: str,
    temperature_guard_c: float,
) -> dict[str, Any]:
    valid = traces[traces.current_a.notna()]
    speedup = float(config.section("gates")["frozen_mpc_total_mean_ms"]) / (
        1000.0 * float((valid.teacher_time_s + valid.supervisor_time_s).mean())
    )
    checks = _safety_checks(config, metrics)
    checks["speedup_above_100"] = speedup > float(config.section("gates")["minimum_end_to_end_speedup"])
    return {
        "variant": variant,
        "temperature_guard_c": temperature_guard_c,
        "trajectory_count": int(len(metrics)),
        "mean_charge_time_s": float(metrics.charge_time_s.mean()),
        "maximum_charge_time_s": float(metrics.charge_time_s.max()),
        "maximum_voltage_v": float(metrics.maximum_voltage_v.max()),
        "maximum_temperature_c": float(metrics.maximum_temperature_c.max()),
        "mean_voltage_intervention_fraction": float(metrics.voltage_intervention_fraction.mean()),
        "mean_thermal_intervention_fraction": float(metrics.thermal_intervention_fraction.mean()),
        "maximum_thermal_correction_a": float(metrics.maximum_thermal_correction_a.max()),
        "end_to_end_speedup": speedup,
        "strictly_safe": bool(all(checks.values())),
        "checks": checks,
    }


def run_development(config: Phase7D2Config, root: Path, resume: bool) -> dict[str, Any]:
    sources = _verify_sources(config, root)
    output = config.section("output")
    data_dir = root / output["data_directory"]
    result_dir = root / output["result_directory"]
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    selected_path = result_dir / "selected_parameters.json"
    if selected_path.exists():
        return json.loads(selected_path.read_text(encoding="utf-8"))
    rows_all = _state_rows(config, root, "development")
    rows_30 = [row for row in rows_all if int(row["ambient_temperature_c"]) == 30]
    seed = (int(config.section("datasets")["development_seed"]),)
    baseline_guard = float(config.section("thermal_candidates")["baseline_temperature_guard_c"])
    summaries: list[dict[str, Any]] = []
    baseline_traces = _run_variant(config, root, "development", "baseline", baseline_guard, rows_all, seed, sources["guards"], resume)
    baseline_traces.to_csv(data_dir / "development_baseline_trajectories.csv", index=False)
    baseline_metrics = _metric_table(config, root, baseline_traces)
    baseline_metrics.to_csv(data_dir / "development_baseline_metrics.csv", index=False)
    baseline_30 = baseline_metrics[baseline_metrics.ambient_temperature_c == 30]
    baseline_traces_30 = baseline_traces[baseline_traces.ambient_temperature_c == 30]
    summaries.append(_summary_row(config, baseline_traces_30, baseline_30, "baseline", baseline_guard))
    for guard in config.section("thermal_candidates")["candidate_temperature_guards_c"]:
        guard = float(guard)
        variant = f"thermal_guard_{int(round(1000 * guard)):03d}mc"
        traces = _run_variant(config, root, "development", variant, guard, rows_30, seed, sources["guards"], resume)
        traces.to_csv(data_dir / f"development_{variant}_trajectories.csv", index=False)
        metrics = _metric_table(config, root, traces)
        metrics.to_csv(data_dir / f"development_{variant}_metrics.csv", index=False)
        summaries.append(_summary_row(config, traces, metrics, variant, guard))
    flat = pd.DataFrame([{key: value for key, value in row.items() if key != "checks"} for row in summaries])
    flat.to_csv(data_dir / "development_candidate_summary.csv", index=False)
    baseline = summaries[0]
    if not baseline["strictly_safe"]:
        raise RuntimeError("Level 4 baseline failed on development data")
    eligible = [row for row in summaries if row["strictly_safe"]]
    eligible.sort(key=lambda row: (row["mean_charge_time_s"], row["mean_thermal_intervention_fraction"], row["maximum_thermal_correction_a"]))
    selected = eligible[0]
    improvement = (baseline["mean_charge_time_s"] - selected["mean_charge_time_s"]) / baseline["mean_charge_time_s"]
    required = float(config.section("selection_contract")["minimum_30c_mean_charge_time_improvement_fraction"])
    success = bool(selected["variant"] != "baseline" and improvement >= required)
    payload = {
        "phase": config.payload["phase"],
        "status": "parameters_frozen_before_internal_validation" if success else "strict_stop_failed",
        "development_seed": seed[0],
        "development_candidate_summaries": summaries,
        "selected_variant": selected["variant"],
        "selected_temperature_guard_c": selected["temperature_guard_c"],
        "prediction_horizon_steps": int(config.section("thermal_candidates")["prediction_horizon_steps"]),
        "baseline_30c_mean_charge_time_s": baseline["mean_charge_time_s"],
        "selected_30c_mean_charge_time_s": selected["mean_charge_time_s"],
        "development_improvement_fraction": improvement,
        "minimum_required_improvement_fraction": required,
        "voltage_guards_unchanged": True,
        "ann_retrained": False,
        "success": success,
        "config_sha256": _sha256(root / CONFIG_RELATIVE, True),
        "runner_sha256": _sha256(root / RUNNER_RELATIVE, True),
        "source_verification": {key: value for key, value in sources.items() if key != "guards"},
        "frozen_model_hashes": _model_hashes(root),
        "internal_validation_started": False,
        "internal_validation_used_for_retuning": False,
        "independent_confirmation_created": False,
    }
    selected_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_internal_validation(config: Phase7D2Config, root: Path, resume: bool) -> dict[str, Any]:
    selection = run_development(config, root, True)
    if not selection["success"]:
        raise RuntimeError("Level 4 development did not authorize validation")
    if selection["config_sha256"] != _sha256(root / CONFIG_RELATIVE, True):
        raise RuntimeError("Level 4 config changed after parameter freeze")
    if selection["runner_sha256"] != _sha256(root / RUNNER_RELATIVE, True):
        raise RuntimeError("Level 4 runner changed after parameter freeze")
    output = config.section("output")
    data_dir = root / output["data_directory"]
    result_dir = root / output["result_directory"]
    metrics_path = result_dir / "internal_validation_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    sources = _verify_sources(config, root)
    rows = _state_rows(config, root, "internal_validation")
    seeds = tuple(int(value) for value in config.section("datasets")["validation_seeds"])
    baseline_guard = float(config.section("thermal_candidates")["baseline_temperature_guard_c"])
    selected_guard = float(selection["selected_temperature_guard_c"])
    selected_variant = str(selection["selected_variant"])
    baseline = _run_variant(config, root, "internal_validation", "baseline", baseline_guard, rows, seeds, sources["guards"], resume)
    selected = _run_variant(config, root, "internal_validation", selected_variant, selected_guard, rows, seeds, sources["guards"], resume)
    baseline.to_csv(data_dir / "internal_validation_baseline_trajectories.csv", index=False)
    selected.to_csv(data_dir / "internal_validation_selected_trajectories.csv", index=False)
    baseline_metrics = _metric_table(config, root, baseline)
    selected_metrics = _metric_table(config, root, selected)
    baseline_metrics.to_csv(data_dir / "internal_validation_baseline_metrics.csv", index=False)
    selected_metrics.to_csv(data_dir / "internal_validation_selected_metrics.csv", index=False)
    baseline_summary = _summary_row(config, baseline, baseline_metrics, "baseline", baseline_guard)
    selected_summary = _summary_row(config, selected, selected_metrics, selected_variant, selected_guard)
    baseline_30 = baseline_metrics[baseline_metrics.ambient_temperature_c == 30]
    selected_30 = selected_metrics[selected_metrics.ambient_temperature_c == 30]
    improvement_30 = (baseline_30.charge_time_s.mean() - selected_30.charge_time_s.mean()) / baseline_30.charge_time_s.mean()
    overall_noninferior = selected_metrics.charge_time_s.mean() <= baseline_metrics.charge_time_s.mean()
    required = float(config.section("selection_contract")["minimum_30c_mean_charge_time_improvement_fraction"])
    checks = {
        "baseline_strictly_safe": baseline_summary["strictly_safe"],
        "selected_strictly_safe": selected_summary["strictly_safe"],
        "overall_charge_time_noninferior": bool(overall_noninferior),
        "30c_improvement_meets_contract": bool(improvement_30 >= required),
        "all_five_seeds_present": set(selected_metrics.seed) == set(seeds),
        "expected_trajectory_count": len(selected_metrics) == len(rows) * len(seeds),
    }
    success = bool(all(checks.values()))
    payload = {
        "phase": "Phase 7D-Level 4-2 internal validation",
        "status": "strict_passed" if success else "strict_stop_failed",
        "selected_variant": selected_variant,
        "selected_temperature_guard_c": selected_guard,
        "baseline_summary": baseline_summary,
        "selected_summary": selected_summary,
        "baseline_30c_mean_charge_time_s": float(baseline_30.charge_time_s.mean()),
        "selected_30c_mean_charge_time_s": float(selected_30.charge_time_s.mean()),
        "validation_30c_improvement_fraction": float(improvement_30),
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "success": success,
        "selection_hash": _sha256(result_dir / "selected_parameters.json"),
        "internal_validation_used_for_retuning": False,
        "independent_confirmation_authorized": success,
        "ann_retrained": False,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload

