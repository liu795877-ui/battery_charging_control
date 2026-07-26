"""Independent R3 confirmation; safe MPC must pass before frozen ANN."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from .phase7cr2f2_runner import _sha256, _state_columns
from .phase7cr2f3_runner import _all_historical_values, _build_state_set, _checks, _run_rows, _temperature_stage_counts
from . import phase7cr2f3_runner as safe_runner
from .phase7b1b_config import load_phase7b1b_config
from .phase7b1b_runner import _load_context
from .phase7cr2f_config import load_phase7cr2f_config
from .phase7cr2f5_runner import verify_frozen_r2f4
from .phase7cr2f_runner import verify_known_teacher_regressions
from .phase7cr2_runner import _trajectory_metrics
from .phase7cr3_config import Phase7CR3Config


CONFIG_RELATIVE = "configs/phase7cr3_independent_confirmation.yaml"


def verify_r2f5(config: Phase7CR3Config, root: Path) -> dict[str, Any]:
    sources = config.section("sources")
    path = root / sources["phase7cr2f5_manifest"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != sources["phase7cr2f5_manifest_sha256"]:
        raise RuntimeError("R2F5 manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["status"] != "strict_passed":
        raise RuntimeError("R2F5 did not strictly pass")
    if manifest["r3_initial_states_generated"] or manifest["level4_entered"]:
        raise RuntimeError("R2F5 evidence boundary changed")
    contract = config.section("control_contract")
    if not contract["safe_mpc_must_pass_before_ann"]:
        raise RuntimeError("R3 sequencing contract is missing")
    if contract["ann_retraining_authorized"] or contract["level4_entry_authorized"]:
        raise RuntimeError("R3 cannot retrain ANN or pre-authorize Level 4")
    return {"manifest_sha256": actual, "status": manifest["status"]}


def prepare_states(config: Phase7CR3Config, root: Path) -> dict[str, Any]:
    verification = verify_r2f5(config, root)
    data_dir = root / config.section("output")["data_directory"]
    data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = data_dir / "initial_state_freeze.json"
    if freeze_path.exists():
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
        for name, record in payload["files"].items():
            if _sha256(data_dir / name) != record["sha256"]:
                raise RuntimeError(f"R3 state hash mismatch: {name}")
        return payload
    existing = _all_historical_values(config, root)
    extra = []
    for key in ("phase7cr2f5_development_states", "phase7cr2f5_internal_states"):
        extra.append(pd.read_csv(root / config.section("sources")[key])[_state_columns()].to_numpy(float))
    existing = np.vstack([existing, *extra])
    files: dict[str, Any] = {}
    state_sets: list[np.ndarray] = []
    for temperature in (15, 30):
        frame, existing = _build_state_set(config, root, "confirmation", temperature, existing)
        frame["trajectory_id"] = frame.trajectory_id.str.replace("phase7cr2f3_", "phase7cr3_", regex=False)
        path = data_dir / f"confirmation_initial_states_{temperature}c.csv"
        frame.to_csv(path, index=False)
        state_sets.append(frame[_state_columns()].to_numpy(float))
        files[path.name] = {
            "sha256": _sha256(path),
            "trajectory_count": len(frame),
            "temperature_c": temperature,
            "design_seed": int(frame.design_seed.iloc[0]),
            "zero_residual_count": int(np.isclose(frame.initial_measured_residual_v, 0.0).sum()),
        }
    if np.any(np.all(np.isclose(state_sets[0][:, None, :], state_sets[1][None, :, :], atol=1e-14), axis=2)):
        raise RuntimeError("R3 temperature state sets overlap")
    payload = {
        "phase": "Phase 7C-R3",
        "status": "frozen_before_any_confirmation_rollout",
        "not_teacher_data": True,
        "not_previous_validation_data": True,
        "safe_mpc_must_pass_before_ann": True,
        "config_sha256": _sha256(root / CONFIG_RELATIVE, True),
        "files": files,
        "r2f5_verification": verification,
    }
    freeze_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _confirmation_rows(config: Phase7CR3Config, root: Path) -> list[dict[str, Any]]:
    data_dir = root / config.section("output")["data_directory"]
    frames = [pd.read_csv(data_dir / f"confirmation_initial_states_{temperature}c.csv") for temperature in (15, 30)]
    return pd.concat(frames, ignore_index=True).to_dict(orient="records")


def run_safe_mpc(config: Phase7CR3Config, root: Path, resume: bool) -> dict[str, Any]:
    freeze = prepare_states(config, root)
    data_dir = root / config.section("output")["data_directory"]
    result_dir = root / config.section("output")["result_directory"]
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / "safe_mpc_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    guards = json.loads((root / config.section("sources")["phase7cr2f5_guards"]).read_text(encoding="utf-8"))["guards"]
    rows = _run_rows(config, root, _confirmation_rows(config, root), "safe_mpc_confirmation", guards, "safe_mpc_confirmation", resume)
    trajectories_path = data_dir / "safe_mpc_confirmation_trajectories.csv"
    rows.to_csv(trajectories_path, index=False)
    base_config = load_phase7cr2f_config(root / config.section("sources")["phase7cr2f_config"])
    metrics = _trajectory_metrics(base_config, rows)
    metrics.to_csv(data_dir / "safe_mpc_trajectory_metrics.csv", index=False)
    checks = _checks(config, metrics, rows)
    teacher = verify_known_teacher_regressions(base_config, root)
    success = bool(len(metrics) == 48 and all(checks.values()) and teacher["all_passed"])
    payload = {
        "phase": "Phase 7C-R3 safe MPC",
        "trajectory_count": int(len(metrics)),
        "checks": checks,
        "guard_exceedance_counts": _temperature_stage_counts(rows),
        "maximum_voltage_v": float(metrics.maximum_voltage_v.max()),
        "maximum_temperature_c": float(metrics.maximum_temperature_c.max()),
        "target_reach_fraction": float(metrics.target_reached.mean()),
        "teacher_regression": teacher,
        "success": success,
        "decision": {"frozen_ann_authorized": success, "ann_executed": False, "level4_entered": False},
        "state_freeze": freeze,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _ann_rollout(config: Phase7CR3Config, root: Path, initial: dict[str, Any], seed: int, guards: dict[str, Any]) -> pd.DataFrame:
    r2f = load_phase7cr2f_config(root / config.section("sources")["phase7cr2f_config"])
    b1 = load_phase7b1b_config(root / r2f.sources["phase7b1b_config"])
    _, _, _, networks, _ = _load_context(b1, root)
    network = networks[seed]
    def ann_candidate(state: Any, model: Any, teacher_config: Any):
        current = float(network.predict(np.asarray([state.soc, state.polarization_1_v, state.polarization_2_v, state.previous_current_a])))
        result = SimpleNamespace(current_a=current, optimizer_success=True, prediction_feasible=True, objective_value=np.nan, status="frozen_ann")
        diagnostics = {
            "teacher_retry_triggered": False, "selected_teacher_branch": "frozen_ann",
            "qualified_teacher_branches": ["frozen_ann"], "default_optimizer_success": True,
            "default_prediction_feasible": True, "default_current_a": current, "default_objective": np.nan,
            "alternative_optimizer_success": True, "alternative_prediction_feasible": True,
            "alternative_current_a": np.nan, "alternative_objective": np.nan,
            "alternative_selected": False, "teacher_branch_objective_improvement": 0.0,
        }
        return result, diagnostics
    original = safe_runner.solve_teacher_r2f
    safe_runner.solve_teacher_r2f = ann_candidate
    try:
        frame = safe_runner._rollout(config, root, initial, f"frozen_ann_seed_{seed}", guards)
    finally:
        safe_runner.solve_teacher_r2f = original
    frame["seed"] = seed
    frame["controller_kind"] = "frozen_ann"
    return frame


def _ann_worker(config: Phase7CR3Config, root_text: str, initial: dict[str, Any], seed: int, guards: dict[str, Any], path_text: str) -> str:
    frame = _ann_rollout(config, Path(root_text), initial, seed, guards)
    frame.to_csv(path_text, index=False)
    return path_text


def _model_hashes(root: Path) -> dict[str, Any]:
    phase7c = json.loads(json.dumps(__import__("yaml").safe_load((root / "configs/phase7c_multitemperature_dfn_validation.yaml").read_text(encoding="utf-8"))))
    expected = {k: v for k, v in phase7c["sources"]["frozen_artifacts"].items() if "level3_deep_lbfgs_seed" in k}
    records = {path: {"expected": value, "actual": hashlib.sha256((root / path).read_bytes()).hexdigest()} for path, value in expected.items()}
    if not all(item["expected"] == item["actual"] for item in records.values()):
        raise RuntimeError("Frozen ANN hash mismatch")
    return records


def run_frozen_ann(config: Phase7CR3Config, root: Path, resume: bool) -> dict[str, Any]:
    safe = run_safe_mpc(config, root, True)
    if not safe["success"] or not safe["decision"]["frozen_ann_authorized"]:
        raise RuntimeError("R3 safe MPC has not authorized frozen ANN")
    models = _model_hashes(root)
    seeds = (22, 42, 73, 101, 137)
    data_dir = root / config.section("output")["data_directory"]
    result_dir = root / config.section("output")["result_directory"]
    metrics_path = result_dir / "ann_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    guards = json.loads((root / config.section("sources")["phase7cr2f5_guards"]).read_text(encoding="utf-8"))["guards"]
    run_dir = data_dir / "runs" / "frozen_ann_confirmation"
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(initial, seed, run_dir / f"ann_{seed}_{initial['trajectory_id']}.csv") for seed in seeds for initial in _confirmation_rows(config, root)]
    pending = [(i, s, p) for i, s, p in jobs if not (resume and p.exists())]
    if pending:
        with ProcessPoolExecutor(max_workers=int(config.section("datasets")["maximum_workers"])) as executor:
            futures = [executor.submit(_ann_worker, config, str(root), initial, seed, guards, str(path)) for initial, seed, path in pending]
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                if index == 1 or index % 10 == 0: print(f"[R3 ANN] {index}/{len(pending)}", flush=True)
    ann = pd.concat([pd.read_csv(path) for _, _, path in jobs], ignore_index=True)
    ann.to_csv(data_dir / "ann_confirmation_trajectories.csv", index=False)
    mpc = pd.read_csv(data_dir / "safe_mpc_confirmation_trajectories.csv")
    gates = config.section("gates")
    seed_rows = []
    for seed in seeds:
        subset = ann[ann.seed == seed]
        nrmse, gaps = [], []
        for trajectory_id, ag in subset.groupby("trajectory_id"):
            mg = mpc[mpc.trajectory_id == trajectory_id]
            paired = mg[["step_index", "current_a"]].merge(ag[["step_index", "current_a"]], on="step_index", suffixes=("_mpc", "_ann")).dropna()
            rmse = float(np.sqrt(np.mean((paired.current_a_ann - paired.current_a_mpc) ** 2)))
            nrmse.append(rmse / 10.0)
            gaps.append(abs(ag.current_a.notna().sum() - mg.current_a.notna().sum()) / mg.current_a.notna().sum())
        valid = subset[subset.current_a.notna()]
        mpc_decision = float((mpc.teacher_time_s + mpc.supervisor_time_s).mean())
        ann_decision = float((valid.teacher_time_s + valid.supervisor_time_s).mean())
        seed_rows.append({"seed": seed, "mean_current_nrmse": float(np.mean(nrmse)), "mean_charge_time_gap_fraction": float(np.mean(gaps)), "target_reach_fraction": float(subset.groupby("trajectory_id").next_soc.max().ge(config.section("datasets")["target_soc"]).mean()), "maximum_voltage_v": float(valid.terminal_voltage_v.max()), "maximum_temperature_c": float(valid.next_temperature_c.max()), "guard_exceedance_count": int(valid.guard_exceeded.sum()), "empty_interval_count": int(valid.empty_final_interval.sum()), "maximum_current_step_a": float(valid.current_step_a.max()), "speedup": mpc_decision / ann_decision})
    table = pd.DataFrame(seed_rows)
    table.to_csv(data_dir / "ann_seed_metrics.csv", index=False)
    checks = {
        "current_nrmse_below_1_percent": bool((table.mean_current_nrmse < 0.01).all()),
        "charge_time_gap_below_2_percent": bool((table.mean_charge_time_gap_fraction < 0.02).all()),
        "target_reach_100_percent": bool((table.target_reach_fraction == 1.0).all()),
        "voltage_safe": bool((table.maximum_voltage_v <= gates["maximum_voltage_v"]).all()),
        "temperature_safe": bool((table.maximum_temperature_c <= gates["maximum_average_temperature_c"]).all()),
        "slew_safe": bool((table.maximum_current_step_a <= gates["maximum_current_step_a"] + gates["numerical_tolerance"]).all()),
        "zero_guard_exceedance": bool((table.guard_exceedance_count == 0).all()),
        "zero_empty_interval": bool((table.empty_interval_count == 0).all()),
        "speedup_above_100": bool((table.speedup > 100.0).all()),
    }
    success = bool(all(checks.values()))
    payload = {"phase": "Phase 7C-R3 frozen ANN", "trajectory_count": 240, "seed_metrics": seed_rows, "checks": checks, "success": success, "frozen_model_hashes": models, "r2f5_manifest_sha256": verify_r2f5(config, root)["manifest_sha256"], "r3_state_hashes": safe["state_freeze"]["files"], "decision": {"level4_authorized": success, "level4_route": "performance_optimization" if success else "thermal_aware_ann_repair", "level4_entered": False, "ann_retrained": False}}
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_phase7cr3(config: Phase7CR3Config, root: Path, stage: str, resume: bool = False) -> dict[str, Any]:
    if stage == "prepare":
        return prepare_states(config, root)
    if stage == "safe-mpc":
        return run_safe_mpc(config, root, resume)
    if stage == "frozen-ann":
        return run_frozen_ann(config, root, resume)
    raise RuntimeError("Unknown R3 stage")
