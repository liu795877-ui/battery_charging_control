"""Independent R3 confirmation; safe MPC must pass before frozen ANN."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .phase7cr2f2_runner import _sha256, _state_columns
from .phase7cr2f3_runner import _all_historical_values, _build_state_set, _checks, _run_rows, _temperature_stage_counts
from .phase7cr2f5_runner import verify_frozen_r2f4
from .phase7cr2f_config import load_phase7cr2f_config
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


def run_phase7cr3(config: Phase7CR3Config, root: Path, stage: str, resume: bool = False) -> dict[str, Any]:
    if stage == "prepare":
        return prepare_states(config, root)
    if stage == "safe-mpc":
        return run_safe_mpc(config, root, resume)
    raise RuntimeError("Frozen ANN stage is not implemented or authorized yet")
