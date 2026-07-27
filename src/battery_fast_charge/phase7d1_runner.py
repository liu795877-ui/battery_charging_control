"""Pre-register and freeze Level 4-1 development and validation states."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .phase7cr2f2_runner import _state_columns
from .phase7cr2f3_runner import _build_state_set
from .phase7d1_config import Phase7D1Config


CONFIG_RELATIVE = "configs/phase7d1_performance_optimization.yaml"


def _sha256(path: Path, normalize_text: bool = False) -> str:
    payload = path.read_bytes()
    if normalize_text:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _verify_level4_entry(config: Phase7D1Config, root: Path) -> dict[str, str]:
    sources = config.section("sources")
    metrics_path = root / sources["phase7d_metrics"]
    manifest_path = root / sources["phase7d_manifest"]
    if _sha256(metrics_path) != sources["phase7d_metrics_sha256"]:
        raise RuntimeError("Level 4-0 metrics changed")
    if _sha256(manifest_path) != sources["phase7d_manifest_sha256"]:
        raise RuntimeError("Level 4-0 manifest changed")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not metrics["success"] or not metrics["decision"]["level4_entered"]:
        raise RuntimeError("Level 4 entry is not authorized")
    if not manifest["level4_1_authorized"]:
        raise RuntimeError("Level 4-1 is not authorized")
    return {
        "phase7d_metrics_sha256": _sha256(metrics_path),
        "phase7d_manifest_sha256": _sha256(manifest_path),
    }


def _excluded_states(config: Phase7D1Config, root: Path) -> np.ndarray:
    frames = [
        pd.read_csv(root / relative)
        for relative in config.section("sources")["excluded_r3t2_state_files"]
    ]
    return pd.concat(frames, ignore_index=True)[_state_columns()].to_numpy(float)


def prepare_phase7d1_states(config: Phase7D1Config, root: Path) -> dict[str, Any]:
    authorization = _verify_level4_entry(config, root)
    output_dir = root / config.section("output")["data_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = output_dir / "state_freeze.json"
    if freeze_path.exists():
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
        for name, record in payload["files"].items():
            if _sha256(output_dir / name) != record["sha256"]:
                raise RuntimeError(f"Level 4-1 frozen state hash mismatch: {name}")
        return payload

    datasets = config.section("datasets")
    temperatures = tuple(int(value) for value in datasets["temperatures_c"])
    existing = _excluded_states(config, root)
    role_frames: dict[str, pd.DataFrame] = {}
    files: dict[str, Any] = {}
    for role in ("development", "internal_validation"):
        frames = []
        for temperature in temperatures:
            frame, existing = _build_state_set(
                config, root, role, temperature, existing
            )
            frame["trajectory_id"] = frame.trajectory_id.str.replace(
                "phase7cr2f3_", "phase7d1_", regex=False
            )
            name = f"{role}_initial_states_{temperature}c.csv"
            path = output_dir / name
            frame.to_csv(path, index=False)
            files[name] = {
                "sha256": _sha256(path),
                "trajectory_count": int(len(frame)),
                "temperature_c": temperature,
                "design_seed": int(frame.design_seed.iloc[0]),
                "zero_residual_count": int(
                    np.isclose(frame.initial_measured_residual_v, 0.0).sum()
                ),
            }
            frames.append(frame)
        role_frames[role] = pd.concat(frames, ignore_index=True)

    development = role_frames["development"][_state_columns()].to_numpy(float)
    validation = role_frames["internal_validation"][_state_columns()].to_numpy(float)
    if np.any(
        np.all(
            np.isclose(development[:, None, :], validation[None, :, :], atol=1e-14),
            axis=2,
        )
    ):
        raise RuntimeError("Level 4-1 development and validation states overlap")

    contract = config.section("optimization_contract")
    if len(role_frames["development"]) != contract["development_trajectory_count"]:
        raise RuntimeError("Unexpected Level 4-1 development count")
    if len(role_frames["internal_validation"]) != contract[
        "internal_validation_trajectory_count"
    ]:
        raise RuntimeError("Unexpected Level 4-1 validation count")
    payload = {
        "phase": config.payload["phase"],
        "status": "states_frozen_before_any_level4_1_rollout",
        "authorization": authorization,
        "config_sha256": _sha256(root / CONFIG_RELATIVE, True),
        "files": files,
        "development_internal_validation_isolated": True,
        "r3t2_states_excluded": True,
        "r3_confirmation_used_for_tuning": False,
        "rollouts_started": False,
        "optimization_parameters_selected": False,
        "internal_validation_consumed": False,
        "independent_confirmation_created": False,
    }
    freeze_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload

