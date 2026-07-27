"""Level 4-3 independent state freeze and final confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .phase7cr2f2_runner import _state_columns
from .phase7cr2f3_runner import _build_state_set
from .phase7cr3_runner import _model_hashes
from .phase7d2_runner import _metric_table, _run_variant, _summary_row
from .phase7d3_config import Phase7D3Config


CONFIG_RELATIVE = "configs/phase7d3_final_confirmation.yaml"
RUNNER_RELATIVE = "src/battery_fast_charge/phase7d3_runner.py"


def _sha256(path: Path, normalize_text: bool = False) -> str:
    payload = path.read_bytes()
    if normalize_text:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _verify_level4_2(config: Phase7D3Config, root: Path) -> dict[str, Any]:
    sources = config.section("sources")
    selection_path = root / sources["phase7d2_selection"]
    validation_path = root / sources["phase7d2_internal_validation"]
    guard_path = root / sources["frozen_voltage_guards"]
    for path, expected, label in (
        (selection_path, sources["phase7d2_selection_sha256"], "selection"),
        (validation_path, sources["phase7d2_internal_validation_sha256"], "validation"),
        (guard_path, sources["frozen_voltage_guards_sha256"], "voltage guards"),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"Frozen Level 4-2 {label} changed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not selection["success"] or not validation["success"]:
        raise RuntimeError("Level 4-2 did not authorize final confirmation")
    if not validation["independent_confirmation_authorized"]:
        raise RuntimeError("Independent confirmation is not authorized")
    selected_guard = float(selection["selected_temperature_guard_c"])
    if selected_guard != float(config.section("thermal_candidates")["selected_temperature_guard_c"]):
        raise RuntimeError("Selected thermal guard changed")
    return {
        "selection_sha256": _sha256(selection_path),
        "validation_sha256": _sha256(validation_path),
        "voltage_guard_sha256": _sha256(guard_path),
        "selection": selection,
        "guards": json.loads(guard_path.read_text(encoding="utf-8"))["guards"],
    }


def _excluded_states(config: Phase7D3Config, root: Path) -> np.ndarray:
    frames = [pd.read_csv(root / relative) for relative in config.section("sources")["excluded_state_files"]]
    return pd.concat(frames, ignore_index=True)[_state_columns()].to_numpy(float)


def prepare_states(config: Phase7D3Config, root: Path) -> dict[str, Any]:
    verification = _verify_level4_2(config, root)
    output = config.section("output")
    data_dir = root / output["data_directory"]
    data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = data_dir / "state_freeze.json"
    if freeze_path.exists():
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
        for name, record in payload["files"].items():
            if _sha256(data_dir / name) != record["sha256"]:
                raise RuntimeError(f"Level 4-3 state hash mismatch: {name}")
        return payload
    existing = _excluded_states(config, root)
    files: dict[str, Any] = {}
    frames = []
    for temperature in config.section("datasets")["temperatures_c"]:
        frame, existing = _build_state_set(config, root, "confirmation", int(temperature), existing)
        frame["trajectory_id"] = frame.trajectory_id.str.replace("phase7cr2f3_", "phase7d3_", regex=False)
        name = f"confirmation_initial_states_{int(temperature)}c.csv"
        path = data_dir / name
        frame.to_csv(path, index=False)
        files[name] = {
            "sha256": _sha256(path),
            "trajectory_count": int(len(frame)),
            "temperature_c": int(temperature),
            "design_seed": int(frame.design_seed.iloc[0]),
            "zero_residual_count": int(np.isclose(frame.initial_measured_residual_v, 0.0).sum()),
        }
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    expected = 2 * int(config.section("datasets")["confirmation_count_per_temperature"])
    if len(combined) != expected:
        raise RuntimeError("Unexpected Level 4-3 confirmation state count")
    payload = {
        "phase": config.payload["phase"],
        "status": "states_frozen_before_final_confirmation",
        "files": files,
        "excluded_prior_state_count": int(len(_excluded_states(config, root))),
        "prior_states_excluded": True,
        "parameters_frozen_before_state_generation": True,
        "config_sha256": _sha256(root / CONFIG_RELATIVE, True),
        "runner_sha256": _sha256(root / RUNNER_RELATIVE, True),
        "selection_sha256": verification["selection_sha256"],
        "validation_sha256": verification["validation_sha256"],
        "confirmation_started": False,
        "confirmation_used_for_retuning": False,
        "level4_completed": False,
    }
    freeze_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _rows(config: Phase7D3Config, root: Path) -> list[dict[str, Any]]:
    data_dir = root / config.section("output")["data_directory"]
    return pd.concat(
        [pd.read_csv(data_dir / f"confirmation_initial_states_{temperature}c.csv") for temperature in config.section("datasets")["temperatures_c"]],
        ignore_index=True,
    ).to_dict(orient="records")


def _minimum_seed_speedup(config: Phase7D3Config, traces: pd.DataFrame) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    mpc_ms = float(config.section("gates")["frozen_mpc_total_mean_ms"])
    for seed, frame in traces[traces.current_a.notna()].groupby("seed"):
        total_ms = 1000.0 * float((frame.teacher_time_s + frame.supervisor_time_s).mean())
        rows.append({"seed": int(seed), "ann_total_mean_ms": total_ms, "end_to_end_speedup": mpc_ms / total_ms})
    return min(row["end_to_end_speedup"] for row in rows), rows


def run_final_confirmation(config: Phase7D3Config, root: Path, resume: bool) -> dict[str, Any]:
    freeze = prepare_states(config, root)
    if freeze["config_sha256"] != _sha256(root / CONFIG_RELATIVE, True):
        raise RuntimeError("Level 4-3 config changed after state freeze")
    if freeze["runner_sha256"] != _sha256(root / RUNNER_RELATIVE, True):
        raise RuntimeError("Level 4-3 runner changed after state freeze")
    verification = _verify_level4_2(config, root)
    output = config.section("output")
    data_dir = root / output["data_directory"]
    result_dir = root / output["result_directory"]
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = _rows(config, root)
    seeds = tuple(int(seed) for seed in config.section("datasets")["confirmation_seeds"])
    thermal = config.section("thermal_candidates")
    baseline_guard = float(thermal["baseline_temperature_guard_c"])
    selected_guard = float(thermal["selected_temperature_guard_c"])
    selected_variant = "thermal_guard_025mc"
    baseline = _run_variant(config, root, "final_confirmation", "baseline", baseline_guard, rows, seeds, verification["guards"], resume)
    selected = _run_variant(config, root, "final_confirmation", selected_variant, selected_guard, rows, seeds, verification["guards"], resume)
    baseline_path = data_dir / "baseline_trajectories.csv"
    selected_path = data_dir / "selected_trajectories.csv"
    baseline.to_csv(baseline_path, index=False)
    selected.to_csv(selected_path, index=False)
    baseline_metrics = _metric_table(config, root, baseline)
    selected_metrics = _metric_table(config, root, selected)
    baseline_metrics_path = data_dir / "baseline_trajectory_metrics.csv"
    selected_metrics_path = data_dir / "selected_trajectory_metrics.csv"
    baseline_metrics.to_csv(baseline_metrics_path, index=False)
    selected_metrics.to_csv(selected_metrics_path, index=False)
    baseline_summary = _summary_row(config, baseline, baseline_metrics, "baseline", baseline_guard)
    selected_summary = _summary_row(config, selected, selected_metrics, selected_variant, selected_guard)
    baseline_30 = baseline_metrics[baseline_metrics.ambient_temperature_c == 30]
    selected_30 = selected_metrics[selected_metrics.ambient_temperature_c == 30]
    improvement_30 = float((baseline_30.charge_time_s.mean() - selected_30.charge_time_s.mean()) / baseline_30.charge_time_s.mean())
    minimum_speedup, seed_timing = _minimum_seed_speedup(config, selected)
    timing_path = data_dir / "selected_seed_timing.csv"
    pd.DataFrame(seed_timing).to_csv(timing_path, index=False)
    contract = config.section("confirmation_contract")
    checks = {
        "baseline_strictly_safe": baseline_summary["strictly_safe"],
        "selected_strictly_safe": selected_summary["strictly_safe"],
        "expected_baseline_trajectory_count": len(baseline_metrics) == int(contract["expected_trajectory_count_per_variant"]),
        "expected_selected_trajectory_count": len(selected_metrics) == int(contract["expected_trajectory_count_per_variant"]),
        "all_five_seeds_present": set(selected_metrics.seed) == set(seeds),
        "overall_charge_time_noninferior": selected_metrics.charge_time_s.mean() <= baseline_metrics.charge_time_s.mean(),
        "30c_improvement_meets_contract": improvement_30 >= float(contract["minimum_30c_mean_charge_time_improvement_fraction"]),
        "all_seed_speedups_above_100": minimum_speedup > float(config.section("gates")["minimum_end_to_end_speedup"]),
        "confirmation_not_used_for_retuning": True,
    }
    success = bool(all(checks.values()))
    payload = {
        "phase": config.payload["phase"],
        "status": "strict_passed" if success else "strict_stop_failed",
        "selected_temperature_guard_c": selected_guard,
        "prediction_horizon_s": float(thermal["prediction_horizon_s"]),
        "baseline_summary": baseline_summary,
        "selected_summary": selected_summary,
        "baseline_30c_mean_charge_time_s": float(baseline_30.charge_time_s.mean()),
        "selected_30c_mean_charge_time_s": float(selected_30.charge_time_s.mean()),
        "confirmation_30c_improvement_fraction": improvement_30,
        "minimum_seed_end_to_end_speedup": minimum_speedup,
        "seed_timing": seed_timing,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "success": success,
        "frozen_model_hashes": _model_hashes(root),
        "confirmation_used_for_retuning": False,
        "ann_retrained": False,
        "decision": {"level4_completed": success, "next_phase_authorized": success},
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = result_dir / "PHASE7D_LEVEL4_中文实验报告.md"
    report_path.write_text(
        "# Phase 7D / Level 4 性能优化最终报告\n\n"
        f"- 状态：{payload['status']}；\n"
        f"- 30 ℃平均充电时间改善：{100 * improvement_30:.4f}%；\n"
        f"- 最高温度：{selected_summary['maximum_temperature_c']:.6f} ℃；\n"
        f"- 最高电压：{selected_summary['maximum_voltage_v']:.6f} V；\n"
        f"- 最低五种子端到端加速：{minimum_speedup:.2f}×；\n"
        "- ANN、电压裕量和300 s预测窗口保持冻结；确认集未用于调参。\n",
        encoding="utf-8",
    )
    artifacts = (
        CONFIG_RELATIVE,
        RUNNER_RELATIVE,
        str((data_dir / "state_freeze.json").relative_to(root)),
        str(baseline_path.relative_to(root)),
        str(selected_path.relative_to(root)),
        str(baseline_metrics_path.relative_to(root)),
        str(selected_metrics_path.relative_to(root)),
        str(timing_path.relative_to(root)),
        str(metrics_path.relative_to(root)),
        str(report_path.relative_to(root)),
    )
    manifest = {
        "phase": config.payload["phase"],
        "status": payload["status"],
        "level4_completed": success,
        "confirmation_used_for_retuning": False,
        "artifacts": {relative: _sha256(root / relative, Path(relative).suffix in {".py", ".yaml", ".md"}) for relative in artifacts},
    }
    (result_dir / "freeze_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload

